# frozen_string_literal: true

module Tilework
  module Graph
    # Elementwise fusion (#17 Phase 2).
    #
    # Folds chained elementwise/unary/scalar nodes into one kernel. A five-step
    # chain's four intermediates and ten memory round trips become one read and
    # one write.
    #
    # ## How far it fuses
    #
    # A node is absorbed into its parent **only if it has exactly one
    # consumer**. A value read from two places needs materializing somewhere
    # anyway, so absorbing it saves no round trip and duplicates the
    # expression, computing it twice. This one rule folds every "a into b, b
    # into c" chain into one and safely declines diamonds (reconvergence).
    #
    # ## Values do not change
    #
    # The unfused path repeats "widen to the compute type → operate → write back
    # in the output type" per node. Wrapping each expression-tree node **in a
    # cast to its output type** reproduces that write-back exactly, so results
    # agree bit for bit (conformance's graph_fused_elementwise_matches_the_unfused_path
    # pins this on every backend).
    #
    # The one exception is float16, where intermediates would need re-rounding
    # to half: skipping it changes values between fused and unfused, and doing
    # it undoes the point of fusing. **Nodes with float16 intermediates are not
    # fused** (leaves and roots may be float16 — they are real buffers and
    # round anyway).
    #
    # ## :cpu fuses too
    #
    # The issue's original plan excluded :cpu (no dialect), which would have
    # made "fused == unfused" checkable only on hardware (CI has no GPU). core
    # builds the expression tree, this file renders dialect source and
    # bridge.ops_codegen.fused renders Ruby source, making **:cpu the oracle
    # for fusion**.
    module Fusion
      # One expression-tree node.
      #   kind    :buffer (input buffer) / :scalar / :binary / :compare / :math /
      #           :abs / :cast / :select
      #   dtype   this node's output type (= the type it would be written back
      #           in without fusion)
      #   compute the type used for computation
      #   op      the operator or function name
      #   index   the argument position for :buffer / :scalar
      #   map     the index remap for :buffer (what a folded broadcast leaves
      #           behind); nil means plain a[i]
      Expr = Struct.new(:kind, :dtype, :compute, :op, :children, :index, :map, keyword_init: true)

      # A cluster that becomes one kernel.
      #   root    the node owning the output (materialized)
      #   members node ids absorbed into root (not materialized)
      #   inputs  nodes of buffers coming from outside (in kernel-argument order)
      #   maps    index remaps in the same order as inputs (folded broadcasts);
      #           nil for inputs read plainly
      #   scalars scalars passed as kernel arguments
      #   reductions row-wise reductions (in dependency order); empty for an
      #           elementwise cluster
      Cluster = Struct.new(:root, :members, :inputs, :maps, :scalars, :expr, :reductions,
                           keyword_init: true) do
        def length = root.length
        def shape = root.shape
        def dtype = root.dtype

        # A cluster containing row-wise reductions launches over **rows**, not elements.
        def row_wise? = !reductions.empty?
        def cols = shape.last
        def rows = length / cols
      end

      # The lower bound for trying row-wise fusion. One thread owns one row, so
      # the row count is the parallelism — with few rows, per-element
      # parallelism wins even at the cost of more kernels. **This threshold is
      # not measured on hardware**: it is provisionally one block's worth of
      # threads, so fix it once measured (Graph.row_fusion = false turns the
      # path off).
      MIN_ROWS = Ops::BLOCK

      module_function

      # Plans fusion. Returns [root.id => Cluster, the set of absorbed node ids].
      #
      # pinned receives nodes named by ordering (WAR) edges. Absorbing one moves
      # "the moment it finished" to root, leaving room to overtake a mutable
      # state write, so they are excluded from fusion.
      def plan(order:, output_ids:, consumers:, pinned:)
        absorbed = {}
        members = Hash.new { |hash, key| hash[key] = [] }

        order.each do |node|
          next unless fusable?(node)

          node.inputs.each do |input|
            # Pull a row-wise reduction into the cluster along with its subtree;
            # if it fits, "reduce → stretch → use" becomes one kernel.
            if (found = row_reduce_behind(input, node, output_ids, consumers, pinned))
              absorb_row_reduce(found, input, node, absorbed, members, consumers, output_ids, pinned)
              next
            end
            # A broadcast **computes nothing**, so it can fold however many
            # readers it has (each just reads the unstretched buffer remapped).
            # The single-consumer rule, which exists to avoid duplicated
            # expressions, does not apply.
            if foldable_broadcast?(input, output_ids, consumers, pinned)
              absorbed[input.id] = true
              members[node.id] |= [input.id]
              next
            end
            next unless absorbable?(input, node, output_ids, consumers, pinned)
            next if absorbed[input.id]

            absorbed[input.id] = true
            members[node.id] = members[node.id] + (members.delete(input.id) || []) + [input.id]
          end
        end

        clusters = {}
        order.each do |node|
          next if absorbed[node.id] || members[node.id].empty?

          clusters[node.id] = build_cluster(node, members[node.id], absorbed)
        end
        [clusters, absorbed]
      end

      # Whether an op can fuse is in the op table (only ops whose launch domain
      # is the element count alone).
      def fusable?(node) = Op.fetch(node.op).fusable?

      def absorbable?(input, node, output_ids, consumers, pinned)
        return false unless fusable?(input)
        # With half-precision intermediates, values change between fused and unfused unless rounding is reproduced.
        return false if Ops::LOW_PRECISION_DTYPES.include?(input.dtype)
        # A different shape is a different launch domain (a different kernel even with the same element count).
        return false unless input.shape == node.shape
        return false if output_ids.include?(input.id)
        return false if pinned.include?(input.id)

        # A value read from two or more places stays materialized even if
        # absorbed, and its expression would be duplicated. Counting edges also
        # rules out self-references like a * a.
        consumers[input.id].length == 1
      end

      # --- Row-wise fusion ---------------------------------------------------
      #
      # Folds the "reduce a row, spread it back over that row" shape (the
      # skeleton of softmax / layernorm / rmsnorm) into one kernel. A reduction
      # always comes back through a broadcast, so that broadcast is the entry point.
      #
      # One thread owns one row and scans it once per reduction. **No shared
      # memory or barriers**, because :cpu's codegen cannot express them — a
      # block-cooperative fold would create a path CI cannot verify (see the note
      # at the top of backend/cpu/ops_codegen.rb). The hand-written NN kernels
      # cooperate within a block because it is faster; this takes the side that
      # works for general expressions.
      def row_reduce_behind(input, consumer, output_ids, consumers, pinned)
        return nil unless Graph.row_fusion
        return nil unless input.op == :broadcast
        return nil if output_ids.include?(input.id) || pinned.include?(input.id)
        # Exactly one reader. This pulls the reduction and its subtree into the
        # cluster, so if two clusters pulled in the same broadcast, the later
        # one would reference an absorbed node as input and pass a nil buffer at
        # run time. With several readers the broadcast folds as an index and the
        # reduction is materialized (cheaper than redoing it per reader).
        return nil unless consumers[input.id].length == 1
        return nil unless consumers[input.id].all? { |reader| fusable?(reader) && reader.shape == input.shape }

        # An expression touching only the folded value may sit between the
        # reduction and the broadcast (rmsnorm's `sqrt(mean + eps)`). It is
        # constant within a row, so it joins the cluster too. Only expressions
        # combining **with scalars only** qualify — one reading a buffer would
        # need the row index.
        chain = []
        node = input.inputs.first
        while node.op != :reduce
          return nil unless post_reduce_link?(node, input.inputs.first.shape, consumers, output_ids, pinned)

          chain << node
          node = node.inputs.first
        end
        return nil unless row_reduce?(node, consumer.shape)
        return nil if output_ids.include?(node.id) || pinned.include?(node.id)
        return nil unless consumers[node.id].length == 1

        [node, chain]
      end

      # Nodes allowed in the cluster as expressions over the folded value.
      # Having exactly one input (unary, cast, binary with a scalar) is what
      # guarantees they read no buffer.
      def post_reduce_link?(node, shape, consumers, output_ids, pinned)
        return false unless %i[unary cast binary].include?(node.op)
        return false if node.op == :binary && !node.attrs.key?(:scalar)
        return false unless node.inputs.length == 1 && node.shape == shape
        return false if Ops::LOW_PRECISION_DTYPES.include?(node.dtype)
        return false if output_ids.include?(node.id) || pinned.include?(node.id)

        consumers[node.id].length == 1
      end

      # A keepdims reduction over the last dim whose pre-reduction shape matches
      # the cluster's. Not folded while rows are few (see MIN_ROWS above).
      def row_reduce?(node, cluster_shape)
        return false unless node.op == :reduce
        return false if Ops::LOW_PRECISION_DTYPES.include?(node.dtype)
        return false unless node.attrs[:keepdims]

        source = node.inputs.first
        axis = node.attrs[:axis]
        return false if axis.nil? || axis != source.shape.length - 1
        return false unless source.shape == cluster_shape

        cluster_shape[0..-2].inject(1, :*) >= MIN_ROWS
      end

      # Puts the reduction and the elementwise ops before it into the cluster.
      # When what comes before cannot be absorbed (read from two places, etc.)
      # it stops there and the reduction reads that materialized value.
      def absorb_row_reduce(found, broadcast, root, absorbed, members, consumers, output_ids, pinned)
        reduce, chain = found
        [broadcast, reduce, *chain].each do |node|
          absorbed[node.id] = true
          members[root.id] |= [node.id]
        end

        stack = reduce.inputs.dup
        while (node = stack.pop)
          next if absorbed[node.id]
          next unless fusable?(node) && node.shape == reduce.inputs.first.shape
          next if Ops::LOW_PRECISION_DTYPES.include?(node.dtype)
          next if output_ids.include?(node.id) || pinned.include?(node.id)
          next unless consumers[node.id].length == 1

          absorbed[node.id] = true
          members[root.id] |= (members.delete(node.id) || []) + [node.id]
          stack.concat(node.inputs)
        end
      end

      # If every reader folds, the broadcast need not be materialized. If even
      # one needs the real buffer (a reduction, matmul, :nn, or the output
      # itself), it must be materialized for that reader, so it is not folded
      # — folding and materializing both would hold the same value two ways,
      # adding lifetime bookkeeping for no gain.
      def foldable_broadcast?(input, output_ids, consumers, pinned)
        return false unless Op.fetch(input.op).index_map?
        return false if Ops::LOW_PRECISION_DTYPES.include?(input.dtype)
        return false if output_ids.include?(input.id) || pinned.include?(input.id)

        readers = consumers[input.id]
        readers.length > 1 && readers.all? { |reader| fusable?(reader) && reader.shape == input.shape }
      end

      def build_cluster(root, member_ids, absorbed)
        inside = (member_ids + [root.id]).to_h { |id| [id, true] }
        inputs = []
        maps = []
        scalars = []
        reductions = []
        expr = build_expr(root, inside, [inputs, maps, reductions], scalars, absorbed)
        Cluster.new(root: root, members: member_ids, inputs: inputs, maps: maps,
                    scalars: scalars, expr: expr, reductions: reductions)
      end

      # Walks the tree from the root into an expression; values from outside
      # the cluster become buffer arguments.
      def build_expr(node, inside, args, scalars, absorbed)
        return leaf(node, args) unless inside[node.id]
        # A broadcast becomes **an index, not an expression**: the unstretched
        # buffer is the argument and the way it is read (which dims repeat how
        # often) folds into the index expression. That removes the kernel
        # writing out the stretched value and the round trip reading it back.
        return remap_leaf(node, inside, args, scalars, absorbed) if Op.fetch(node.op).index_map?
        return reduce_expr(node, inside, args, scalars, absorbed) if node.op == :reduce

        children = node.inputs.map { |input| build_expr(input, inside, args, scalars, absorbed) }
        case node.op
        when :unary  then unary_expr(node, children.first)
        when :binary then binary_expr(node, children, scalars)
        when :cast   then cast_expr(node, children.first)
        when :where  then select_expr(node, children, scalars)
        else raise Error, "cannot fuse #{node.op.inspect}"
        end
      end

      # The same node read with a different map is a separate argument. Plain
      # and stretched reads can share one expression, so identity is [node, map].
      def leaf(node, args, map = nil)
        inputs, maps, = args
        index = inputs.each_index.find { |i| inputs[i].equal?(node) && maps[i] == map }
        unless index
          index = inputs.length
          inputs << node
          maps << map
        end
        Expr.new(kind: :buffer, dtype: node.dtype, compute: node.dtype, index: index, map: map, children: [])
      end

      # When what is behind the broadcast is inside the cluster (= a row-wise
      # reduction was pulled in), it becomes **that row's reduced value**, not an
      # index remap. It is constant within the row, so no index is needed.
      def remap_leaf(node, inside, args, scalars, absorbed)
        source = node.inputs.first
        return build_expr(source, inside, args, scalars, absorbed) if inside[source.id]

        leaf(source, args, Kernels.remap_dims(node))
      end

      # A row-wise reduction. The expression is the pre-reduction value, and the
      # result is computed once per row under the name red<k>.
      def reduce_expr(node, inside, args, scalars, absorbed)
        reductions = args[2]
        expr = build_expr(node.inputs.first, inside, args, scalars, absorbed)
        index = reductions.length
        reductions << { kind: node.attrs[:kind], dtype: node.dtype, expr: expr }
        Expr.new(kind: :reduced, dtype: node.dtype, compute: node.dtype, index: index, children: [])
      end

      def unary_expr(node, child)
        fn = node.attrs[:fn]
        kind = fn == :abs ? :abs : :math
        Expr.new(kind: kind, dtype: node.dtype, compute: node.dtype == :int32 ? :int32 : :float32,
                 op: fn, children: [child])
      end

      def binary_expr(node, children, scalars)
        kind = node.attrs[:kind] == :compare ? :compare : :binary
        compute = compute_dtype(node)
        operands = children
        if node.attrs.key?(:scalar)
          index = scalars.length
          value = compute == :float32 ? node.attrs[:scalar].to_f : node.attrs[:scalar]
          scalars << { value: value, dtype: compute }
          scalar = Expr.new(kind: :scalar, dtype: compute, compute: compute, index: index, children: [])
          operands = node.attrs[:reversed] ? [scalar, children.first] : [children.first, scalar]
        end
        Expr.new(kind: kind, dtype: node.dtype, compute: compute, op: node.attrs[:op], children: operands)
      end

      # A cast is exactly "a cast to the output type", the same as the unfused
      # path's write-back, so in the tree it wraps like any other node.
      def cast_expr(node, child)
        Expr.new(kind: :cast, dtype: node.dtype, compute: node.dtype, children: [child])
      end

      # Select by condition. Children are [cond, a, b]; a side that is a scalar
      # goes in as a kernel scalar argument (like a binary op's scalar).
      def select_expr(node, children, scalars)
        rest = children.drop(1)
        operands = %i[a b].map do |key|
          value = node.attrs[:"#{key}_scalar"]
          next rest.shift if value.nil?

          index = scalars.length
          scalars << { value: value, dtype: node.dtype }
          Expr.new(kind: :scalar, dtype: node.dtype, compute: node.dtype, index: index, children: [])
        end
        Expr.new(kind: :select, dtype: node.dtype, compute: node.dtype,
                 children: [children.first, *operands])
      end

      # Ops "widens to the compute type, then operates". A comparison outputs
      # int32 but computes in its operands' type, so the output type cannot
      # decide it.
      def compute_dtype(node)
        return node.dtype unless node.attrs[:kind] == :compare

        operands = node.inputs.map(&:dtype)
        operands << (node.attrs[:scalar].is_a?(Integer) ? :int32 : :float32) if node.attrs.key?(:scalar)
        operands.all?(:int32) ? :int32 : :float32
      end

      # --- Dialect source ----------------------------------------------------

      # A cluster with row-wise reductions. One thread owns one row, scanning it
      # once per reduction and then once more to write. **Intermediates are
      # recomputed rather than kept** — as in the hand-written NN kernels
      # (softmax computes exp twice); not holding a row's worth of storage is
      # this form's advantage.
      def row_dialect_source(cluster)
        params = kernel_params(cluster)
        params << "constant int& rows"
        params << "constant int& cols"

        passes = cluster.reductions.each_with_index.map do |reduction, k|
          acc = Ops::CTYPE.fetch(reduction[:dtype])
          <<~PASS.chomp
                #{acc} acc#{k} = (#{acc})0;
                for (int c = 0; c < cols; ++c) {
                  int i = base + c;
                  #{acc} v = (#{acc})#{dialect_expr(reduction[:expr])};
                  acc#{k} = (c == 0) ? v : (#{combine(reduction[:kind], "acc#{k}", 'v')});
                }
                #{acc} red#{k} = #{finish(reduction[:kind], "acc#{k}", acc)};
          PASS
        end

        <<~KERNEL
          GPU_KERNEL void row_fused(GPU_ARGS,
                                    #{params.join(",\n                        ")}) {
            int r = GLOBAL_ID_X;
            if (r < rows) {
              int base = r * cols;
          #{passes.join("\n")}
              for (int c = 0; c < cols; ++c) {
                int i = base + c;
                out[i] = #{dialect_expr(cluster.expr)};
              }
            }
          }
        KERNEL
      end

      # How a reduction folds. The order is "from the first element", the same
      # as the axis-reduction kernel (Kernels.axis_reduce), so fused and unfused
      # give the same value.
      def combine(kind, acc, value)
        case kind
        when :sum, :mean then "#{acc} + #{value}"
        when :min        then "#{value} < #{acc} ? #{value} : #{acc}"
        when :max        then "#{value} > #{acc} ? #{value} : #{acc}"
        else raise Error, "cannot fuse the reduction #{kind.inspect}"
        end
      end

      def finish(kind, acc, ctype) = kind == :mean ? "#{acc} / (#{ctype})cols" : acc

      def kernel_params(cluster)
        params = cluster.inputs.each_with_index.map do |node, i|
          "device const #{Ops::CTYPE.fetch(node.dtype)}* a#{i}"
        end
        params += cluster.scalars.each_with_index.map do |scalar, i|
          "constant #{Ops::CTYPE.fetch(scalar[:dtype])}& s#{i}"
        end
        params << "device #{Ops::CTYPE.fetch(cluster.dtype)}* out"
        params
      end

      # Dialect source for a fused kernel. The name follows from the expression's
      # shape, so clusters of the same shape collapse to one kernel in Kernel's
      # SHA cache (32 layers with the same expression compile once).
      def dialect_source(cluster)
        return row_dialect_source(cluster) if cluster.row_wise?

        params = kernel_params(cluster)
        params << "constant int& n"

        <<~KERNEL
          GPU_KERNEL void fused(GPU_ARGS,
                                #{params.join(",\n                      ")}) {
            int i = GLOBAL_ID_X;
            if (i < n) out[i] = #{dialect_expr(cluster.expr)};
          }
        KERNEL
      end

      # Wraps each node **in a cast to its output type**, reproducing inside the
      # expression the write-back the unfused path performed.
      def dialect_expr(expr)
        ct = Ops::CTYPE.fetch(expr.dtype)
        case expr.kind
        when :buffer  then "(#{ct})a#{expr.index}[#{index_expr(expr.map)}]"
        when :reduced then "red#{expr.index}"
        when :scalar  then "s#{expr.index}"
        when :binary  then "(#{ct})(#{operand(expr, 0)} #{expr.op} #{operand(expr, 1)})"
        when :compare then "(#{ct})((#{operand(expr, 0)} #{expr.op} #{operand(expr, 1)}) ? 1 : 0)"
        when :math    then "(#{ct})#{expr.op}((float)#{dialect_expr(expr.children.first)})"
        when :cast    then "(#{ct})#{dialect_expr(expr.children.first)}"
        when :select
          "(#{ct})(#{dialect_expr(expr.children[0])} ? #{operand(expr, 1)} : #{operand(expr, 2)})"
        when :abs
          fn = expr.dtype == :int32 ? "abs" : "fabs"
          "(#{ct})#{fn}(#{dialect_expr(expr.children.first)})"
        else raise Error, "unknown fused expression #{expr.kind.inspect}"
        end
      end

      # Index remap. Splits the output index i per dim and rebuilds the input
      # index, dropping stretched dims (stride 0). **It is a single
      # expression**, so it fits inside a kernel body that cannot hold statements.
      #
      #   [d] -> [n, d]     ->  i % d
      #   [n, 1] -> [n, d]  ->  (i / d) % n
      def index_expr(map)
        return "i" if map.nil?

        total = map.inject(1) { |acc, (size, _)| acc * size }
        suffix = 1
        terms = map.reverse.filter_map do |size, stride|
          span = suffix
          suffix *= size
          # A stretched dim does not affect the input index, so its term drops.
          next if stride.zero?

          base = span == 1 ? "i" : "(i / #{span})"
          # The outermost dim is already in range after the division, so no remainder is needed.
          term = suffix == total ? base : "(#{base} % #{size})"
          stride == 1 ? term : "#{term} * #{stride}"
        end
        terms.empty? ? "0" : terms.join(" + ")
      end

      # Operands are widened to the parent's compute type first (as in Ops.binary).
      def operand(expr, index)
        child = expr.children[index]
        rendered = dialect_expr(child)
        return rendered if child.dtype == expr.compute

        "(#{Ops::CTYPE.fetch(expr.compute)})#{rendered}"
      end

      # No need for a different name per cluster (the cache key is the source's SHA).
      def kernel_name(_cluster) = "fused"
    end
  end
end
