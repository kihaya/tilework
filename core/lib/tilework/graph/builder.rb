# frozen_string_literal: true

module Tilework
  module Graph
    # Records graph inputs, operations, mutable state, and outputs without touching a device.
    class Builder
      # float16 appears **only as a storage type**, as in Ops: arithmetic always
      # widens to float and rounds only on write-back. The rule differs per
      # expression and is mirrored in result_dtype / unary_math / reduce /
      # matmul — if this drifts from Ops, the same expression gets a different
      # type only in the graph.
      #
      # Whether a dtype can be allocated depends on the backend, so only typing
      # is checked here; Plan checks against capabilities[:dtypes] at compile time.
      DTYPES = %i[float32 int32 float16 bfloat16].freeze

      attr_reader :nodes, :outputs, :states, :writers

      def initialize(differentiable: false)
        # A differentiable graph records high-level ops as their decompositions,
        # since gradients follow from primitive vjps alone (docs/autograd.md).
        @differentiable = differentiable
        @nodes = []
        @cse = {}
        @outputs = []
        @inputs = {}
        @states = []
        # Version node id -> nodes overwriting that version; for linearity checks and WAR edges.
        @writers = Hash.new { |h, k| h[k] = [] }
        @random_draws = 0
        # Regions rebuilt for backward (g.checkpoint).
        @checkpoints = []
      end

      # --- Inputs and constants ---

      # Declares a named value supplied to {Plan#run}.
      #
      # @param name [Symbol, String] binding name
      # @param shape [Array<Integer>] logical tensor shape
      # @param dtype [:float32, :float16, :int32]
      # @return [Tensor]
      def input(name, shape, dtype: :float32)
        name = name.to_sym
        raise Error, "input #{name.inspect} is already declared" if @inputs.key?(name)
        # plan.run(seed:) feeds RNG keys, so :seed cannot be an input name.
        raise Error, "input name :seed is reserved for the plan's RNG seed" if name == :seed

        check_dtype!(dtype)
        node = record(:input, attrs: { name: name }, dtype: dtype, shape: normalize_shape(shape))
        @inputs[name] = node
        Tensor.new(self, node)
      end

      # Binds an existing buffer as an immutable graph constant.
      #
      # @param buffer [Buffer]
      # @return [Tensor]
      def const(buffer)
        raise TypeError, "const requires a Tilework::Buffer, got #{buffer.class}" unless buffer.is_a?(Buffer)

        check_dtype!(buffer.dtype)
        node = record(:const, attrs: { buffer: buffer }, dtype: buffer.dtype, shape: buffer.shape,
                      cse: [:const, buffer.object_id])
        Tensor.new(self, node)
      end
      alias param const

      # --- Mutable state ---

      # Declares a buffer as mutable state that survives across plan runs.
      #
      # @param buffer [Buffer]
      # @param name [Symbol, String, nil] diagnostic name
      # @return [StateRef] version-zero state reference
      def state(buffer, name: nil)
        raise TypeError, "state requires a Tilework::Buffer, got #{buffer.class}" unless buffer.is_a?(Buffer)

        check_dtype!(buffer.dtype)
        st = State.new(@states.length, buffer, name)
        @states << st
        node = record(:state_init, attrs: { state: st, version: 0 }, dtype: st.dtype, shape: st.shape)
        ref = StateRef.new(self, st, 0, node)
        st.leaves << ref
        ref
      end

      # Records a write and returns a new state version.
      #
      # @param ref [StateRef]
      # @param value [Tensor]
      # @param at [Integer, Range, Region, Index, nil] target region
      # @return [StateRef]
      #
      #   kv1 = g.write_into(kv0, at: g.index(:pos), value: k_new)
      #
      # at: takes an Integer / Range (outermost dim), a Region, or a run-time
      # index. One version may be written twice only when the regions are
      # **provably disjoint** (they must then be rejoined with g.merge).
      def write_into(ref, value:, at: nil)
        ref = check_ref!(ref)
        region = to_region(ref, at)
        value = check_tensor!(value, "write_into value")
        st = ref.state

        unless value.dtype == st.dtype
          raise Error, "cannot write a #{value.dtype} value into #{st.name} (#{st.dtype})"
        end
        unless value.length == region.length
          raise Error, "value has #{value.length} elements but #{region.inspect} covers #{region.length}"
        end

        @writers[ref.node.id].each do |other|
          next if region.disjoint?(other.region)

          raise Error, "#{st.name} v#{ref.version} is written twice by regions that may overlap " \
                       "(#{other.region.inspect} and #{region.inspect}); a version may only fork into " \
                       "provably disjoint writes"
        end

        version = st.next_version
        node = record(:state_write, inputs: [ref.node, value.node],
                      attrs: { state: st, region: region, version: version },
                      dtype: st.dtype, shape: st.shape)
        @writers[ref.node.id] << node
        st.leaves.delete_if { |leaf| leaf.node.equal?(ref.node) }
        StateRef.new(self, st, version, node).tap { |new_ref| st.leaves << new_ref }
      end

      # Reads all or part of a state version as a tensor.
      #
      # @param ref [StateRef]
      # @param at [Integer, Range, Region, Index, nil]
      # @return [Tensor]
      def read(ref, at = nil)
        ref = check_ref!(ref)
        region = to_region(ref, at)
        node = record(:state_read, inputs: [ref.node],
                      attrs: { state: ref.state, region: region, version: ref.version },
                      dtype: ref.state.dtype, shape: region.shape,
                      # The version is part of the value's identity; that alone
                      # keeps reads on either side of a write from merging in CSE.
                      cse: [:state_read, ref.state.id, ref.version, region.key])
        Tensor.new(self, node)
      end

      # Rejoins writes to disjoint regions. **It writes no memory** (it only
      # places a join point in the ordering), so it carries no WAR constraint.
      def merge(*refs)
        refs = refs.flatten.map { |r| check_ref!(r) }
        raise ArgumentError, "merge needs at least two versions" if refs.length < 2

        st = refs.first.state
        unless refs.all? { |r| r.state.equal?(st) }
          raise Error, "cannot merge versions of different states (#{refs.map { |r| r.state.name }.uniq.join(', ')})"
        end
        refs.each do |r|
          next if st.leaves.any? { |leaf| leaf.node.equal?(r.node) }

          raise Error, "#{st.name} v#{r.version} is not a current version of #{st.name}; " \
                       "only the versions produced by the latest writes can be merged"
        end

        version = st.next_version
        node = record(:state_merge, inputs: refs.map(&:node),
                      attrs: { state: st, version: version },
                      dtype: st.dtype, shape: st.shape)
        st.leaves.reject! { |leaf| refs.any? { |r| r.node.equal?(leaf.node) } }
        StateRef.new(self, st, version, node).tap { |new_ref| st.leaves << new_ref }
      end

      # Creates a runtime outer-dimension index binding.
      #
      # @param name [Symbol, String] binding name passed to {Plan#run}
      # @param count [Integer] contiguous outer-dimension elements
      # @return [Index]
      #
      #   g.write_into(kv, at: g.index(:pos), value: k)   # plan.run(pos: 7)
      def index(name, count: 1)
        Index.new(name.to_sym, count)
      end

      # --- Operations ---

      def arithmetic(op, tensor, other, reversed: false)
        case other
        when Tensor
          dtype = result_dtype(tensor.dtype, other.dtype)
          binary_node(op, tensor, other, dtype: dtype, kind: :arith)
        when Numeric
          dtype = result_dtype(tensor.dtype, other)
          scalar_node(op, tensor, other, dtype: dtype, kind: :arith, reversed: reversed)
        else
          raise TypeError, "cannot combine Tilework::Graph::Tensor with #{other.class}"
        end
      end

      def compare(op, tensor, other)
        case other
        when Tensor  then binary_node(op, tensor, other, dtype: :int32, kind: :compare)
        when Numeric then scalar_node(op, tensor, other, dtype: :int32, kind: :compare, reversed: false)
        else
          raise TypeError, "cannot compare Tilework::Graph::Tensor with #{other.class}"
        end
      end

      def unary_math(fn, tensor)
        # As in Ops.unary_math, half stays half (the point is saving bandwidth,
        # so widening to float32 here would defeat it).
        dtype = Ops::LOW_PRECISION_DTYPES.include?(tensor.dtype) ? tensor.dtype : :float32
        node = record(:unary, inputs: [tensor.node], attrs: { fn: fn },
                      dtype: dtype, shape: tensor.shape, cse: [:unary, fn, dtype])
        Tensor.new(self, node)
      end

      def unary_abs(tensor)
        node = record(:unary, inputs: [tensor.node], attrs: { fn: :abs },
                      dtype: tensor.dtype, shape: tensor.shape, cse: [:unary, :abs, tensor.dtype])
        Tensor.new(self, node)
      end

      # Reductions. With axis: only that axis folds; without it everything folds
      # to shape [1].
      #
      # Eager `buffer.sum` returns a host number, which would end the plan
      # there (reading back to the host drains the queue). A graph needs to
      # carry it as a value, so a whole-buffer reduction takes Ops' device: true
      # path and an axis reduction a kernel with one thread per output element.
      def reduce(kind, tensor, axis: nil, keepdims: false)
        # As in Ops.accumulator_dtype: never accumulate in half (around 2048
        # adding stops increasing the sum).
        dtype = kind == :mean || Ops::LOW_PRECISION_DTYPES.include?(tensor.dtype) ? :float32 : tensor.dtype
        reduce_node(:reduce, kind, tensor, axis, keepdims, dtype)
      end

      # A reduction returning positions rather than values, hence int32.
      def arg_reduce(kind, tensor, axis: nil, keepdims: false)
        reduce_node(:arg_reduce, kind, tensor, axis, keepdims, :int32)
      end

      # Cast. It **only changes the storage type**, not the meaning (float ->
      # int truncates toward zero as in C). float16 appears only as a storage
      # type, as in Ops, so this is for placing conversions explicitly at
      # boundaries, not in the middle of computation.
      def cast(tensor, dtype)
        tensor = check_tensor!(tensor, "cast")
        check_dtype!(dtype)
        return tensor if tensor.dtype == dtype

        node = record(:cast, inputs: [tensor.node], attrs: {}, dtype: dtype,
                      shape: tensor.shape, cse: [:cast, dtype])
        Tensor.new(self, node)
      end

      # Select by condition. cond is a 0/1 mask (a comparison's output as is).
      #
      #   g.where(x > 0, x, 0.0)        # relu
      #   g.where(mask.eq(0), scores, -1e9)
      #
      # a / b may be Tensors or scalars. All three operands broadcast by the
      # same rule as binary ops.
      def where(cond, a, b)
        cond = check_tensor!(cond, "where condition")
        dtype = where_dtype(a, b)
        shape = where_shape!(cond, a, b)

        attrs = {}
        inputs = [broadcast(cond, shape).node]
        { a: a, b: b }.each do |key, value|
          if value.is_a?(Numeric)
            attrs[:"#{key}_scalar"] = dtype == :int32 ? value.to_i : value.to_f
          else
            inputs << broadcast(check_tensor!(value, "where #{key}"), shape).node
          end
        end

        node = record(:where, inputs: inputs, attrs: attrs, dtype: dtype, shape: shape,
                      cse: [:where, dtype, attrs])
        Tensor.new(self, node)
      end

      # The index along an axis itself, so masks and position embeddings can be
      # built inside the graph without bringing constants from the host.
      #
      #   pos = g.iota([n, n], axis: 0)
      #   causal = g.where(pos >= g.iota([n, n], axis: 1), scores, -1e9)
      def iota(shape, axis: -1, dtype: :int32)
        shape = normalize_shape(shape)
        unless %i[int32 float32].include?(dtype)
          raise Error, "iota supports :int32 or :float32 (got #{dtype.inspect})"
        end

        axis = normalize_axis!(axis, shape)
        node = record(:iota, attrs: { axis: axis }, dtype: dtype, shape: shape,
                      cse: [:iota, axis, dtype, shape])
        Tensor.new(self, node)
      end

      # --- Shape operations ---

      # Explicit broadcast. **The rule is NumPy's right alignment**: dims match
      # or one of them is 1.
      #
      # It is inserted as a node at record time, so downstream (fusion,
      # placement, liveness) need not know the implicit rule — they see one
      # ordinary node. Fusion folds it into the consumer's index where it can;
      # otherwise it is materialized.
      def broadcast(tensor, shape)
        tensor = check_tensor!(tensor, "broadcast")
        shape = normalize_shape(shape)
        return tensor if tensor.shape == shape

        unless broadcastable?(tensor.shape, shape)
          raise ArgumentError, "cannot broadcast #{tensor.shape.inspect} to #{shape.inspect}"
        end

        node = record(:broadcast, inputs: [tensor.node], attrs: {}, dtype: tensor.dtype,
                      shape: shape, cse: [:broadcast, shape])
        Tensor.new(self, node)
      end

      # Axis permutation. axes lists, for each output dim d, which input dim it is.
      #
      #   x.permute(1, 0)          # 2D transpose
      #   x.permute(1, 0, 2)       # [n, h, dh] -> [h, n, dh]
      def permute(tensor, *axes)
        tensor = check_tensor!(tensor, "permute")
        axes = axes.first if axes.length == 1 && axes.first.is_a?(Array)
        rank = tensor.shape.length
        unless axes.sort == (0...rank).to_a
          raise ArgumentError, "permute needs each axis of #{tensor.shape.inspect} exactly once, " \
                               "got #{axes.inspect}"
        end
        return tensor if axes == (0...rank).to_a

        shape = axes.map { |axis| tensor.shape[axis] }
        node = record(:permute, inputs: [tensor.node], attrs: { axes: axes }, dtype: tensor.dtype,
                      shape: shape, cse: [:permute, axes])
        Tensor.new(self, node)
      end

      # A slice of the outermost dim. **No kernel runs** — as with Buffer#slice
      # the slice is contiguous, so a view expresses it. Slicing an inner dim is
      # not contiguous, so it is not accepted here (docs/graph-ir.md).
      def slice(tensor, range)
        tensor = check_tensor!(tensor, "slice")
        raise Error, "slice needs at least 2 dimensions, got #{tensor.shape.inspect}" if tensor.shape.length < 2

        first, count = slice_range!(range, tensor.shape[0])
        return tensor if first.zero? && count == tensor.shape[0]

        shape = [count, *tensor.shape.drop(1)]
        node = record(:slice, inputs: [tensor.node], attrs: { offset: first * shape.drop(1).inject(1, :*) },
                      dtype: tensor.dtype, shape: shape, cse: [:slice, first, count])
        Tensor.new(self, node)
      end

      # Table lookup. indices is an int32 Tensor whose elements point at
      # positions along axis.
      #
      #   g.gather(table, ids)             # embedding: [vocab, d] indexed by [n] gives [n, d]
      #
      # Out-of-range indices are clamped to the ends (in the kernel).
      def gather(source, indices, axis: 0)
        source = check_tensor!(source, "gather source")
        indices = check_tensor!(indices, "gather indices")
        raise Error, "gather indices must be :int32, got #{indices.dtype.inspect}" unless indices.dtype == :int32

        axis = normalize_axis!(axis, source.shape)
        shape = source.shape.take(axis) + indices.shape + source.shape.drop(axis + 1)
        node = record(:gather, inputs: [source.node, indices.node], attrs: { axis: axis },
                      dtype: source.dtype, shape: shape, cse: [:gather, axis])
        Tensor.new(self, node)
      end

      # The inverse of table lookup (gather's vjp). shape is the shape added
      # back into (gather's source).
      #
      # It has no forward counterpart, so it is not on the Tensor surface — only
      # Grad's gather row uses it.
      def scatter_add(indices, values, shape:, axis: 0)
        indices = check_tensor!(indices, "scatter_add indices")
        values = check_tensor!(values, "scatter_add values")
        shape = normalize_shape(shape)
        axis = normalize_axis!(axis, shape)
        node = record(:scatter_add, inputs: [indices.node, values.node], attrs: { axis: axis },
                      dtype: values.dtype, shape: shape, cse: [:scatter_add, axis, shape])
        Tensor.new(self, node)
      end

      # Fills a border. before / after are the amounts added per dim.
      #
      #   g.pad(x, before: [0, 1], after: [0, 1], value: 0.0)
      def pad(tensor, before:, after: nil, value: 0.0)
        tensor = check_tensor!(tensor, "pad")
        rank = tensor.shape.length
        before = Array.new(rank, before) if before.is_a?(Integer)
        after = after.nil? ? before : (after.is_a?(Integer) ? Array.new(rank, after) : after)
        unless before.length == rank && after.length == rank &&
               (before + after).all? { |v| v.is_a?(Integer) && !v.negative? }
          raise ArgumentError, "pad needs a non-negative Integer per dimension of #{tensor.shape.inspect}"
        end
        return tensor if (before + after).all?(&:zero?)

        shape = tensor.shape.each_with_index.map { |dim, d| dim + before[d] + after[d] }
        value = tensor.dtype == :int32 ? value.to_i : value.to_f
        node = record(:pad, inputs: [tensor.node], attrs: { before: before, value: value },
                      dtype: tensor.dtype, shape: shape, cse: [:pad, before, after, value])
        Tensor.new(self, node)
      end

      # Changes only the shape, keeping element order. **No kernel runs** — at
      # run time it is a view of the input buffer (like Buffer#reshape). A
      # single -1 is inferred.
      def reshape(tensor, *dims)
        tensor = check_tensor!(tensor, "reshape")
        dims = dims.first if dims.length == 1 && dims.first.is_a?(Array)
        shape = resolve_reshape(tensor, dims)
        return tensor if tensor.shape == shape

        node = record(:reshape, inputs: [tensor.node], attrs: {}, dtype: tensor.dtype,
                      shape: shape, cse: [:reshape, shape])
        Tensor.new(self, node)
      end

      # Accepts **batches** as well as 2D pairs:
      #
      #   [m, k]    x [k, n]    -> [m, n]      same as eager matmul
      #   [b, m, k] x [b, k, n] -> [b, m, n]   one per batch element
      #
      # The latter has no eager counterpart (`Buffer#matmul` is 2D only). It
      # becomes one strided batched GEMM when available and per-batch matmuls
      # otherwise; conformance checks that the path does not change values.
      def matmul(a, b, out_dtype: nil)
        b = check_tensor!(b, "matmul")
        mode, shape = matmul_shape(a, b)
        out_dtype ||= promote(a.dtype, b.dtype)
        allowed = [promote(a.dtype, b.dtype)]
        allowed << a.dtype if a.dtype == b.dtype && Ops::LOW_PRECISION_DTYPES.include?(a.dtype)
        unless allowed.include?(out_dtype)
          raise ArgumentError, "matmul out_dtype must be #{allowed.map(&:inspect).join(' or ')} " \
                               "for #{a.dtype.inspect} x #{b.dtype.inspect} inputs"
        end
        node = record(:matmul, inputs: [a.node, b.node], attrs: { mode: mode, out_dtype: out_dtype },
                      dtype: out_dtype, shape: shape, cse: [:matmul, mode, out_dtype])
        Tensor.new(self, node)
      end

      # Concatenation along axis:. Along the outermost dim each piece lands
      # contiguously in the output and one copy per piece suffices; along an
      # inner dim pieces are scattered, so a kernel writes each into its slot.
      def concat(*tensors, axis: 0)
        tensors = tensors.flatten.map { |t| check_tensor!(t, "concat") }
        raise ArgumentError, "concat needs at least two tensors, got #{tensors.length}" if tensors.length < 2

        dtype = tensors.first.dtype
        axis = normalize_axis!(axis, tensors.first.shape)
        tensors.each do |t|
          same = t.dtype == dtype && t.shape.length == tensors.first.shape.length &&
                 t.shape.each_with_index.all? { |dim, d| d == axis || dim == tensors.first.shape[d] }
          next if same

          raise ArgumentError, "concat needs the same dtype and the same dims outside axis #{axis}: " \
                               "#{tensors.map { |o| "#{o.shape.inspect} #{o.dtype}" }.join(' , ')}"
        end

        shape = tensors.first.shape.dup
        shape[axis] = tensors.sum { |t| t.shape[axis] }
        node = record(:concat, inputs: tensors.map(&:node), attrs: { axis: axis }, dtype: dtype,
                      shape: shape, cse: [:concat, dtype, axis])
        Tensor.new(self, node)
      end

      # Built-ins the graph **does not decompose**, like NN's fused kernels. The
      # recording entry point is Graph.record_nn (one line on the NN side), so
      # this is not normally called directly.
      #
      # Unlike opaque, it knows by name what it is, so CSE applies and fusion can
      # treat it as a partner. Weights passed as Buffers (gamma / bias ...) are
      # lifted to const nodes like g.param, so writing a graph needs no rewrapping
      # of weights.
      def nn(fn, inputs, shape:, dtype:, tensors: {}, args: [], opts: {})
        check_dtype!(dtype)
        positional = inputs.map { |value| lift(value, "#{fn} input") }
        keyed = tensors.reject { |_, value| value.nil? }
                       .to_h { |key, value| [key, lift(value, "#{fn} #{key}")] }
        # A differentiable graph records decompositions — except that **functions
        # with a hand-written vjp stay :nn**. They differentiate without
        # decomposing, so there is no reason to give up the fused forward kernel
        # (Grad::NN_REGISTRY).
        if HighLevel.lower?(fn) || (@differentiable && HighLevel.defined?(fn) && !Grad.nn_defined?(fn))
          return lowered(fn, positional, keyed, args, opts, shape, dtype)
        end

        node = record(:nn, inputs: (positional + keyed.values).map(&:node),
                      attrs: { fn: fn, arity: positional.length, keys: keyed.keys,
                               args: args, opts: opts },
                      dtype: dtype, shape: normalize_shape(shape),
                      cse: [:nn, fn, args, keyed.keys, opts])
        Tensor.new(self, node)
      end

      # Expands a high-level op into primitives (Graph.lower_high_level).
      # Whether to expand is decided **at record time**, so one plan never mixes
      # the two.
      #
      # The declared shape and type are checked here, so a wrong decomposition
      # fails at record time instead of surviving into execution or numeric
      # comparison. Only the type is coerced — the hand-written kernel also
      # computes in float and rounds to x's type on write-back, so the trailing
      # cast reproduces that write-back.
      def lowered(fn, inputs, tensors, args, opts, shape, dtype)
        result = HighLevel.lower(self, fn, inputs, tensors: tensors, args: args, opts: opts)
        result = check_tensor!(result, "the decomposition of #{fn}")
        unless result.shape == shape
          raise Error, "the decomposition of #{fn} produced #{result.shape.inspect} " \
                       "but #{fn} declares #{shape.inspect}"
        end

        cast(result, dtype)
      end

      # 2D convolution. Conv.forward calls this, so it is not normally called
      # directly.
      #
      # Without convolution on this backend (caps.conv is nil) it is **expanded
      # into primitives** at record time — only in the graph layer can
      # convolution be written anywhere. The expansion is graph/high_level.rb's :conv2d.
      def conv2d(input, weight, shape:, pad:, stride:, dilation: [1, 1], groups: 1)
        input = lift(input, "conv2d input")
        weight = lift(weight, "conv2d weight")
        opts = { pad: pad, stride: stride, dilation: dilation, groups: groups }
        if HighLevel.lower?(:conv2d) || !Tilework.capabilities.conv_dtypes.include?(input.dtype)
          return lowered(:conv2d, [input, weight], {}, [], opts, shape, input.dtype)
        end

        node = record(:conv2d, inputs: [input.node, weight.node], attrs: opts,
                      dtype: input.dtype, shape: normalize_shape(shape),
                      cse: [:conv2d, pad, stride, dilation, groups])
        Tensor.new(self, node)
      end

      # Convolution backward (#22 Step 2). Calls cuDNN / MPSGraph's dedicated
      # kernels as is — when forward is delegated, delegating backward is
      # fastest (differentiating the im2col decomposition doubles the expansion).
      #
      # like is an input used only for its shape (as in Conv.backward_data /
      # backward_filter's arguments). It must be among the node's inputs, or
      # liveness would release that buffer early.
      def conv2d_backward(kind, primary, grad_output, like, pad:, stride:, dilation: [1, 1], groups: 1)
        node = record(:"conv2d_backward_#{kind}",
                      inputs: [primary.node, grad_output.node, like.node],
                      attrs: { pad: pad, stride: stride, dilation: dilation, groups: groups },
                      dtype: like.dtype, shape: like.shape,
                      cse: [:"conv2d_backward_#{kind}", pad, stride, dilation, groups])
        Tensor.new(self, node)
      end

      # --- Random numbers ---
      #
      # Counter-based RNG (Philox) takes its key as an argument, so **deriving
      # each node's key from the plan seed and a static index makes it pure**:
      # independent of execution order, same seed same sequence. That is why
      # RNG is not mutable state (g.state) — #19 §6's conclusion.
      #
      #   plan.run(x: buf, seed: 42)
      #
      # The same expression must draw **different values each time**, so draws
      # are exempt from CSE.
      def random(shape, dtype: :float32) = random_node(:uniform, shape, dtype, {})

      def randn(shape, mean: 0.0, stddev: 1.0, dtype: :float32)
        random_node(:normal, shape, dtype, { mean: mean.to_f, stddev: stddev.to_f })
      end

      # How many draws the plan makes; the step for advancing the seed (Plan#next_seed).
      def random_draws = @random_draws

      # A node whose contents are not inspected: the exit for things the graph
      # cannot (or must not) decompose, like Tilework.kernel {} DSL kernels or
      # NN's fused functions. body takes the input Buffers and returns a Buffer
      # as declared.
      #
      # By default it synchronizes on both sides of the block (what it does is
      # unknown). When **the caller can declare** the block stays inside our
      # queue via `Ops.launch_kernel`, passing `fenced: false` removes the fence
      # so it joins its neighbors' submission (#86). Never pass it for a block
      # calling a library that creates its own command buffer / stream.
      def opaque(*inputs, shape:, dtype: :float32, name: "opaque", fenced: true, &body)
        raise ArgumentError, "opaque requires a block" if body.nil?

        check_dtype!(dtype)
        inputs = inputs.map { |t| check_tensor!(t, "opaque input") }
        node = record(:opaque, inputs: inputs.map(&:node), attrs: { name: name, body: body, fenced: fenced },
                      dtype: dtype, shape: normalize_shape(shape))
        Tensor.new(self, node)
      end

      # A tensor filled with one value. To **avoid bringing a buffer from the
      # host** it adds iota multiplied by 0 (both nodes are elementwise, so
      # fusion folds them into one). Used for the backward seed (the loss
      # cotangent 1.0) and zero gradients of parameters the loss does not use.
      def fill(shape, value, dtype: :float32)
        base = (iota(shape, axis: 0, dtype: :float32) * 0.0) + value.to_f
        dtype == :float32 ? base : cast(base, dtype)
      end

      # --- checkpointing (#22 Step 5) -----------------------------------------

      # Does **not keep** intermediates made inside the block for backward;
      # instead the same computation is recorded again when backward is recorded.
      #
      #   h = g.checkpoint { Tilework::NN.gelu(x.matmul(w1)).matmul(w2) }
      #
      # Backward is a peak-memory problem, and its cause is "forward
      # intermediates live to the end" (vjps read them, so liveness rightly
      # extends them). With checkpoint, the region's intermediates are read only
      # inside forward, so **they die when the region ends**; backward reads the
      # rebuilt values.
      #
      # **In a graph, "recomputation" is not a run-time mechanism, just more
      # nodes.** The region's nodes are duplicated under new ids and backward's
      # inputs are switched to them. The copies bypass CSE (going through it
      # would fold them back onto the originals, doing nothing).
      #
      # Three kinds are never duplicated: opaque (its contents are unknown —
      # rebuilding reruns the user's block), state reads/writes (memory, not
      # values) and inputs/constants (already live). Their original values stay
      # alive, so correctness is unaffected; only the saving shrinks.
      #
      # **Random draws can be duplicated.** The key comes from (plan seed, the
      # node's static index) and the index lives in attrs, so a copy **redraws
      # the same sequence** — #19's choice not to make RNG mutable state is what
      # guarantees recomputation gives the same values. (Assigning indices
      # dynamically by recording order would break this silently.)
      NEVER_RECOMPUTE = %i[input const opaque state_init state_read
                           state_write state_merge].freeze

      Checkpoint = Struct.new(:from, :to)

      def checkpoint
        from = @nodes.length
        result = check_tensor!(yield, "the checkpoint block")
        @checkpoints << Checkpoint.new(from, @nodes.length)
        result
      end

      # --- Backward (#22) -----------------------------------------------------

      # Records the gradients of loss with respect to each wrt: tensor **as nodes
      # of the same graph**. They come back as ordinary Tensors, usable as outputs
      # or for writing back to state (parameter updates).
      #
      #   dw, db = g.grad(loss, wrt: [w, b])
      #   g.write_into(w_ref, value: g.read(w_ref) - (lr * dw))
      #
      # Walking **ids in descending order** is enough for the reverse pass:
      # inputs are always recorded before their consumers, so ascending id
      # order is itself a topological order.
      #
      # High-level ops (:nn) differentiate only as decompositions, so graphs to
      # differentiate are recorded with `Graph.build(differentiable: true)`.
      def grad(loss, wrt:)
        loss = check_tensor!(loss, "grad")
        unless loss.length == 1
          raise Error, "grad requires a scalar loss (shape [1]), got #{loss.shape.inspect}; " \
                       "reduce it first (for example loss.mean)"
        end
        targets = Array(wrt).map { |t| grad_target!(t) }
        if (target = targets.find { |value| value.dtype == :bfloat16 })
          raise Error, "bfloat16 training is not supported yet (wrt target #{target.shape.inspect}); " \
                       "keep trainable parameters and optimizer state in :float32"
        end

        cotangents = Hash.new { |h, k| h[k] = [] }
        cotangents[loss.node.id] << fill(loss.shape, 1.0, dtype: loss.dtype)
        backward(loss.node, cotangents)

        targets.map do |target|
          parts = cotangents[target.node.id]
          parts.empty? ? fill(target.shape, 0.0, dtype: target.dtype) : sum_cotangents(parts)
        end
      end

      # --- Outputs and compile ---

      # Marks a tensor as a plan output.
      #
      # @param tensor [Tensor]
      # @return [Tensor]
      def output(tensor)
        tensor = check_tensor!(tensor, "output")
        # Returning a state view as is would let later writes change it, so
        # outputs are always materialized.
        tensor.node.attrs[:materialize] = true if tensor.node.op == :state_read
        @outputs << tensor.node
        tensor
      end

      # Compiles this recorded graph into an executable plan.
      #
      # @param vram [Integer, nil] optional device-memory budget in bytes
      # @param spill_dir [String, nil] directory used for planned weight offload
      # @return [Plan]
      #
      # vram: adds the residency pass that fits weights (:const) into the budget
      # (#17 Phase 4 / docs/graph-offload.md). Without it everything stays
      # resident as before and the pass does not exist.
      #
      #   plan = g.compile(vram: 8 * 1024**3)
      #   plan.residency   # tells whether it fits before running
      def compile(vram: nil, spill_dir: nil, backing: :file, host_budget: nil) =
        Plan.new(self, vram: vram, spill_dir: spill_dir, backing: backing, host_budget: host_budget)

      def input_nodes = @inputs

      # --- Internal ---

      def record(op, attrs: {}, inputs: [], dtype:, shape:, cse: nil)
        # Ops missing from the table are not recorded. Running with undeclared
        # behavior (may it be freed, does it run synchronously) would misschedule
        # rather than raise, so fail at record time.
        Op.fetch(op)
        key = cse && [op, inputs.map(&:id), cse]
        found = key && @cse[key]
        return found if found

        node = Node.new(@nodes.length, op, inputs, attrs, dtype, shape)
        @nodes << node
        @cse[key] = node if key
        node
      end

      private

      # The body of backward. Visits only nodes reachable from loss in
      # descending id order, sums the cotangents accumulated at each node and
      # has the vjp table produce the input-side cotangents.
      #
      # Consumers always have larger ids than producers, so visiting in
      # descending order guarantees a node's cotangents are complete.
      def backward(root, cotangents)
        reachable = {}
        stack = [root]
        while (node = stack.pop)
          next if reachable[node.id]

          reachable[node.id] = true
          node.inputs.each { |input| stack << input }
        end

        # A node with a copy passes the copy to its vjp, which also swaps rows
        # that use their own output (exp / tanh / softmax). Only cotangents
        # accumulate under **the original node's** input ids, since the walk is
        # over the original graph.
        recomputed = {}
        rebuilt = {}
        # Adding copies grows @nodes, so walk a snapshot of the current list.
        @nodes.dup.reverse_each do |node|
          next unless reachable[node.id]

          parts = cotangents[node.id]
          next if parts.empty?

          dy = sum_cotangents(parts)
          cotangents[node.id] = [dy]
          next if node.inputs.empty?

          # Rebuild the region where it is first needed (= its exit, walking in
          # reverse). Building them all up front lets the scheduler hoist them
          # once their dependencies are ready, and the peak does not drop.
          rebuild_region(node, reachable, recomputed, rebuilt)
          target = recomputed[node.id] || node
          inputs = target.inputs.map { |input| Tensor.new(self, input) }
          Grad.apply(self, target, dy, inputs).each_with_index do |part, i|
            cotangents[node.inputs[i].id] << part if part
          end
        end
      end

      # If node is in a checkpoint region not yet rebuilt, duplicate the whole
      # region. Inputs inside the same region go to the copies; those outside
      # (the boundary) connect as is. It bypasses record to avoid CSE — the same
      # expression folds to the same node, so going through it would erase the copies.
      def rebuild_region(node, reachable, map, rebuilt)
        index = @checkpoints.index { |region| node.id >= region.from && node.id < region.to }
        return if index.nil? || rebuilt[index]

        rebuilt[index] = true
        region = @checkpoints[index]
        @nodes[region.from...region.to].each do |original|
          next unless reachable[original.id]
          next if NEVER_RECOMPUTE.include?(original.op)

          inputs = original.inputs.map { |input| map[input.id] || input }
          clone = Node.new(@nodes.length, original.op, inputs, original.attrs,
                           original.dtype, original.shape)
          @nodes << clone
          map[original.id] = clone
        end
      end

      # A node with several consumers receives several cotangents. They are
      # summed with ordinary binary ops (fusion folds them) — accumulating into
      # a buffer would create a mutation the scheduler was never told about.
      def sum_cotangents(parts) = parts.length == 1 ? parts.first : parts.inject(:+)

      def grad_target!(target)
        return target if target.is_a?(Tensor)

        if target.is_a?(StateRef)
          raise Error, "grad wrt: takes the tensor the forward pass used; for a state, " \
                       "pass the value you read (g.read(ref)), not the reference itself"
        end

        raise TypeError, "grad wrt: requires a Tilework::Graph::Tensor, got #{target.class}"
      end

      # Operands of different shapes get a broadcast node before recording. From
      # then on shapes match, so **only recording** knows about broadcasting.
      def binary_node(op, a, b, dtype:, kind:)
        shape = common_shape!(a, b)
        a = broadcast(a, shape)
        b = broadcast(b, shape)
        node = record(:binary, inputs: [a.node, b.node], attrs: { op: op, kind: kind },
                      dtype: dtype, shape: shape, cse: [:binary, kind, op, dtype])
        Tensor.new(self, node)
      end

      def reduce_node(op, kind, tensor, axis, keepdims, dtype)
        axis = normalize_axis!(axis, tensor.shape) unless axis.nil?
        shape = axis.nil? ? [1] : reduced_shape(tensor.shape, axis, keepdims)
        node = record(op, inputs: [tensor.node], attrs: { kind: kind, axis: axis, keepdims: keepdims },
                      dtype: dtype, shape: shape, cse: [op, kind, dtype, axis, keepdims])
        Tensor.new(self, node)
      end

      def normalize_axis!(axis, shape)
        rank = shape.length
        raise TypeError, "axis must be an Integer, got #{axis.class}" unless axis.is_a?(Integer)

        resolved = axis.negative? ? rank + axis : axis
        raise ArgumentError, "axis #{axis} is out of range for #{shape.inspect}" unless resolved.between?(0, rank - 1)

        resolved
      end

      # where's output type: the binary-op rule applied to the two selectable
      # sides (the condition's int32 does not affect the result type — it only selects).
      def where_dtype(a, b)
        left = a.is_a?(Numeric) ? a : check_tensor!(a, "where a").dtype
        right = b.is_a?(Numeric) ? b : check_tensor!(b, "where b").dtype
        # Both may be scalars (building a mask); the shape comes from the condition.
        if left.is_a?(Numeric) && right.is_a?(Numeric)
          return left.is_a?(Integer) && right.is_a?(Integer) ? :int32 : :float32
        end

        left.is_a?(Numeric) ? result_dtype(right, left) : result_dtype(left, right)
      end

      # The common shape of the three operands; scalars have no shape and are skipped.
      def where_shape!(cond, a, b)
        operands = [a, b].reject { |value| value.is_a?(Numeric) }
                         .map { |value| check_tensor!(value, "where operand") }
        operands.inject(cond.shape) do |acc, tensor|
          shape = broadcast_shape(acc, tensor.shape)
          unless shape
            raise ArgumentError, "cannot broadcast #{acc.inspect} and #{tensor.shape.inspect} together in where"
          end

          shape
        end
      end

      # Folded axes disappear (kept as 1 with keepdims). If none remain the
      # shape is [1] — the same as a whole-buffer reduction, so 0-dim Tensors
      # never need handling downstream.
      def reduced_shape(shape, axis, keepdims)
        dims = shape.dup
        keepdims ? dims[axis] = 1 : dims.delete_at(axis)
        dims.empty? ? [1] : dims
      end

      def scalar_node(op, a, scalar, dtype:, kind:, reversed:)
        node = record(:binary, inputs: [a.node],
                      attrs: { op: op, kind: kind, scalar: scalar, reversed: reversed },
                      dtype: dtype, shape: a.shape,
                      cse: [:binary, kind, op, dtype, scalar, scalar.class.name, reversed])
        Tensor.new(self, node)
      end

      # The result shape of a binary op: NumPy right alignment, dims matching or
      # one being 1. **Eager (Ops) does not broadcast**, so this is expressive
      # power only the graph has (like batched matmul and concat).
      def common_shape!(a, b)
        shape = broadcast_shape(a.shape, b.shape)
        return shape if shape

        raise ArgumentError, "tensor shape mismatch: #{a.shape.inspect} vs #{b.shape.inspect} " \
                             "(shapes must be equal or broadcastable)"
      end

      # nil if they do not line up. Right-align, then stretch the side that is 1.
      def broadcast_shape(left, right)
        rank = [left.length, right.length].max
        a = Array.new(rank - left.length, 1) + left
        b = Array.new(rank - right.length, 1) + right
        dims = a.zip(b).map do |x, y|
          return nil unless x == y || x == 1 || y == 1

          [x, y].max
        end
        dims
      end

      # Infer a single -1, then compare element counts (as Buffer#reshape does).
      def resolve_reshape(tensor, dims)
        unless dims.count(-1) <= 1 && dims.all? { |d| d.is_a?(Integer) && (d.positive? || d == -1) }
          raise ArgumentError, "reshape dims must be positive Integers with at most one -1, got #{dims.inspect}"
        end

        if (hole = dims.index(-1))
          known = dims.reject { |d| d == -1 }.inject(1, :*)
          unless known.positive? && (tensor.length % known).zero?
            raise ArgumentError, "cannot reshape #{tensor.shape.inspect} to #{dims.inspect}"
          end

          dims = dims.dup
          dims[hole] = tensor.length / known
        end
        unless dims.inject(1, :*) == tensor.length
          raise ArgumentError, "cannot reshape #{tensor.shape.inspect} (#{tensor.length} elements) " \
                               "to #{dims.inspect}"
        end

        normalize_shape(dims)
      end

      # Accepts the same forms as Buffer#slice (an Integer for one, a Range for a range).
      def slice_range!(range, limit)
        case range
        when Integer then [normalize_slice_index!(range, limit), 1]
        when Range
          first = normalize_slice_index!(range.begin || 0, limit)
          last = range.end.nil? ? limit - 1 : normalize_slice_index!(range.end, limit)
          last -= 1 if range.exclude_end?
          raise ArgumentError, "slice #{range.inspect} is empty" if last < first

          [first, last - first + 1]
        else raise TypeError, "slice takes an Integer or a Range, got #{range.class}"
        end
      end

      def normalize_slice_index!(index, limit)
        resolved = index.negative? ? limit + index : index
        raise ArgumentError, "slice index #{index} is out of range for #{limit} rows" unless resolved.between?(0, limit - 1)

        resolved
      end

      def broadcastable?(from, to)
        return false if from.length > to.length

        broadcast_shape(from, to) == to
      end

      def matmul_shape(a, b)
        case [a.shape.length, b.shape.length]
        when [2, 2]
          check_inner_dim!(a, b, a.shape[1], b.shape[0])
          [:plain, [a.shape[0], b.shape[1]]]
        when [3, 3]
          unless a.shape[0] == b.shape[0]
            raise ArgumentError, "batched matmul needs the same batch on both sides, " \
                                 "got #{a.shape.inspect} x #{b.shape.inspect}"
          end

          check_inner_dim!(a, b, a.shape[2], b.shape[1])
          [:batched, [a.shape[0], a.shape[1], b.shape[2]]]
        else
          raise ArgumentError, "matmul takes [m, k] x [k, n] or [b, m, k] x [b, k, n], " \
                               "got #{a.shape.inspect} x #{b.shape.inspect}"
        end
      end

      def check_inner_dim!(a, b, left, right)
        return if left == right

        raise ArgumentError, "matmul shape mismatch: #{a.shape.inspect} x #{b.shape.inspect}"
      end

      # The same rule as Ops.promoted_dtype (int32 only when both are int32).
      # float16 is never a compute type, so it does not appear here.
      def promote(dtype, other)
        other_int = other.is_a?(Symbol) ? other == :int32 : other.is_a?(Integer)
        dtype == :int32 && other_int ? :int32 : :float32
      end

      # The same rule as Ops.result_dtypes' output type. Only float16 with
      # float16 (or with a scalar) stays float16 — so intermediate operations do
      # not throw away the bandwidth half precision saves.
      def result_dtype(dtype, other)
        same_storage = other.is_a?(Symbol) ? other == dtype : other.is_a?(Numeric)
        return dtype if Ops::LOW_PRECISION_DTYPES.include?(dtype) && same_storage

        promote(dtype, other)
      end

      def to_region(ref, at)
        case at
        when nil    then ref.whole
        when Region then validate_region!(ref, at)
        else ref.rows(at)
        end
      end

      def validate_region!(ref, region)
        return region if region.total == ref.state.buffer.length

        raise Error, "region #{region.inspect} was built for a #{region.total}-element buffer, " \
                     "but #{ref.state.name} has #{ref.state.buffer.length}"
      end

      def check_ref!(ref)
        raise TypeError, "expected a Tilework::Graph::StateRef, got #{ref.class}" unless ref.is_a?(StateRef)
        raise Error, "state #{ref.state.name} belongs to another graph" unless ref.graph.equal?(self)

        ref
      end

      def random_node(kind, shape, dtype, attrs)
        unless dtype == :float32
          raise Error, "random generation only supports :float32 (got #{dtype.inspect})"
        end

        index = @random_draws
        @random_draws += 1
        node = record(:random, attrs: attrs.merge(kind: kind, index: index),
                      dtype: dtype, shape: normalize_shape(shape))
        Tensor.new(self, node)
      end

      # Tensors as is; Buffers become const nodes.
      def lift(value, what)
        return const(value) if value.is_a?(Buffer)

        check_tensor!(value, what)
      end

      def check_tensor!(value, what)
        return value if value.is_a?(Tensor) && value.graph.equal?(self)
        raise TypeError, "#{what} requires a Tilework::Graph::Tensor, got #{value.class}" unless value.is_a?(Tensor)

        raise Error, "#{what} got a tensor from another graph"
      end

      def check_dtype!(dtype)
        return if DTYPES.include?(dtype)

        raise Error, "the graph layer supports #{DTYPES.join(' / ')} for now, got #{dtype.inspect}"
      end

      def normalize_shape(shape)
        dims = shape.is_a?(Array) ? shape : [shape]
        unless dims.any? && dims.all? { |d| d.is_a?(Integer) && d.positive? }
          raise ArgumentError, "shape must be a positive Integer or an Array of them, got #{shape.inspect}"
        end

        dims
      end
    end
  end
end
