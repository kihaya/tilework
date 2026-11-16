# frozen_string_literal: true

module Tilework
  module Graph
    # The contract for how an IR op behaves. REGISTRY is the single source of
    # the op set and of each op's properties as the scheduler sees them.
    #
    # It is a table to stop things falling through the cracks. These
    # properties used to be scattered across four constants in plan.rb
    # (NON_OWNED / SYNCHRONOUS_OPS / FOREIGN_OPS / PLACEABLE), a case in
    # execute, FUSABLE in fusion.rb and node.rb's labels, so adding an op meant
    # touching six places. Forgetting one raised nothing; the scheduler just
    # silently did the wrong thing (freeing a buffer it must not, not waiting
    # where it must). Collected here, an undeclared op fails in Op.fetch — the
    # same idea as Capabilities::FIELDS.
    #
    #   op = Graph::Op.fetch(node.op)
    #   op.owned?              # does it own its buffer (= may it be freed)?
    #   op.synchronous?(node)  # does it wait in place instead of queueing?
    #
    # Properties that depend on node attributes or the backend's capabilities
    # (batched matmul, delegated GEMM / RNG, ...) are declared as Procs;
    # anything expressible as a boolean stays a boolean.
    class Op
      # NN functions delegated to a vendor library when gemm_library is set.
      # They need waits on both sides for the same reason as matmul (decided
      # without backend names).
      FOREIGN_NN = %i[multi_head_attention].freeze

      REGISTRY = {}

      class << self
        # One row of the table.
        #
        #   run          the Plan method to execute; all take (node, inputs, bindings)
        #   launches     whether it launches a kernel (or copy). A node with
        #                false leaves the queue as it was, so after a synchronous
        #                node it is "still empty" — used to decide whether
        #                automatic offload's eviction can ride on that wait
        #                (#17 Phase 4c)
        #   owned        whether it owns its buffer. Buffers of nodes with false
        #                (inputs, constants, state) belong outside the graph, so
        #                liveness leaves them alone
        #   placeable    whether, as a write_into value, it may write straight
        #                into the state's view. Only ops that allocate their
        #                output once and fill it (#19 §3)
        #   fusable      whether elementwise fusion can fold it into one kernel
        #   index_map    whether it only changes how its input is read (fusion
        #                folds it into the index)
        #   synchronous  whether it runs in place instead of queueing and moving on
        #   foreign      whether it may run outside our queue (synchronize on both sides)
        #   requires     the capability needed to run this node (nil if none).
        #                Plan#requires collects them to report what is missing
        #                before running
        #   writes_state whether it advances a mutable state's version
        #   label        the line for explain / inspect (defaults to the op name)
        def define(name, run:, owned: true, placeable: false, fusable: false, index_map: false,
                   synchronous: false, foreign: false, writes_state: false, requires: nil,
                   launches: true, label: nil)
          raise Error, "graph op #{name.inspect} is already defined" if REGISTRY.key?(name)

          REGISTRY[name] = new(name: name, run: run, owned: owned, placeable: placeable,
                               fusable: fusable, index_map: index_map, synchronous: synchronous,
                               foreign: foreign, writes_state: writes_state, requires: requires,
                               launches: launches, label: label)
        end

        def fetch(name)
          REGISTRY.fetch(name) do
            raise Error, "unknown graph op #{name.inspect} (known: #{names.map(&:inspect).join(', ')})"
          end
        end

        def defined?(name) = REGISTRY.key?(name)
        def names = REGISTRY.keys

        # Whether a delegated GEMM leaves our queue, and so has to wait on both
        # sides. matmul and :nn make the same decision, so it lives in one
        # place. A bridge whose vendor GEMM is enqueued on the same queue as
        # its kernels, in submission order, declares `ordered_gemm?`; ordering
        # then covers it exactly as it covers a kernel and no fence is needed.
        def foreign_gemm?
          return false if Tilework.capabilities.gemm_library.nil?

          bridge = Tilework.bridge
          !(bridge.respond_to?(:ordered_gemm?) && bridge.ordered_gemm?)
        end

        # Whether a capability is actually usable; nil and false both mean no.
        def satisfied?(capability)
          value = Tilework.capabilities.public_send(capability)
          !value.nil? && value != false
        end

        # The axis shown on a reduction (none for a whole-buffer reduction).
        def axis_suffix(node)
          axis = node.attrs[:axis]
          return nil if axis.nil?

          "axis #{axis}#{node.attrs[:keepdims] ? ' keepdims' : ''}"
        end
      end

      attr_reader :name, :run

      def initialize(name:, run:, owned:, placeable:, fusable:, index_map:, synchronous:, foreign:,
                     writes_state:, requires:, launches:, label:)
        @name = name
        @run = run
        @owned = owned
        @placeable = predicate(placeable)
        @fusable = fusable
        @index_map = index_map
        @synchronous = predicate(synchronous)
        @foreign = predicate(foreign)
        @writes_state = writes_state
        @requires = requires
        @launches = launches
        @label = label
      end

      # The capability needed to run this node (nil if none).
      attr_reader :requires

      def owned? = @owned
      def fusable? = @fusable

      # Whether it launches anything; false means the node leaves the queue as it was.
      def launches? = @launches

      # An op that creates no value and only changes how its source buffer is
      # read (fusion folds it into the index).
      def index_map? = @index_map
      def writes_state? = @writes_state
      def placeable?(node) = @placeable.call(node)
      def synchronous?(node) = @synchronous.call(node)
      def foreign?(node) = @foreign.call(node)
      def label_for(node) = @label ? @label.call(node) : name.to_s

      private

      # Boolean properties are wrapped in Procs too, so callers need no case split.
      def predicate(value) = value.respond_to?(:call) ? value : ->(_node) { value }
    end

    # --- The table -----------------------------------------------------------
    #
    # Structural nodes. They create no value (they point at existing buffers),
    # so owned: false.

    Op.define :input, run: :run_input, owned: false, launches: false,
              label: ->(node) { "input #{node.attrs[:name].inspect}" }

    Op.define :const, run: :run_const, owned: false, launches: false, label: ->(_node) { "const" }

    Op.define :state_init, run: :run_state_init, owned: false, launches: false

    # A read is a view by default, so it may or may not own memory; when
    # materialized it may be freed, hence owned: true. The release side rejects views.
    Op.define :state_read, run: :run_state_read,
              label: ->(node) { "read #{node.state.name} v#{node.attrs[:version]} #{node.region.inspect}" }

    Op.define :state_write, run: :run_state_write, owned: false, writes_state: true,
              label: lambda { |node|
                "write #{node.state.name} -> v#{node.attrs[:version]} #{node.region.inspect}"
              }

    Op.define :state_merge, run: :run_state_merge, owned: false, writes_state: true, launches: false,
              label: ->(node) { "merge #{node.state.name} -> v#{node.attrs[:version]}" }

    # Compute nodes.

    Op.define :binary, run: :run_binary, placeable: true, fusable: true,
              label: lambda { |node|
                "#{node.attrs[:op]}#{node.attrs.key?(:scalar) ? " #{node.attrs[:scalar]}" : ''}"
              }

    Op.define :unary, run: :run_unary, placeable: true, fusable: true,
              label: ->(node) { node.attrs[:fn].to_s }

    # A whole-buffer reduction (axis: nil) folds as a tree, and the execution
    # context takes over freeing the partials between passes (#86), so it keeps
    # queueing and moves on. An axis reduction is one kernel with no
    # intermediates (and its output can be placed).
    Op.define :reduce, run: :run_reduce,
              placeable: ->(node) { !node.attrs[:axis].nil? },
              label: ->(node) { [node.attrs[:kind], Op.axis_suffix(node)].compact.join(" ") }

    Op.define :arg_reduce, run: :run_arg_reduce,
              placeable: ->(node) { !node.attrs[:axis].nil? },
              label: ->(node) { ["arg#{node.attrs[:kind]}", Op.axis_suffix(node)].compact.join(" ") }

    # Change the type / select by condition / write indices. Each is closed per
    # element, so each can fuse (:iota has no inputs, so it is never absorbed
    # as an operand, though it can be the root that owns the output).
    Op.define :cast, run: :run_cast, placeable: true, fusable: true,
              label: ->(node) { "cast #{node.dtype}" }

    Op.define :where, run: :run_where, placeable: true, fusable: true

    Op.define :iota, run: :run_iota, placeable: true,
              label: ->(node) { "iota axis #{node.attrs[:axis]}" }

    # A node that only changes the shape. Execution returns a view of its input:
    # no kernel runs and no new buffer is owned (= owned: false). Being an
    # alias, the source buffer's lifetime must extend to this node's consumers
    # (Plan#alias_root).
    Op.define :reshape, run: :run_reshape, owned: false, launches: false,
              label: ->(node) { "reshape #{node.shape.inspect}" }

    # A node that only remaps indices. It can fuse — but unlike other ops it
    # folds **into the index**, not into the expression (the consumer's kernel
    # reads the source buffer with remapped indices). Only when not fused does
    # one materializing kernel run. Broadcast (stretching) and permute
    # (swapping axes) are the same operation with different stride tables.
    Op.define :broadcast, run: :run_remap, placeable: true, fusable: true, index_map: true

    Op.define :permute, run: :run_remap, placeable: true, fusable: true, index_map: true,
              label: ->(node) { "permute #{node.attrs[:axes].inspect}" }

    # A slice of the outermost dimension. Returns a view, like reshape.
    Op.define :slice, run: :run_slice, owned: false, launches: false,
              label: ->(node) { "slice #{node.shape.inspect}" }

    # Table lookup and border fill. Both allocate their output once and fill it.
    Op.define :gather, run: :run_gather, placeable: true,
              label: ->(node) { "gather axis #{node.attrs[:axis]}" }

    # The inverse of table lookup. It allocates its output once and fills it
    # (each thread finishes its own share), so it is placeable. No atomics, so
    # it stays inside our queue.
    Op.define :scatter_add, run: :run_scatter_add, placeable: true,
              label: ->(node) { "scatter_add axis #{node.attrs[:axis]}" }

    Op.define :pad, run: :run_pad, placeable: true,
              label: ->(node) { "pad #{node.attrs[:before].inspect}" }

    # matmul is delegated to cuBLAS / MPS when gemm_library is set. Where that
    # leaves our execution path (Op.foreign_gemm?) it waits on both sides;
    # a bridge that declares `ordered_gemm?` stays asynchronous. Freeing the
    # per-batch temporaries of an undelegated batch is handled by #86's
    # deferred free.
    Op.define :matmul, run: :run_matmul,
              placeable: ->(node) { node.attrs[:mode] != :batched },
              synchronous: ->(_node) { Op.foreign_gemm? },
              foreign: ->(_node) { Op.foreign_gemm? }

    Op.define :concat, run: :run_concat, placeable: true

    # A delegated RNG (cuRAND / MPS) leaves our execution path, and some paths
    # fall back to writing from the host.
    Op.define :random, run: :run_random, synchronous: true,
              foreign: ->(_node) { !Tilework.capabilities.rng.nil? }

    # NN's fused kernels hand freeing their intermediates (scores / probs) to the
    # execution context (#86), so they wait only when delegated.
    Op.define :nn, run: :run_nn,
              synchronous: ->(node) { Op::FOREIGN_NN.include?(node.attrs[:fn]) && Op.foreign_gemm? },
              foreign: ->(node) { Op::FOREIGN_NN.include?(node.attrs[:fn]) && Op.foreign_gemm? },
              label: ->(node) { ["nn", node.attrs[:fn], *node.attrs[:args]].join(" ") }

    # 2D convolution. Delegated to cuDNN / MPSGraph, so it waits on both sides.
    # On a backend without convolution it is decomposed into primitives at
    # record time and never becomes this node (graph/high_level.rb's :conv2d).
    Op.define :conv2d, run: :run_conv2d, synchronous: true, foreign: true, requires: :conv,
              label: ->(node) {
                "conv2d pad #{node.attrs[:pad].inspect} stride #{node.attrs[:stride].inspect} " \
                "dilation #{node.attrs[:dilation].inspect} groups #{node.attrs[:groups].inspect}"
              }

    # Convolution backward (#22 Step 2). Runs on the delegate like the forward
    # pass, so it waits on both sides.
    %i[data filter].each do |kind|
      Op.define :"conv2d_backward_#{kind}", run: :"run_conv2d_backward_#{kind}",
                synchronous: true, foreign: true, requires: :conv_backward,
                label: ->(node) {
                  "conv2d backward #{kind} pad #{node.attrs[:pad].inspect} " \
                  "dilation #{node.attrs[:dilation].inspect} groups #{node.attrs[:groups].inspect}"
                }
    end

    # A user block. What it does is unknown, so by default it waits on both
    # sides. `g.opaque(..., fenced: false)` is the caller's declaration that
    # the block stays inside our queue, and only then is the fence removed.
    Op.define :opaque, run: :run_opaque,
              synchronous: ->(node) { node.attrs[:fenced] },
              foreign: ->(node) { node.attrs[:fenced] },
              label: ->(node) { "opaque #{node.attrs[:name]}" }
  end
end
