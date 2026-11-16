# frozen_string_literal: true

require_relative "graph/region"
require_relative "graph/op"
require_relative "graph/node"
require_relative "graph/state"
require_relative "graph/tensor"
require_relative "graph/pool"
require_relative "graph/residency"
require_relative "graph/kernels"
require_relative "graph/high_level"
require_relative "graph/grad"
require_relative "graph/fusion"
require_relative "graph/builder"
require_relative "graph/plan"
require_relative "graph/optim"

module Tilework
  # Opt-in deferred graph execution for tensors and plans.
  #
  #   plan = Tilework::Graph.build do |g|
  #     x = g.input(:x, [4, 8])
  #     w = g.param(weights)
  #     g.output(x.matmul(w) * 2.0)
  #   end.compile
  #
  #   y = plan.run(x: input_buffer)
  #
  # A layer added **on top of** the eager layer (Buffer / Ops); nothing below
  # changes. The graph is opt-in, so being able to write eager code is never
  # traded away.
  #
  # ## Mutable state
  #
  # A graph is a DAG of values, so things that live and change across plan
  # runs, like a KV cache, do not fit naturally. Here a write is **an operation
  # that returns a version**, turning the invisible memory dependency into a
  # value edge:
  #
  #   kv0   = g.state(k_cache)
  #   kv1   = g.write_into(kv0, at: g.index(:pos), value: k_new)
  #   k_all = g.read(kv1, 0..pos)     # depends on version 1 = a real edge
  #
  # Unlike SSA, a version points at the same memory rather than a new value,
  # so version edges express only RAW. Plan#compile adds WAR (writing while a
  # reader is still reading) as ordering edges.
  module Graph
    # Read semantics.
    #   :view — zero copy (default); no kernel
    #   :copy — materialize. The graph becomes a pure DAG, so comparing it with
    #           :view's result verifies the state machinery itself (slow)
    STRICT_MODES = %i[view copy].freeze

    class << self
      # Records a graph without allocating device memory.
      #
      # @param differentiable [Boolean] record decompositions required for reverse-mode gradients
      # @yieldparam builder [Graph::Builder]
      # @return [Graph::Builder] the recorded graph; call `#compile` to create a plan
      #
      # With differentiable: true, high-level ops (NN.softmax etc.) are recorded
      # as their decompositions. Gradients follow from primitive vjps alone, so
      # graphs that use g.grad are recorded this way (docs/autograd.md).
      def build(differentiable: false)
        builder = Builder.new(differentiable: differentiable)
        yield builder
        builder
      end

      # Whether any value is being recorded into a graph. Each `NN` function
      # branches on this in one line so it can be called eager or in a graph.
      def symbolic?(*values) = values.any? { |value| value.is_a?(Tensor) }

      # Records a **built-in the graph does not decompose**, like `NN`, as a node.
      #
      #   return Graph.record_nn(:softmax, [x], shape: x.shape, dtype: x.dtype) if Graph.symbolic?(x)
      #
      # Unlike `g.opaque`, the graph knows what this is by name: CSE applies, it
      # runs as the same-named `NN` function, and fusion can treat it as a
      # partner (opaque stays opaque to the end).
      #
      # tensors: are keyword arguments taking Tensors (gamma / beta / bias ...);
      # nils are dropped. args: are non-Tensor positional arguments and opts:
      # non-Tensor keyword arguments, both carried as node attributes.
      def record_nn(fn, inputs, shape:, dtype:, tensors: {}, args: [], opts: {})
        builder = builder_for(inputs, tensors, fn)
        builder.nn(fn, inputs, shape: shape, dtype: dtype, tensors: tensors, args: args, opts: opts)
      end

      # Records a convolution (called in one line from Conv.forward). Unlike :nn
      # the callee is Tilework::Conv, hence its own op.
      def record_conv2d(input, weight, shape:, pad:, stride:, dilation: [1, 1], groups: 1)
        builder = builder_for([input, weight], {}, :conv2d)
        builder.conv2d(input, weight, shape: shape, pad: pad, stride: stride, dilation: dilation, groups: groups)
      end

      # Queue launches without waiting (default). false waits per node like
      # eager — for measuring the effect and for bisecting ordering suspicions.
      def async = @async.nil? ? true : @async
      attr_writer :async

      # Reuse intermediates from the pool (default). false allocates per op and
      # frees at the end. Like async, for measuring and bisecting.
      def buffer_pool = @buffer_pool.nil? ? true : @buffer_pool
      attr_writer :buffer_pool

      # Fold chained elementwise ops into one kernel (#17 Phase 2). Can be
      # turned off like async / buffer_pool for measuring and bisecting;
      # conformance pins that fusion does not change values.
      def fusion = @fusion.nil? ? true : @fusion
      attr_writer :fusion

      # Fold expressions containing row-wise reductions into one kernel
      # (docs/graph-ir.md Phase C). One thread owns one row, so with few rows
      # per-element parallelism can be faster — switchable until measured on hardware.
      def row_fusion = @row_fusion.nil? ? true : @row_fusion
      attr_writer :row_fusion

      # Hand the node producing a write_into value the state's view as its
      # output (#19 §3). Off, the value is produced and then copied — like the
      # other switches, for measuring and bisecting.
      def placement = @placement.nil? ? true : @placement
      attr_writer :placement

      # Record high-level ops (:nn nodes) as their primitive decompositions.
      # Default false — execution stays on the hand-written fused kernels and
      # the decomposition serves as the definition of meaning (Tier 1 in
      # docs/graph-ir.md). true lets the two be compared, which is what the
      # conformance oracle uses.
      #
      # **It takes effect at record time**, so switch it before Graph.build.
      def lower_high_level = @lower_high_level.nil? ? false : @lower_high_level
      attr_writer :lower_high_level

      # Page weights in and out of VRAM to fit a budget (#17 Phase 4). On by
      # default, but **nothing happens without compile(vram:)** — this layer
      # never starts paging without being given a budget. Turning it off is,
      # like the other switches, for measuring and bisecting.
      def offload = @offload.nil? ? true : @offload
      attr_writer :offload

      # Ride eviction fences on nodes that already synchronize (#17 Phase 4c).
      # Off, every eviction adds a Tilework.synchronize — a switch for
      # measuring how much riding along actually saves.
      def offload_fence_reuse = @offload_fence_reuse.nil? ? true : @offload_fence_reuse
      attr_writer :offload_fence_reuse

      # Fetch asynchronously ahead of the demand point (#17 Phase 4b), closing
      # the gap measured on Jetson between one synchronous transfer at a time
      # (0.56 GB/s) and async depth 2–4 (1.85–2.24 GB/s). Without an async
      # transfer ABI (Offload.async? false) it never starts, so this switch
      # matters only for measuring and bisecting on the same machine.
      def offload_prefetch = @offload_prefetch.nil? ? true : @offload_prefetch
      attr_writer :offload_prefetch

      def strict_state = @strict_state || :view

      # Selects zero-copy or copied state reads for subsequently compiled plans.
      #
      # @param mode [:view, :copy]
      # @return [:view, :copy]
      # @raise [ArgumentError] for an unsupported mode
      def strict_state=(mode)
        mode = mode.to_sym
        unless STRICT_MODES.include?(mode)
          raise ArgumentError, "unknown strict_state #{mode.inspect} (expected #{STRICT_MODES.join(' or ')})"
        end

        @strict_state = mode
      end

      # An on-device copy, used for state writes that cannot be placed and for
      # materializing reads. When the producer can write straight into the
      # state's view (Graph.placement), no copy runs.
      def copy(src, dst)
        unless src.dtype == dst.dtype && src.length == dst.length
          raise Error, "cannot copy #{src.length} #{src.dtype} elements into " \
                       "#{dst.length} #{dst.dtype} elements"
        end

        n = src.length
        args = [flat(src, n), flat(dst, n), n]
        grid = (n / Ops::BLOCK.to_f).ceil
        # The bounds guard is in the kernel body, so launch with explicit
        # grid/block rather than the auto-guarded .() (only this form can be async).
        if Ops.context&.async?
          copy_kernel.launch_async(grid: grid, block: Ops::BLOCK, args: args)
        else
          copy_kernel.launch(grid: grid, block: Ops::BLOCK, args: args)
        end
        dst
      end

      private

      def builder_for(inputs, tensors, fn)
        found = (inputs + tensors.values).find { |value| value.is_a?(Tensor) }
        raise Error, "#{fn} got no Tilework::Graph::Tensor to record against" if found.nil?

        found.graph
      end

      def flat(buffer, n) = buffer.shape.length > 1 ? buffer.reshape(n) : buffer

      def copy_kernel
        @copy_kernel ||= Tilework.kernel("graph_copy") do |a, b, n|
          i = global_id_x
          b[i] = a[i] if i < n
        end
      end
    end
  end
end
