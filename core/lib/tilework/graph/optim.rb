# frozen_string_literal: true

module Tilework
  module Graph
    # Base class for graph-recorded parameter optimizers.
    #
    # An optimizer is **not a new mechanism** — it is a helper on top of
    # mutable state (#19) and gradients (graph/grad.rb) that does three things:
    #
    #   1. declare the parameters as g.state and return the read Tensors (attach)
    #   2. take the gradients of the loss (g.grad)
    #   3. write back the moments and parameters (g.write_into)
    #
    # It all lands in one plan, so **a step is one plan.run**. The update is
    # elementwise, so fusion folds it to one kernel per parameter.
    #
    #   opt = Tilework::Graph::Adam.new({ w1: w1, b1: b1 }, lr: 1e-3)
    #
    #   plan = Tilework::Graph.build(differentiable: true) do |g|
    #     x = g.input(:x, [n, d])
    #     p = opt.attach(g)                       # { w1: Tensor, b1: Tensor }
    #     loss = Tilework::NN.cross_entropy(x.matmul(p[:w1]) + p[:b1], labels)
    #     opt.step(loss)                          # records gradients and the update
    #     g.output(loss)
    #   end.compile
    #
    #   50.times { plan.run(x: xs) }              # the weight Buffers update in place
    #
    # **Buffers are allocated outside Graph.build** (in Adam.new), keeping the
    # graph layer's promise that recording touches no device.
    class Optimizer
      attr_reader :params, :grads

      # Creates an optimizer for mutable parameter buffers.
      #
      # @param params [Hash{Symbol, String => Buffer}] named parameter buffers
      # @return [Optimizer]
      def initialize(params)
        raise ArgumentError, "an optimizer needs at least one parameter" if params.empty?

        @params = params.transform_keys(&:to_sym)
        @refs = {}
        @grads = {}
        @slots = {} # the optimizer's own mutable state (moments etc.)
        allocate_slots!
      end

      # Attaches parameter buffers as graph state and returns their tensors.
      #
      # @param builder [Builder]
      # @return [Hash{Symbol => Tensor}]
      #
      # **The forward pass must use these Tensors.** Binding the parameters
      # separately with g.param still yields gradients, but the update would be
      # written somewhere else and have no effect.
      def attach(builder)
        @builder = builder
        @refs = @params.to_h { |name, buffer| [name, builder.state(buffer, name: name)] }
        @slot_refs = @slots.to_h { |key, buffer| [key, builder.state(buffer, name: key)] }
        @reads = @refs.transform_values { |ref| builder.read(ref) }
      end

      # Records gradient calculation and one parameter update step.
      #
      # @param loss [Tensor] scalar loss tensor
      # @return [Hash{Symbol => Tensor}] gradients by parameter name
      #
      # The gradients are returned so they can be inspected (for clipping or logging).
      def step(loss)
        raise Error, "call attach(g) before step(loss)" if @builder.nil?

        names = @reads.keys
        gradients = @builder.grad(loss, wrt: names.map { |name| @reads[name] })
        @grads = names.zip(gradients).to_h
        names.each { |name| update(name, @reads[name], @grads[name]) }
        @grads
      end

      # Releases optimizer-owned state buffers such as momentum slots.
      #
      # @return [Hash]
      def free!
        @slots.each_value(&:free)
        @slots = {}
      end

      private

      # One zero-filled buffer per name (Buffer.new zero-fills on allocation).
      def zeros_for(name, suffix)
        buffer = @params.fetch(name)
        @slots[:"#{name}_#{suffix}"] = Buffer.new(buffer.dtype, buffer.length, shape: buffer.shape)
      end

      def slot(name, suffix) = @slot_refs.fetch(:"#{name}_#{suffix}")

      def allocate_slots!
        # No extra state by default (SGD).
      end

      def update(_name, _param, _grad)
        raise NotImplementedError, "#{self.class} must implement #update"
      end
    end

    # Stochastic gradient descent with `p -= lr * gradient` updates.
    #
    # No momentum, deliberately; adding it would take the same shape as Adam
    # (one state + one update line) when it is needed.
    class SGD < Optimizer
      # @param params [Hash{Symbol, String => Buffer}]
      # @param lr [Float] learning rate
      def initialize(params, lr: 1e-2)
        @lr = lr.to_f
        super(params)
      end

      private

      def update(name, param, grad)
        @builder.write_into(@refs.fetch(name), value: param - (@lr * grad))
      end
    end

    # Adam optimizer with bias correction.
    #
    #   m = β1·m + (1-β1)·g,  v = β2·v + (1-β2)·g²
    #   p -= lr · (m / (1-β1^t)) / (sqrt(v / (1-β2^t)) + eps)
    #
    # The bias-correction t is **state inside the graph**. As a plan input,
    # every caller would have to pass a step count only Adam needs. Since t is
    # a run-time value, β^t is written `exp(t · ln β)` (pow is not a primitive
    # and this is the only place that wants it).
    class Adam < Optimizer
      # @param params [Hash{Symbol, String => Buffer}]
      # @param lr [Float] learning rate
      # @param beta1 [Float] first-moment decay
      # @param beta2 [Float] second-moment decay
      # @param eps [Float] denominator stability term
      def initialize(params, lr: 1e-3, beta1: 0.9, beta2: 0.999, eps: 1e-8)
        @lr = lr.to_f
        @beta1 = beta1.to_f
        @beta2 = beta2.to_f
        @eps = eps.to_f
        super(params)
      end

      def attach(builder)
        super
        # t is a one-element state shared by all parameters.
        @step_ref = @slot_refs.fetch(:__step)
        t = builder.read(@step_ref) + 1.0
        builder.write_into(@step_ref, value: t)
        # 1 - β^t. t has shape [1], so it broadcasts into the expressions below.
        @correction1 = 1.0 - (t * Math.log(@beta1)).exp
        @correction2 = 1.0 - (t * Math.log(@beta2)).exp
        @reads
      end

      private

      def allocate_slots!
        @params.each_key do |name|
          zeros_for(name, :m)
          zeros_for(name, :v)
        end
        @slots[:__step] = Buffer.new(:float32, 1, shape: [1])
      end

      def update(name, param, grad)
        m_ref = slot(name, :m)
        v_ref = slot(name, :v)
        m = (@beta1 * @builder.read(m_ref)) + ((1.0 - @beta1) * grad)
        v = (@beta2 * @builder.read(v_ref)) + ((1.0 - @beta2) * grad * grad)
        @builder.write_into(m_ref, value: m)
        @builder.write_into(v_ref, value: v)

        step = (m / @correction1) / ((v / @correction2).sqrt + @eps)
        @builder.write_into(@refs.fetch(name), value: param - (@lr * step))
      end
    end
  end
end
