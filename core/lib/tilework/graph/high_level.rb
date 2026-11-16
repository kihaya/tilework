# frozen_string_literal: true

module Tilework
  module Graph
    # The contract for high-level ops (Tier 1 in docs/graph-ir.md).
    #
    # A function like `NN.softmax` is high-level **because it is fast**, not
    # because it cannot be written any other way — that is this layer's
    # premise, and this table makes it checkable. Each row says how to write
    # the function in primitives alone.
    #
    #   HighLevel.define(:softmax) do |_g, inputs, _tensors, _args, _opts|
    #     e = (inputs[0] - inputs[0].max(axis: -1, keepdims: true)).exp
    #     e / e.sum(axis: -1, keepdims: true)
    #   end
    #
    # The default execution path is still the hand-written fused kernel (a
    # `:nn` node); the decomposition does not run. Writing it down still pays
    # three ways:
    #
    #   1. **Portability** — there is a form that runs on a backend without
    #      that fused kernel (every backend has them today, so it is unused)
    #   2. **Backward (#22)** — gradients follow from primitive vjps alone, so a
    #      high-level op can be "decomposed, then differentiated"; hand-written
    #      vjps are added only where speed demands
    #   3. **Oracle** — `Graph.lower_high_level = true` switches to the
    #      decomposition to compare against the fused kernel, the same trick as
    #      verifying Fusion on :cpu, so both paths are checked without a GPU in CI
    #
    # Agreement is to a tolerance, not bits: the decomposition reduces in a
    # different order and accumulates in a different type than the hand-written
    # kernel, which cannot be matched.
    #
    # **Some functions have no decomposition.** `gelu_exact` needs erf, which
    # is not a primitive, and `groupnorm` folds across channels, which is not a
    # row-wise reduction. Functions not in the table stay `:nn` nodes — the
    # table never pretends to express what it cannot.
    module HighLevel
      REGISTRY = {}
      # The capability the fast path needs. None means every backend has it
      # (NN's fused kernels can be generated anywhere).
      REQUIREMENTS = {}

      module_function

      def define(name, requires: nil, &lower)
        raise Error, "high-level op #{name.inspect} is already defined" if REGISTRY.key?(name)
        raise ArgumentError, "define needs a block" if lower.nil?

        REGISTRY[name] = lower
        REQUIREMENTS[name] = requires if requires
      end

      def defined?(name) = REGISTRY.key?(name)
      def names = REGISTRY.keys

      # Decompose in two cases: **explicitly switched on as the oracle**, and
      # **this backend lacks the fast path**. The latter is portability itself;
      # conv2d is the first instance (:cpu has no convolution).
      def lower?(name)
        return false unless REGISTRY.key?(name)

        Graph.lower_high_level || !available?(name)
      end

      # Whether the fast path is available. A function with no requirement always is.
      def available?(name)
        capability = REQUIREMENTS[name]
        capability.nil? || Op.satisfied?(capability)
      end

      def lower(builder, name, inputs, tensors:, args:, opts:)
        REGISTRY.fetch(name).call(builder, inputs, tensors, args, opts)
      end
    end

    # Every activation is a single expression, so they are written together
    # here, in the same form as nn.rb's ACTIVATIONS. sigmoid / silu split on
    # the sign because `exp(-v)` overflows for very negative v (that side uses
    # the rearranged form).
    ACTIVATION_EXPRS = {
      relu: ->(g, x) { g.where(x > 0.0, x, 0.0) },
      sigmoid: ->(g, x) { g.where(x >= 0.0, 1.0 / (1.0 + (-1.0 * x).exp), x.exp / (1.0 + x.exp)) },
      silu: ->(g, x) { x * ACTIVATION_EXPRS[:sigmoid].call(g, x) },
      gelu_tanh: lambda { |_g, x|
        inner = 0.7978845608028654 * (x + (0.044715 * x * x * x))
        0.5 * x * (1.0 + inner.tanh)
      },
    }.freeze

    HighLevel.define(:softmax) do |_g, inputs, _tensors, _args, _opts|
      x = inputs[0]
      # Subtract the max before exp, like the hand-written kernel (prevents overflow).
      e = (x - x.max(axis: -1, keepdims: true)).exp
      e / e.sum(axis: -1, keepdims: true)
    end

    HighLevel.define(:layernorm) do |_g, inputs, tensors, _args, opts|
      x = inputs[0]
      centered = x - x.mean(axis: -1, keepdims: true)
      variance = (centered * centered).mean(axis: -1, keepdims: true)
      y = centered / (variance + opts.fetch(:eps)).sqrt
      y *= tensors[:gamma] if tensors[:gamma]
      y += tensors[:beta] if tensors[:beta]
      y
    end

    HighLevel.define(:rmsnorm) do |_g, inputs, tensors, _args, opts|
      x = inputs[0]
      # No mean subtraction (that is the definition of rmsnorm).
      y = x / ((x * x).mean(axis: -1, keepdims: true) + opts.fetch(:eps)).sqrt
      y *= tensors[:gamma] if tensors[:gamma]
      y
    end

    # Loss reduction. Anything but :none folds everything into shape [1] at the end.
    LOSS_REDUCE = lambda { |value, reduction|
      case reduction
      when :none then value
      when :sum  then value.sum
      else value.mean
      end
    }

    HighLevel.define(:mse) do |_g, inputs, _tensors, _args, opts|
      # Losses are kept in float (the same rule as nn.rb's loss_dtype).
      pred, target = inputs.map { |t| Ops::LOW_PRECISION_DTYPES.include?(t.dtype) ? t.cast(:float32) : t }
      d = pred - target
      LOSS_REDUCE.call(d * d, opts.fetch(:reduction))
    end

    # log-sum-exp in plain ops. Like the hand-written kernel it subtracts the
    # max before exp, so large logits do not overflow.
    #
    # The label's logit is picked with an iota equality mask rather than
    # gather. That needs only elementwise ops and a row reduction, so it
    # differentiates without going through gather's vjp (scatter-add).
    # Clamping out-of-range labels to the ends also matches the hand-written kernel.
    HighLevel.define(:cross_entropy) do |g, inputs, _tensors, _args, opts|
      logits, labels = inputs
      x = Ops::LOW_PRECISION_DTYPES.include?(logits.dtype) ? logits.cast(:float32) : logits
      cols = x.shape[-1]
      rows_shape = x.shape[0..-2]

      m = x.max(axis: -1, keepdims: true)
      lse = m + (x - m).exp.sum(axis: -1, keepdims: true).log

      clamped = g.where(labels < 0, 0, labels)
      clamped = g.where(clamped > (cols - 1), cols - 1, clamped)
      picked = g.where(g.iota(x.shape, axis: -1).eq(clamped.reshape(*rows_shape, 1)), x, 0.0)
                .sum(axis: -1, keepdims: true)

      LOSS_REDUCE.call((lse - picked).reshape(*rows_shape), opts.fetch(:reduction))
    end

    HighLevel.define(:transpose) do |g, inputs, _tensors, _args, _opts|
      g.permute(inputs[0], 1, 0)
    end

    # Heads split the last dim evenly ([n, d] with d = heads x dh). Split
    # (reshape), bring heads to the front (permute), compute all heads with a
    # batched matmul, then restore. bias arrives as [heads * n, m], which is
    # heads stacked vertically, so reading it as [heads, n, m] is enough (same
    # element order).
    HighLevel.define(:multi_head_attention) do |_g, inputs, tensors, _args, opts|
      q, k, v = inputs
      heads = opts.fetch(:heads)
      n, d = q.shape
      m = k.shape[0]
      dh = d / heads

      scores = q.reshape(n, heads, dh).permute(1, 0, 2)     # [h, n, dh]
                .matmul(k.reshape(m, heads, dh).permute(1, 2, 0)) * opts.fetch(:scale)
      scores += tensors[:bias].reshape(heads, n, m) if tensors[:bias]
      NN.softmax(scores)                                     # rows = last dim
        .matmul(v.reshape(m, heads, dh).permute(1, 0, 2))    # [h, n, dh]
        .permute(1, 0, 2).reshape(n, d)
    end

    # Opens a 2D convolution into im2col + matmul.
    #
    # It never runs while a fast path (cuDNN / MPSGraph) exists. It is here so
    # graphs run on a backend without convolution (:cpu) and so backward
    # follows from primitive vjps alone.
    #
    # The index table is built inside the graph from iota and integer
    # arithmetic (recording touches no device).
    HighLevel.define(:conv2d, requires: :conv) do |g, inputs, _tensors, _args, opts|
      x, weight = inputs
      n, cin, h, w = x.shape
      cout, _, kh, kw = weight.shape
      ph, pw = opts.fetch(:pad)
      sh, sw = opts.fetch(:stride)
      dh, dw = opts.fetch(:dilation, [1, 1])
      groups = opts.fetch(:groups, 1)
      # dilation acts through the effective kernel width, leaving (d - 1) gaps between taps.
      effective_kh = ((kh - 1) * dh) + 1
      effective_kw = ((kw - 1) * dw) + 1
      hp = h + (2 * ph)
      wp = w + (2 * pw)
      oh = ((hp - effective_kh) / sh) + 1
      ow = ((wp - effective_kw) / sw) + 1
      rows = n * oh * ow

      padded = g.pad(x, before: [0, 0, ph, pw], after: [0, 0, ph, pw], value: 0.0)
                .reshape(n * cin * hp * wp)

      modulo = ->(value, by) { value - ((value / by) * by) }

      # im2col + matmul for one group. channel_offset is the channel of the
      # padded input where that group starts (always 0 when groups == 1).
      im2col_matmul = lambda do |weight_slice, cin_group, cout_group, channel_offset|
        k = cin_group * kh * kw
        r = g.iota([rows, k], axis: 0)
        c = g.iota([rows, k], axis: 1)
        batch = r / (oh * ow)
        out_y = modulo.call(r / ow, oh)
        out_x = modulo.call(r, ow)
        channel = (c / (kh * kw)) + channel_offset
        tap_y = modulo.call(c / kw, kh)
        tap_x = modulo.call(c, kw)

        index = (batch * (cin * hp * wp)) + (channel * (hp * wp)) +
                (((out_y * sh) + (tap_y * dh)) * wp) + ((out_x * sw) + (tap_x * dw))

        g.gather(padded, index)                                       # [rows, k]
         .matmul(weight_slice.reshape(cout_group, k).permute(1, 0))  # [rows, cout_group]
      end

      # Open output row r = (n, oh, ow) and column c = (cin_per_group, kh, kw)
      # into indices. % is not a primitive, so it is written a - (a / b) * b
      # (integer division truncates the same on the dialect side and on :cpu).
      cols = if groups == 1
               im2col_matmul.call(weight, cin, cout, 0)
             else
               # groups > 1: build an independent im2col + matmul per group and
               # concatenate along cout (axis 1, not the outermost). Groups'
               # input channel ranges do not overlap, so this is correct.
               cin_per_group = cin / groups
               cout_per_group = cout / groups
               group_outputs = (0...groups).map do |gi|
                 first = gi * cout_per_group
                 weight_slice = weight.slice(first..(first + cout_per_group - 1))
                 im2col_matmul.call(weight_slice, cin_per_group, cout_per_group, gi * cin_per_group)
               end
               g.concat(group_outputs, axis: 1)
             end

      cols.permute(1, 0).reshape(cout, n, oh, ow)
          .permute(1, 0, 2, 3)
    end

    HighLevel.define(:activation) do |g, inputs, _tensors, args, _opts|
      kind = args.first
      expr = ACTIVATION_EXPRS[kind]
      # gelu_exact cannot be decomposed (erf is not a primitive). Failing beats
      # silently substituting another approximation.
      if expr.nil?
        raise Error, "#{kind} has no decomposition (erf is not a primitive), so it cannot run " \
                     "with Graph.lower_high_level = true"
      end

      expr.call(g, inputs[0])
    end
  end
end
