# frozen_string_literal: true

module Tilework
  module Graph
    # The contract for reverse mode (#22): a vjp table with **one row per op**,
    # keyed by the same names as `Op::REGISTRY`.
    #
    # It is a table for the same reason as Op::REGISTRY: adding an op and
    # forgetting its gradient should fail in `Grad.fetch`, not silently produce
    # a wrong plan. Rows declaring "not differentiable" live in the table too
    # (`opaque`'s contents are unknown, etc.), so failures carry a reason.
    #
    # **Rows are written in the builder's primitives only.** Backward nodes are
    # ordinary nodes, so fusion, placement and liveness all apply, and their
    # gradients can be differentiated again (docs/autograd.md).
    #
    #   Grad.define(:unary) { |g, node, dy, inputs| [...] }
    #
    #   g      the builder (where new nodes are recorded)
    #   node   the forward node
    #   dy     the cotangent arriving at this node's output (Tensor)
    #   inputs node.inputs wrapped as Tensors
    #
    # Returns one cotangent per input. Inputs no gradient flows into (indices,
    # masks, comparison outputs) get nil.
    module Grad
      REGISTRY = {}

      module_function

      def define(name, &vjp)
        raise Error, "the vjp of #{name.inspect} is already defined" if REGISTRY.key?(name)
        raise ArgumentError, "define needs a block" if vjp.nil?

        REGISTRY[name] = vjp
      end

      def fetch(name)
        REGISTRY.fetch(name) do
          raise Error, "no vjp is defined for the graph op #{name.inspect}"
        end
      end

      def defined?(name) = REGISTRY.key?(name)
      def names = REGISTRY.keys

      def apply(builder, node, dy, inputs) = fetch(node.op).call(builder, node, dy, inputs)

      # --- Hand-written vjps for high-level ops --------------------------------
      #
      # The default is "decompose, then differentiate", which gives correct
      # gradients, but expressions with row-wise reductions cost more round
      # trips decomposed, so **only those faster written as an expression** are
      # added here. Functions in this table stay :nn nodes even in a
      # differentiable graph (builder.rb #nn).
      #
      # Adding one follows a single procedure: **the decomposition is the
      # oracle**. With `Graph.lower_high_level = true` its gradient comes from
      # primitive vjps automatically and can be compared with the hand-written
      # row (conformance's graph_hand_written_vjps_match_the_decomposed_ones).
      # Finite differences alone cannot tell whether the fast path or the
      # decomposition is wrong.
      NN_REGISTRY = {}

      def define_nn(fn, &vjp)
        raise Error, "the vjp of NN.#{fn} is already defined" if NN_REGISTRY.key?(fn)

        NN_REGISTRY[fn] = vjp
      end

      def nn_defined?(fn) = NN_REGISTRY.key?(fn)
      def nn_names = NN_REGISTRY.keys

      # Splits a :nn node's inputs into positional and keyword arguments.
      def nn_arguments(node, inputs)
        arity = node.attrs[:arity]
        [inputs.first(arity), node.attrs[:keys].zip(inputs.drop(arity)).to_h]
      end

      # Shared by functions with row-wise reductions: the last dim is columns, the rest rows.
      def row_count(shape) = shape[0..-2].inject(1, :*)

      # The gamma / beta gradient folds over all rows, leaving only columns.
      def fold_rows(tensor)
        flat = tensor.reshape(row_count(tensor.shape), tensor.shape[-1])
        flat.sum(axis: 0)
      end

      # --- Helpers for the rows ----------------------------------------------

      # Folds a cotangent of the broadcast shape back to the original shape.
      # The rule is broadcasting reversed (NumPy right alignment): added leading
      # axes are folded away, and axes that were 1 are folded with keepdims.
      def unbroadcast(dy, shape)
        extra = dy.shape.length - shape.length
        extra.times { dy = dy.sum(axis: 0) }
        shape.each_with_index do |dim, axis|
          dy = dy.sum(axis: axis, keepdims: true) if dim == 1 && dy.shape[axis] != 1
        end
        dy.shape == shape ? dy : dy.reshape(*shape)
      end

      # The inverse of a reduction: spreads values back along the folded axis.
      def spread(dy, node, shape)
        axis = node.attrs[:axis]
        return dy.broadcast_to(shape) if axis.nil?

        # If it was folded with keepdims: false, restore the axis before broadcasting.
        dy = dy.reshape(*shape.each_with_index.map { |dim, i| i == axis ? 1 : dim }) unless node.attrs[:keepdims]
        dy.broadcast_to(shape)
      end

      # The inverse of padding (a crop). `slice` takes only the outermost dim,
      # so cropping an arbitrary axis builds an index table and pulls it with
      # `gather` — the same trick conv2d's decomposition uses for its im2col
      # table, with no constants brought in from the host.
      def crop(builder, dy, before, shape)
        total = shape.inject(1, :*)
        padded = suffix_products(dy.shape)
        index = builder.iota([total], axis: 0)
        flat = nil
        suffix_products(shape).each_with_index do |stride, axis|
          along = index / stride
          coord = along - ((along / shape[axis]) * shape[axis]) # along % shape[axis]
          term = (coord + before[axis]) * padded[axis]
          flat = flat.nil? ? term : flat + term
        end
        builder.gather(dy.reshape(dy.length), flat).reshape(*shape)
      end

      # Per axis, how far the flat index advances for one step.
      def suffix_products(shape)
        shape.each_index.map { |i| shape[(i + 1)..].inject(1, :*) }
      end

      # Non-differentiable ops are collected here with a reason.
      def undefined!(what, reason)
        raise Error, "#{what} has no gradient: #{reason}"
      end
    end

    # --- The table -----------------------------------------------------------
    #
    # Leaves. They have no inputs (or pass no gradient), so backward stops here.

    Grad.define(:input) { |_g, _node, _dy, _inputs| [] }
    Grad.define(:const) { |_g, _node, _dy, _inputs| [] }
    Grad.define(:iota) { |_g, _node, _dy, _inputs| [] }
    Grad.define(:random) { |_g, _node, _dy, _inputs| [] }
    Grad.define(:state_init) { |_g, _node, _dy, _inputs| [] }

    # Reads are treated as leaves: the gradient is taken with respect to the
    # read's result (pass the g.read(ref) Tensor in wrt:). Folding back into the
    # state itself involves versions and regions, so it is not done here.
    Grad.define(:state_read) { |_g, _node, _dy, _inputs| [nil] }

    # Index reductions and comparisons output integers, so no gradient flows.
    Grad.define(:arg_reduce) { |_g, _node, _dy, _inputs| [nil] }

    # --- Elementwise ---------------------------------------------------------

    # Binary ops. With a scalar there is one input, and attrs hold scalar and
    # reversed (reversed means swapped operands, as in `2.0 - x`).
    Grad.define(:binary) do |_g, node, dy, inputs|
      # Comparisons only return 0/1, so no gradient flows.
      next Array.new(node.inputs.length) if node.attrs[:kind] == :compare

      op = node.attrs[:op]
      if node.attrs.key?(:scalar)
        x = inputs[0]
        s = node.attrs[:scalar]
        rev = node.attrs[:reversed]
        [case op
         when :+ then dy
         when :- then rev ? -1.0 * dy : dy            # rev: s - x
         when :* then dy * s
         when :/ then rev ? -1.0 * dy * s / (x * x) : dy / s
         end]
      else
        a, b = inputs
        case op
        when :+ then [dy, dy]
        when :- then [dy, -1.0 * dy]
        when :* then [dy * b, dy * a]
        when :/ then [dy / b, -1.0 * dy * a / (b * b)]
        end
      end
    end

    # Unary math functions. Those that can use their output do (exp / tanh),
    # which saves keeping one more value.
    Grad.define(:unary) do |g, node, dy, inputs|
      x = inputs[0]
      y = Tensor.new(g, node)
      [case node.attrs[:fn]
       when :exp  then dy * y
       when :log  then dy / x
       when :sqrt then dy / (2.0 * y)
       when :sin  then dy * x.cos
       when :cos  then -1.0 * dy * x.sin
       when :tanh then dy * (1.0 - (y * y))
       when :abs  then dy * g.where(x >= 0.0, 1.0, -1.0)
       # Step functions: the derivative is 0 almost everywhere, so return that.
       when :floor, :ceil, :round then g.fill(x.shape, 0.0, dtype: dy.dtype)
       else Grad.undefined!(node.attrs[:fn], "no derivative is registered for it")
       end]
    end

    # It only selects, so the cotangent is routed by the same condition; none
    # flows into the condition itself.
    Grad.define(:where) do |g, node, dy, inputs|
      zero = 0.0
      grads = [nil]
      grads << g.where(inputs[0], dy, zero) unless node.attrs.key?(:a_scalar)
      grads << g.where(inputs[0], zero, dy) unless node.attrs.key?(:b_scalar)
      grads
    end

    # Only changes the storage type. Conversions involving integers are step
    # functions, so no gradient flows.
    Grad.define(:cast) do |_g, node, dy, inputs|
      float = %i[float32 float16]
      next [nil] unless float.include?(node.dtype) && float.include?(inputs[0].dtype)

      [dy.cast(inputs[0].dtype)]
    end

    # --- Reductions ----------------------------------------------------------

    # sum / mean just spread back along the folded axis. min / max pass only to
    # the winning element (on ties **every** winner gets the same cotangent —
    # the one place that does not match finite differences exactly, so avoid ties).
    Grad.define(:reduce) do |g, node, dy, inputs|
      x = inputs[0]
      spread = Grad.spread(dy, node, x.shape)
      [case node.attrs[:kind]
       when :sum then spread
       when :mean
         axis = node.attrs[:axis]
         spread / (axis.nil? ? x.length : x.shape[axis]).to_f
       when :min, :max
         y = Grad.spread(Tensor.new(g, node), node, x.shape)
         g.where(x.eq(y), spread, 0.0)
       end]
    end

    # --- Shape and indexing --------------------------------------------------

    Grad.define(:reshape) { |_g, _node, dy, inputs| [dy.reshape(*inputs[0].shape)] }

    Grad.define(:broadcast) { |_g, _node, dy, inputs| [Grad.unbroadcast(dy, inputs[0].shape)] }

    Grad.define(:permute) do |_g, node, dy, inputs|
      axes = node.attrs[:axes]
      inverse = Array.new(axes.length)
      axes.each_with_index { |from, to| inverse[from] = to }
      [dy.permute(*inverse)]
    end

    # The inverse of a slice puts it back in place and zero-fills the rest.
    # slice takes only the outermost dim, so the fill is outermost-only too.
    Grad.define(:slice) do |g, node, dy, inputs|
      shape = inputs[0].shape
      rows = node.shape[0]
      start = node.attrs[:offset] / shape.drop(1).inject(1, :*)
      before = [start] + Array.new(shape.length - 1, 0)
      after = [shape[0] - start - rows] + Array.new(shape.length - 1, 0)
      [g.pad(dy, before: before, after: after, value: 0.0)]
    end

    # Drops what the border added.
    Grad.define(:pad) do |g, node, dy, inputs|
      [Grad.crop(g, dy, node.attrs[:before], inputs[0].shape)]
    end

    # The inverse of concat slices out each piece's range.
    Grad.define(:concat) do |g, node, dy, inputs|
      axis = node.attrs[:axis]
      offset = 0
      inputs.map do |piece|
        before = Array.new(piece.shape.length, 0)
        before[axis] = offset
        offset += piece.shape[axis]
        Grad.crop(g, dy, before, piece.shape)
      end
    end

    # The inverse of table lookup is scatter-add. It scans the index list per
    # output element without atomics (Kernels.scatter_add), so values do not
    # vary across runs and no [table rows, index count] mask is materialized.
    Grad.define(:gather) do |g, node, dy, inputs|
      source, indices = inputs
      [g.scatter_add(indices.reshape(indices.length), dy.reshape(*source.shape.take(node.attrs[:axis]),
                                                                indices.length,
                                                                *source.shape.drop(node.attrs[:axis] + 1)),
                     shape: source.shape, axis: node.attrs[:axis]),
       nil]
    end

    # scatter_add's own inverse is gather (reading back from where it added).
    # One line so second derivatives do not stop at an embedding layer.
    Grad.define(:scatter_add) do |g, node, dy, inputs|
      [nil, g.gather(dy, inputs[0], axis: node.attrs[:axis])]
    end

    # --- Matrix multiply -----------------------------------------------------
    #
    # dA = dY Bᵀ, dB = Aᵀ dY. Transposes are written with permute — folding
    # them into GEMM transpose flags is a later optimization, and this plain
    # form is its reference (docs/autograd.md, "matmul, and #22's wall 2").
    Grad.define(:matmul) do |_g, node, dy, inputs|
      a, b = inputs
      swap = node.attrs[:mode] == :batched ? [0, 2, 1] : [1, 0]
      [dy.matmul(b.permute(*swap)), a.permute(*swap).matmul(dy)]
    end

    # --- Not differentiable --------------------------------------------------

    # High-level ops default to "decompose, then differentiate". Recording with
    # Graph.build(differentiable: true) yields decompositions, so two kinds stay
    # :nn — functions with a hand-written vjp (NN_REGISTRY below) and functions
    # without a decomposition (gelu_exact / groupnorm). The former go to their
    # row; the latter fail with a reason.
    Grad.define(:nn) do |g, node, dy, inputs|
      fn = node.attrs[:fn]
      vjp = Grad::NN_REGISTRY[fn]
      if vjp.nil?
        Grad.undefined!("NN.#{fn}",
                        "it has no decomposition into primitives, so it cannot be differentiated " \
                        "(record the graph with Graph.build(differentiable: true) for the ops that do)")
      end

      positional, keyed = Grad.nn_arguments(node, inputs)
      grads = vjp.call(g, node, dy, positional, keyed)
      # The result is {input -> cotangent}; put it back in the node's input
      # order (positional, then keyword).
      positional.each_index.map { |i| grads[i] } + node.attrs[:keys].map { |key| grads[key] }
    end

    # --- Hand-written vjps ---------------------------------------------------
    #
    # Each is "row-wise reduction + elementwise", so row fusion folds it into
    # one or two kernels. The only difference from differentiating the
    # decomposition is the number of round trips; the values agree
    # (conformance compares them).

    # dx = (dy - Σ(dy·y)) · y; one fold per row is enough.
    Grad.define_nn(:softmax) do |_g, node, dy, positional, _keyed|
      y = Tensor.new(positional[0].graph, node)
      { 0 => (dy - (dy * y).sum(axis: -1, keepdims: true)) * y }
    end

    # y = x̂·γ + β, x̂ = (x - μ)/σ.
    # dx = (g - mean(g) - x̂·mean(g·x̂)) / σ, g = dy·γ.
    # The γ / β gradients are folded over rows.
    Grad.define_nn(:layernorm) do |_g, node, dy, positional, keyed|
      x = positional[0]
      eps = node.attrs[:opts].fetch(:eps)
      centered = x - x.mean(axis: -1, keepdims: true)
      sigma = ((centered * centered).mean(axis: -1, keepdims: true) + eps).sqrt
      hat = centered / sigma
      inner = keyed[:gamma] ? dy * keyed[:gamma] : dy

      grads = { 0 => (inner - inner.mean(axis: -1, keepdims: true) -
                      (hat * (inner * hat).mean(axis: -1, keepdims: true))) / sigma }
      grads[:gamma] = Grad.fold_rows(dy * hat) if keyed[:gamma]
      grads[:beta] = Grad.fold_rows(dy) if keyed[:beta]
      grads
    end

    # y = x·γ/r, r = sqrt(mean(x²) + eps). ∂r/∂x_j = x_j/(n·r), so
    # dx = (g - x·Σ(g·x)/(n·r²)) / r, g = dy·γ.
    Grad.define_nn(:rmsnorm) do |_g, node, dy, positional, keyed|
      x = positional[0]
      eps = node.attrs[:opts].fetch(:eps)
      r = ((x * x).mean(axis: -1, keepdims: true) + eps).sqrt
      inner = keyed[:gamma] ? dy * keyed[:gamma] : dy

      grads = { 0 => (inner - (x * ((inner * x).mean(axis: -1, keepdims: true) / (r * r)))) / r }
      grads[:gamma] = Grad.fold_rows(dy * (x / r)) if keyed[:gamma]
      grads
    end

    # (softmax(x) - onehot) · scale. **This form is why cross_entropy takes
    # logits** — differentiating log(softmax(x)) reaches the same value but
    # passes through reciprocals of probabilities and loses digits.
    Grad.define_nn(:cross_entropy) do |g, node, dy, positional, _keyed|
      logits, labels = positional
      cols = logits.shape[-1]
      rows_shape = logits.shape[0..-2]
      reduction = node.attrs[:opts].fetch(:reduction)

      clamped = g.where(labels < 0, 0, labels)
      clamped = g.where(clamped > (cols - 1), cols - 1, clamped)
      onehot = g.where(g.iota(logits.shape, axis: -1).eq(clamped.reshape(*rows_shape, 1)), 1.0, 0.0)

      # The reduction contributes a scale; for :none each row's cotangent goes to that row.
      scale = case reduction
              when :none then dy.reshape(*rows_shape, 1)
              when :sum  then dy
              else dy / Grad.row_count(logits.shape).to_f
              end
      { 0 => (Tilework::NN.softmax(logits) - onehot) * scale, 1 => nil }
    end

    # mse is elementwise only, so its decomposition is fast enough and it has no hand-written row.

    # Convolution uses the delegate's backward kernels (cuDNN / MPSGraph) as
    # is. On a backend without convolution :conv2d is never recorded (it is
    # opened into im2col), so primitive vjps handle it there (which is why the
    # gather row is needed).
    Grad.define(:conv2d) do |g, node, dy, inputs|
      x, weight = inputs
      pad = node.attrs[:pad]
      stride = node.attrs[:stride]
      dilation = node.attrs[:dilation]
      groups = node.attrs[:groups]
      [g.conv2d_backward(:data, weight, dy, x, pad: pad, stride: stride, dilation: dilation, groups: groups),
       g.conv2d_backward(:filter, x, dy, weight, pad: pad, stride: stride, dilation: dilation, groups: groups)]
    end

    # The backward nodes themselves are not differentiated again (there is no
    # second-order convolution kernel).
    %i[conv2d_backward_data conv2d_backward_filter].each do |op|
      Grad.define(op) do |_g, _node, _dy, _inputs|
        Grad.undefined!(op.to_s, "the delegated convolution kernels have no second derivative")
      end
    end

    Grad.define(:opaque) do |_g, node, _dy, _inputs|
      Grad.undefined!("opaque #{node.attrs[:name].inspect}", "the graph cannot see inside it")
    end

    # Writes to mutable state follow memory, not values. Backward through them
    # would need folding across versions and regions, so they fail for now
    # (#22 Step 3).
    %i[state_write state_merge].each do |op|
      Grad.define(op) do |_g, _node, _dy, _inputs|
        Grad.undefined!(op.to_s, "differentiating through a state write is not supported yet (#22 Step 3)")
      end
    end
  end
end
