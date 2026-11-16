# frozen_string_literal: true

module Tilework
  # Two-dimensional NCHW convolution provided by an available backend library.
  #
  # Tensors are **NCHW**:
  #   input [N, C_in, H, W] / weight [C_out, C_in, kh, kw] / output [N, C_out, H', W']
  #
  # **Do not hand-write im2col + GEMM.** Measured, im2col takes 74–84% of the
  # time: GEMM alone reaches 2581 GF/s but the whole convolution drops to
  # 613 GF/s, because im2col writes out kh*kw times the data and is memory-
  # bound. cuDNN's implicit GEMM has no expansion at all and reaches
  # fp16 4747 GF/s (7.7x) on the same shape.
  module Conv
    FP32 = 0
    FP16 = 1

    module_function

    # @return [Boolean] whether the active backend exposes forward convolution
    def available?
      return false unless Tilework.backend
      return false unless Tilework.bridge.respond_to?(:conv2d_available?)

      Tilework.bridge.conv2d_available?
    end

    # @return [Symbol, nil] backend convolution library identifier
    def library
      return nil unless available?
      return Tilework.bridge.conv_library_name if Tilework.bridge.respond_to?(:conv_library_name)

      :cudnn
    end

    # @return [Boolean] whether the active backend exposes convolution backward passes
    #
    # Backward is a separate optional ABI from forward; a bridge may have only
    # forward, so it is queried independently.
    def backward_available?
      return false unless available?
      return false unless Tilework.bridge.respond_to?(:conv2d_backward_available?)

      Tilework.bridge.conv2d_backward_available?
    end

    # Calculates one output spatial dimension.
    #
    # @param size [Integer] input size
    # @param kernel [Integer] kernel size
    # @param pad [Integer] padding on each side
    # @param stride [Integer]
    # @param dilation [Integer]
    # @return [Integer]
    def output_size(size, kernel, pad, stride, dilation = 1)
      effective_kernel = ((kernel - 1) * dilation) + 1
      ((size + (2 * pad) - effective_kernel) / stride) + 1
    end

    # Performs NCHW convolution.
    #
    # @param input [Buffer, Graph::Tensor] `[N, C_in, H, W]`
    # @param weight [Buffer, Graph::Tensor] `[C_out, C_in / groups, kh, kw]`
    # @param pad [Integer, Array<Integer>]
    # @param stride [Integer, Array<Integer>]
    # @param dilation [Integer, Array<Integer>]
    # @param groups [Integer]
    # @return [Buffer, Graph::Tensor]
    #
    # padding / stride / dilation take a scalar (same for both axes) or [h, w].
    # `groups` is the group count (`groups == cin` for depthwise).
    #
    #   x = Tilework.randn([1, 3, 256, 256])     # NCHW
    #   w = Tilework.randn([64, 3, 3, 3])        # C_out, C_in, kh, kw
    #   y = Tilework::Conv.forward(x, w, pad: 1) # => [1, 64, 256, 256]
    #
    # Input and weight dtypes must match. float16 takes the tensor-core path,
    # but **accumulation is always FP32**. With `groups > 1` the weight is
    # `[C_out, C_in / groups, kh, kw]`.
    def forward(input, weight, pad: 0, stride: 1, dilation: 1, groups: 1)
      n, cin, h, w = check_input!(input)
      cout, wcin, kh, kw = check_weight!(weight, cin)
      pad_h, pad_w = pair(pad)
      stride_h, stride_w = pair(stride)
      dilation_h, dilation_w = pair(dilation)

      check_groups!(cin, cout, wcin, groups)

      oh = output_size(h, kh, pad_h, stride_h, dilation_h)
      ow = output_size(w, kw, pad_w, stride_w, dilation_w)
      unless oh.positive? && ow.positive?
        raise ArgumentError, "kernel #{kh}x#{kw} with pad #{pad_h},#{pad_w} is larger than the input #{h}x#{w}"
      end

      dtype = check_dtypes!(input, weight)
      # Record into the graph. On a backend without convolution it is
      # decomposed into primitives (im2col + matmul) at record time, so it
      # works on :cpu too (graph/high_level.rb's :conv2d).
      if Graph.symbolic?(input, weight)
        return Graph.record_conv2d(input, weight, shape: [n, cout, oh, ow],
                                                  pad: [pad_h, pad_w], stride: [stride_h, stride_w],
                                                  dilation: [dilation_h, dilation_w], groups: groups)
      end

      require_available!
      unless Tilework.capabilities.conv_dtypes.include?(input.dtype)
        raise Error, "#{Tilework.capabilities.conv} convolution does not support #{input.dtype.inspect} " \
                     "on the #{Tilework.backend.inspect} backend; use Graph for the portable decomposition"
      end
      # Take the output from the pool. A direct Buffer.new would never reuse the
      # slots Plan#release returns to the pool, so they would pile up every run.
      out = Ops.allocate_output(input.dtype, n * cout * oh * ow, [n, cout, oh, ow])
      Tilework.bridge.conv2d(input.__handle__, weight.__handle__, out.__handle__,
                        [n, cin, h, w, cout, kh, kw, pad_h, pad_w, stride_h, stride_w,
                         dilation_h, dilation_w, groups], dtype)
      out
    rescue StandardError
      out&.free
      raise
    end

    # Calculates input and filter gradients for NCHW convolution.
    #
    # @param input [Buffer]
    # @param weight [Buffer]
    # @param grad_output [Buffer]
    # @param pad [Integer, Array<Integer>]
    # @param stride [Integer, Array<Integer>]
    # @param dilation [Integer, Array<Integer>]
    # @param groups [Integer]
    # @return [Array<Buffer>] input gradient and filter gradient
    #
    # Takes the forward arguments plus the output gradient `grad_output` and
    # returns **the input gradient and the weight gradient**.
    #
    #   y  = Tilework::Conv.forward(x, w, pad: 1)
    #   dx, dw = Tilework::Conv.backward(x, w, dy, pad: 1)
    #
    # Descriptors are shared with the forward plan, so **a forward pass of the
    # same shape must have run once first** — which a training loop always does.
    def backward(input, weight, grad_output, pad: 0, stride: 1, dilation: 1, groups: 1)
      data = backward_data(weight, grad_output, input, pad: pad, stride: stride, dilation: dilation, groups: groups)
      filter = backward_filter(input, grad_output, weight, pad: pad, stride: stride, dilation: dilation, groups: groups)
      [data, filter]
    rescue StandardError
      data&.free
      filter&.free
      raise
    end

    # Input gradient dx. `like` is a Buffer shaped like the input (used only for its shape).
    def backward_data(weight, grad_output, like, pad: 0, stride: 1, dilation: 1, groups: 1)
      dims, dtype = backward_setup!(like, weight, grad_output, pad, stride, dilation, groups)
      out = Ops.allocate_output(like.dtype, like.length, like.shape)
      Tilework.bridge.conv2d_backward(weight.__handle__, grad_output.__handle__, out.__handle__,
                                 dims, dtype, 0)
      out
    rescue StandardError
      out&.free
      raise
    end

    # Weight gradient dw. `like` is a Buffer shaped like the weight.
    def backward_filter(input, grad_output, like, pad: 0, stride: 1, dilation: 1, groups: 1)
      dims, dtype = backward_setup!(input, like, grad_output, pad, stride, dilation, groups)
      out = Ops.allocate_output(like.dtype, like.length, like.shape)
      Tilework.bridge.conv2d_backward(input.__handle__, grad_output.__handle__, out.__handle__,
                                 dims, dtype, 1)
      out
    rescue StandardError
      out&.free
      raise
    end

    # --- Internal ---

    # Validates backward arguments and builds dims. Besides the forward checks,
    # grad_output must have the forward output's shape (otherwise cuDNN quietly
    # reads somewhere else).
    def backward_setup!(input, weight, grad_output, pad, stride, dilation = 1, groups = 1)
      unless backward_available?
        raise Error, "convolution backward passes are not available on the " \
                     "#{Tilework.backend.inspect} backend (cuDNN is used on :cuda)"
      end
      if [input, weight, grad_output].any? { |buffer| buffer.dtype == :bfloat16 }
        raise Error, "bfloat16 convolution backward is not supported; keep training buffers in :float32"
      end

      n, cin, h, w = check_input!(input)
      cout, wcin, kh, kw = check_weight!(weight, cin)
      pad_h, pad_w = pair(pad)
      stride_h, stride_w = pair(stride)
      dilation_h, dilation_w = pair(dilation)

      check_groups!(cin, cout, wcin, groups)

      oh = output_size(h, kh, pad_h, stride_h, dilation_h)
      ow = output_size(w, kw, pad_w, stride_w, dilation_w)
      expected = [n, cout, oh, ow]
      unless grad_output.shape == expected
        raise ArgumentError,
              "grad_output shape #{grad_output.shape.inspect} does not match the forward output #{expected.inspect}"
      end

      dtype = check_dtypes!(input, weight)
      unless grad_output.dtype == input.dtype
        raise ArgumentError,
              "grad_output dtype #{grad_output.dtype.inspect} does not match the input's #{input.dtype.inspect}"
      end

      [[n, cin, h, w, cout, kh, kw, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, groups], dtype]
    end


    def pair(value)
      value.is_a?(Array) ? value : [value, value]
    end

    # Validates `groups`: cin and cout must both divide, and the weight's input
    # channel count must equal cin/groups. With groups == 1 this is the usual
    # "weight input channels == cin" check.
    def check_groups!(cin, cout, wcin, groups)
      unless groups.positive? && (cin % groups).zero? && (cout % groups).zero?
        raise ArgumentError,
              "groups #{groups} must evenly divide both input channels #{cin} and output channels #{cout}"
      end

      expected_wcin = cin / groups
      return if wcin == expected_wcin

      raise ArgumentError,
            "weight input channels #{wcin} do not match input channels #{cin} / groups #{groups} (#{expected_wcin})"
    end

    def check_input!(input)
      return input.shape if input.shape.length == 4

      raise ArgumentError, "conv2d input must be 4D [N, C, H, W], got #{input.shape.inspect}"
    end

    def check_weight!(weight, _cin)
      return weight.shape if weight.shape.length == 4

      raise ArgumentError,
            "conv2d weight must be 4D [C_out, C_in, kh, kw], got #{weight.shape.inspect}"
    end

    def check_dtypes!(input, weight)
      unless input.dtype == weight.dtype
        raise ArgumentError,
              "input and weight dtypes must match (got #{input.dtype.inspect} and #{weight.dtype.inspect})"
      end

      case input.dtype
      when :float32 then FP32
      when :float16 then FP16
      when :bfloat16 then 2
      else
        raise Error, "conv2d supports :float32, :float16, and :bfloat16 (got #{input.dtype.inspect})"
      end
    end

    def require_available!
      return if available?

      raise Error, "no convolution library is available on the #{Tilework.backend.inspect} backend " \
                   "(cuDNN is used on :cuda)"
    end
  end
end
