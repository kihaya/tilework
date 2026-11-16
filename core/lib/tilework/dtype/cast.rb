# frozen_string_literal: true

module Tilework
  # dtype conversion on the GPU.
  #
  # The pure-Ruby Tilework::Half does per-element bit manipulation, so loading
  # large weights is host-bound. The conversion is element-independent, so it
  # can go to the GPU.
  #
  # **Both round to nearest even, so the results are bit-identical.**
  module Cast
    # source -> target. The kernels are written in the CUDA-flavored dialect,
    # so the same source compiles on every backend with a compat header. A C
    # cast to / from GPU_HALF becomes __half's converting constructor
    # (= __float2half) on CUDA and half's implicit conversion in MSL — both
    # round to nearest even.
    CONVERSIONS = {
      %i[float32 float16] => { in: "float", out: "GPU_HALF" },
      %i[float16 float32] => { in: "GPU_HALF", out: "float" },
      %i[float32 bfloat16] => { in: "float", out: "GPU_BFLOAT" },
      %i[bfloat16 float32] => { in: "GPU_BFLOAT", out: "float" },
    }.freeze

    BLOCK = 256

    module_function

    # Being in the table is not enough: a backend that does not accept dialect
    # kernels (:cpu), or cannot allocate the dtype, cannot convert. Without this
    # capability gate Buffer#gpu_conversion? would answer yes and hit a
    # CompileError.
    def supported?(from, to)
      return false unless CONVERSIONS.key?([from, to])

      caps = Tilework.capabilities
      caps[:dialect] && caps[:dtypes].include?(from) && caps[:dtypes].include?(to)
    end

    def convert(buffer, target)
      # Inside a graph the result is an ordinary node output that returns to
      # the pool. Allocating it directly would never borrow that slot back, so
      # the pool would grow on every repeated conversion (e.g. a low-precision
      # matmul's out_dtype).
      out = Ops.allocate_output(target, buffer.length, buffer.shape)
      if Tilework.host_fallback?
        out.write(buffer.to_a)
        return out
      end
      convert_into(buffer, out)
      out
    rescue StandardError
      out&.free
      raise
    end

    # Converts into an existing Buffer. Buffer#write uses it to fill fp16 via a
    # temporary float32 buffer (it cannot create a new Buffer there).
    def convert_into(src, dst)
      spec = CONVERSIONS[[src.dtype, dst.dtype]] or
        raise Error, "cannot convert #{src.dtype.inspect} to #{dst.dtype.inspect} " \
                     "(supported: #{CONVERSIONS.keys.map { |f, t| "#{f}->#{t}" }.join(', ')})"
      unless supported?(src.dtype, dst.dtype)
        raise Error, "the #{Tilework.backend.inspect} backend cannot convert " \
                     "#{src.dtype.inspect} to #{dst.dtype.inspect}"
      end
      unless src.length == dst.length
        raise ArgumentError, "length mismatch: #{src.length} -> #{dst.length}"
      end

      kernel = Tilework.compile(source(spec), name: "cast_kernel")
      Ops.launch_kernel(kernel, grid: (src.length / BLOCK.to_f).ceil, block: BLOCK,
                            args: [src, dst, src.length])
      dst
    end

    def source(spec)
      <<~KERNEL
        GPU_KERNEL void cast_kernel(GPU_ARGS,
                                    device const #{spec[:in]}* src,
                                    device #{spec[:out]}* dst,
                                    constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) dst[i] = (#{spec[:out]})src[i];
        }
      KERNEL
    end
  end
end
