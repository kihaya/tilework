# frozen_string_literal: true

module Tilework
  # Fast Fourier transforms provided by an available backend library.
  #
  # **Complex numbers are interleaved float32.** A complex array of length N
  # is a `:float32` Buffer of length 2N laid out as `[re0, im0, re1, im1, ...]`.
  # That is exactly cuFFT's `cufftComplex`, so a Buffer can be passed as is
  # (a complex dtype would ripple through the whole Ops / DSL type system).
  #
  # Intended for audio and signal processing (spectrogram → MFCC →
  # classification) and large-kernel convolution. **Not for a diffusion
  # model's 3x3 convolutions** — FFT convolution only wins at kernel sizes of
  # about 15–20 and up; small kernels are faster as direct (im2col + GEMM).
  module FFT
    R2C = 0
    C2R = 1
    C2C = 2
    FORWARD = 1
    INVERSE = -1

    module_function

    # @return [Boolean] whether the active backend exposes an FFT library
    def available?
      return false unless Tilework.backend
      return false unless Tilework.bridge.respond_to?(:fft_available?)

      Tilework.bridge.fft_available?
    end

    # @return [Symbol, nil] backend FFT library identifier
    def library
      return nil unless available?
      return Tilework.bridge.fft_library_name if Tilework.bridge.respond_to?(:fft_library_name)

      :cufft
    end

    # Performs a real-to-complex transform.
    #
    # @param input [Buffer] real float32 values
    # @param batch [Integer] number of equal-length signals
    # @return [Buffer] interleaved float32 complex spectrum
    # @raise [Error] if no FFT library is available
    #
    # Real -> complex (R2C). A real signal of length N gives a complex spectrum
    # of length **N/2+1** (a real signal's spectrum is conjugate-symmetric, so
    # the rest is redundant).
    #
    #   spec = Tilework::FFT.forward(signal)        # float32 of length 2*(N/2+1)
    #   Tilework::FFT.magnitude(spec)               # magnitude spectrum (length N/2+1)
    #
    # batch: transforms several equal-length signals at once (for spectrograms).
    def forward(input, batch: 1)
      require_available!
      n = input.length / batch
      check_divisible!(input.length, batch, "input")

      out = Buffer.new(:float32, 2 * (n / 2 + 1) * batch)
      Tilework.bridge.fft(input.__handle__, out.__handle__, n, batch, R2C, FORWARD)
      out
    rescue StandardError
      out&.free
      raise
    end

    # Performs a complex-to-real inverse transform.
    #
    # @param input [Buffer] interleaved float32 complex spectrum
    # @param n [Integer] original real signal length
    # @param batch [Integer] number of signals
    # @param normalize [Boolean] scale the inverse by `1 / n`
    # @return [Buffer]
    # @note The backend may modify the input buffer.
    #
    # cuFFT does not normalize, so 1/n is applied by default to make
    # `inverse(forward(x)) == x` hold (the same convention as numpy's ifft).
    # Pass normalize: false for raw cuFFT values.
    #
    # **The input complex buffer may be overwritten** (cuFFT's C2R behavior).
    # Copy it first if it is needed afterwards.
    def inverse(input, n, batch: 1, normalize: true)
      require_available!
      expected = 2 * (n / 2 + 1) * batch
      unless input.length == expected
        raise ArgumentError,
              "expected #{expected} floats for n=#{n} batch=#{batch}, got #{input.length}"
      end

      out = Buffer.new(:float32, n * batch)
      Tilework.bridge.fft(input.__handle__, out.__handle__, n, batch, C2R, INVERSE)
      normalize ? scaled(out, 1.0 / n) : out
    rescue StandardError
      out&.free
      raise
    end

    # Performs a complex-to-complex transform.
    #
    # @param input [Buffer] interleaved float32 complex values
    # @param direction [:forward, :inverse]
    # @param batch [Integer]
    # @param normalize [Boolean] scale inverse output by `1 / n`
    # @return [Buffer]
    def transform(input, direction: :forward, batch: 1, normalize: true)
      require_available!
      check_even!(input.length)
      n = input.length / 2 / batch
      check_divisible!(input.length / 2, batch, "input (complex)")

      dir = direction == :inverse ? INVERSE : FORWARD
      out = Buffer.new(:float32, input.length)
      Tilework.bridge.fft(input.__handle__, out.__handle__, n, batch, C2C, dir)
      dir == INVERSE && normalize ? scaled(out, 1.0 / n) : out
    rescue StandardError
      out&.free
      raise
    end

    # Returns magnitudes for interleaved complex values.
    #
    # @param complex [Buffer] interleaved float32 complex values
    # @return [Buffer] real float32 magnitudes
    def magnitude(complex)
      check_even!(complex.length)
      n = complex.length / 2
      magnitude_kernel.(complex, n, over: n)
    end

    # --- Internal ---

    def magnitude_kernel
      @magnitude_kernel ||= Tilework.kernel("fft_magnitude", out: 1) do |z, out, n|
        i = global_id_x
        if i < n
          re = z[2 * i]
          im = z[(2 * i) + 1]
          out[i] = Math.sqrt((re * re) + (im * im))
        end
      end
    end

    def scaled(buffer, factor)
      scaled = buffer * factor
      buffer.free
      scaled
    rescue StandardError
      buffer.free
      raise
    end

    def require_available!
      return if available?

      raise Error, "no FFT library is available on the #{Tilework.backend.inspect} backend " \
                   "(cuFFT is used on :cuda)"
    end

    def check_even!(length)
      return if length.even?

      raise ArgumentError, "a complex buffer must have an even length (got #{length})"
    end

    def check_divisible!(total, batch, label)
      return if (total % batch).zero?

      raise ArgumentError, "#{label} length #{total} is not divisible by batch #{batch}"
    end
  end
end
