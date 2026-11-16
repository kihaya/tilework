# frozen_string_literal: true

module Tilework
  # Random number generation for Tilework buffers.
  #
  # CUDA delegates to cuRAND (Philox, counter-based). Other backends generate
  # with Ruby's Random and transfer — slower, but the result has the same
  # shape, so examples and tests work on any backend (the same policy as the
  # optional acceleration ABI).
  #
  # Generating on the host costs hundreds of ms of Ruby plus the transfer for a
  # million elements. Generating on the GPU stays on the device, which changes
  # the order of magnitude for large uses like weight initialization or Monte
  # Carlo.
  #
  # ## Reproducibility
  #
  # The default (`mode == :native`) guarantees reproducibility **within one
  # process and one backend** only. cuRAND, MPSMatrixRandomPhilox and Ruby's
  # Random return different sequences for the same seed (counter/key layout
  # and the normal transform are implementation-specific, so sharing the name
  # "Philox" does not make them agree).
  #
  # For the same sequence across backends, switch to the portable path:
  #
  #   Tilework::Random.portable!          # same sequence on :cpu / :metal / :cuda
  #   Tilework::Random.seed!(42)
  #   Tilework.randn([64, 64])
  #
  # The portable path uses Tilework::Philox (a dialect kernel plus a Ruby
  # implementation of the same formula). `random` is bit-identical and `randn`
  # agrees within Box-Muller's transcendental precision (about 1e-6 relative).
  # It can be slower than the native RNG, so it is not the default — the same
  # idea as the optional ABI: use the fast path when there is one, align only
  # when needed.
  module Random
    UNIFORM = 0
    NORMAL = 1
    BLOCK = 256
    MODES = %i[native portable].freeze

    class << self
      # @return [:native, :portable] current generation mode
      def mode = @mode || :native

      # Selects native or cross-backend portable generation.
      #
      # @param value [:native, :portable]
      # @return [:native, :portable]
      # @raise [ArgumentError] for an unsupported mode
      def mode=(value)
        value = value.to_sym
        raise ArgumentError, "unknown random mode #{value.inspect} (expected #{MODES.join(' or ')})" unless MODES.include?(value)

        @mode = value
      end

      # Enables deterministic portable Philox generation.
      # @return [:portable]
      def portable! = self.mode = :portable
      # Enables the fastest backend-native generator when available.
      # @return [:native]
      def native! = self.mode = :native
      # @return [Boolean] whether portable generation is enabled
      def portable? = mode == :portable

      # Without an explicit seed each call returns a different sequence;
      # seed: makes it deterministic (same seed, same sequence).
      attr_reader :default_seed

      # Sets the process-wide seed and resets the draw counter.
      #
      # @param value [Integer, nil] seed; `nil` restores nondeterministic defaults
      # @return [Integer, nil] the configured seed
      #
      # Makes the whole process deterministic. It resets **the call counter as
      # well as the seed** — a leftover counter would continue the sequence
      # even after setting the same default_seed, which is not reproduction.
      #
      #   Tilework::Random.seed!(42)
      #   a = Tilework.random(100).to_a
      #   Tilework::Random.seed!(42)
      #   a == Tilework.random(100).to_a   # => true
      def seed!(value)
        @default_seed = value
        @counter = 0
        value
      end

      def native?
        return false unless Tilework.backend
        return false unless Tilework.bridge.respond_to?(:random_available?)

        Tilework.bridge.random_available?
      end

      def library
        return nil unless native?
        return Tilework.bridge.random_library_name if Tilework.bridge.respond_to?(:random_library_name)

        :curand
      end

      def next_seed
        @counter = (@counter || 0) + 1
        (default_seed || ::Random.new_seed) ^ (@counter * 0x9E3779B97F4A7C15)
      end
    end

    module_function

    # Fills a float32 buffer with uniformly distributed values in `(0, 1]`.
    #
    # @param buffer [Buffer]
    # @param seed [Integer, nil]
    # @return [Buffer] the supplied buffer
    #
    # That is how cuRAND's curandGenerateUniform is defined, so the host
    # fallback matches it (Ruby's rand is [0, 1), the opposite ends).
    def uniform!(buffer, seed: nil)
      fill(buffer, UNIFORM, seed, 0.0, 1.0)
    end

    # Fills a float32 buffer with normally distributed values.
    #
    # @param buffer [Buffer]
    # @param mean [Float]
    # @param stddev [Float]
    # @param seed [Integer, nil]
    # @return [Buffer] the supplied buffer
    def normal!(buffer, mean: 0.0, stddev: 1.0, seed: nil)
      fill(buffer, NORMAL, seed, mean, stddev)
    end

    def fill(buffer, kind, seed, mean, stddev)
      unless buffer.dtype == :float32
        raise Error, "random generation only supports :float32 buffers (got #{buffer.dtype.inspect})"
      end

      seed ||= Random.next_seed
      if Random.portable?
        portable_fill(buffer, kind, seed, mean, stddev)
      elsif Random.native?
        Tilework.bridge.random(buffer.__handle__, buffer.length, kind, seed & 0xFFFFFFFFFFFFFFFF,
                          mean, stddev)
      else
        buffer.write(host_values(buffer.length, kind, seed, mean, stddev))
      end
      buffer
    end

    # The portable path. Backends that accept the dialect run it on the device;
    # others (:cpu) fill it with a Ruby implementation of **the same formula**.
    # Both give the same sequence (see Philox's comment).
    def portable_fill(buffer, kind, seed, mean, stddev)
      key_lo, key_hi = Philox.key_parts(seed)
      unless Tilework.capabilities[:dialect]
        return buffer.write(Philox.host_values(buffer.length, kind, key_lo, key_hi, mean, stddev))
      end

      kernel = Tilework.compile(Philox.kernel_source(kind), name: "philox_fill")
      Ops.launch_kernel(kernel, grid: (buffer.length / BLOCK.to_f).ceil, block: BLOCK,
                            args: [buffer, Philox.signed32(key_lo), Philox.signed32(key_hi),
                                   mean.to_f, stddev.to_f, buffer.length])
      buffer
    end

    # Fallback for backends without a native path. It does not match cuRAND (a
    # different algorithm altogether) but stays reproducible for the same seed.
    def host_values(length, kind, seed, mean, stddev)
      rng = ::Random.new(seed & 0x7FFFFFFF)
      if kind == UNIFORM
        # (0, 1] to match cuRAND; Ruby's rand is [0, 1)
        Array.new(length) { 1.0 - rng.rand }
      else
        # Box-Muller, the same idea as cuRAND's normal generator
        Array.new(length) do
          u1 = 1.0 - rng.rand # avoid log(0)
          u2 = rng.rand
          mean + stddev * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2 * Math::PI * u2)
        end
      end
    end
  end

  class << self
    # Allocates a float32 buffer of uniformly distributed values in `(0, 1]`.
    #
    # @param shape [Integer, Array<Integer>] positive dimensions
    # @param seed [Integer, nil]
    # @return [Buffer]
    #
    #   Tilework.random(1000)          # => a Buffer of length 1000
    #   Tilework.random([256, 128])    # => a buffer with a 2D shape
    #   Tilework.random(1000, seed: 42) # deterministic
    def random(shape, seed: nil)
      fill_new_buffer(shape) { |buffer| Random.uniform!(buffer, seed: seed) }
    end

    # Allocates a float32 buffer of normally distributed values.
    #
    # @param shape [Integer, Array<Integer>] positive dimensions
    # @param mean [Float]
    # @param stddev [Float]
    # @param seed [Integer, nil]
    # @return [Buffer]
    #
    #   Tilework.randn([out, inp]) * Math.sqrt(2.0 / inp)   # He initialization
    def randn(shape, mean: 0.0, stddev: 1.0, seed: nil)
      fill_new_buffer(shape) { |buffer| Random.normal!(buffer, mean: mean, stddev: stddev, seed: seed) }
    end

    private

    def fill_new_buffer(shape)
      buffer = new_random_buffer(shape)
      yield buffer
    rescue StandardError
      buffer&.free
      raise
    end

    def new_random_buffer(shape)
      dims = shape.is_a?(Array) ? shape : [shape]
      unless dims.all? { |d| d.is_a?(Integer) && d.positive? }
        raise ArgumentError, "shape must be a positive Integer or an Array of them, got #{shape.inspect}"
      end

      Buffer.new(:float32, dims.reduce(:*), shape: dims)
    end
  end
end
