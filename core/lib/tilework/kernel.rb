# frozen_string_literal: true

require "digest"

module Tilework
  # A compiled kernel pipeline.
  #
  # `grid` is the number of thread groups and `block` is the thread-group size,
  # matching CUDA's `<<<grid, block>>>` execution model.
  class Kernel
    @cache = {}
    @cache_mutex = Mutex.new

    class << self
      # Compiles source and reuses an existing pipeline for identical source and name.
      #
      # @api private
      # @param src [String]
      # @param name [String, Symbol]
      # @param raw [Boolean]
      # @return [Kernel]
      def compile(src, name:, raw: false)
        processed = raw ? src : Dialect.expand(src)
        bridge = Tilework.bridge
        key = [bridge.object_id, cache_key(processed, name)]
        pipeline = @cache_mutex.synchronize do
          cache_hit = @cache.key?(key)
          Telemetry.instrument(:compile, label: name, cache_hit: cache_hit) do
            @cache[key] ||= create_pipeline(bridge, processed, name)
          end
        end
        new(pipeline, name, bridge, processed)
      end

      def cache_key(processed_src, name)
        Digest::SHA256.hexdigest("#{processed_src}\0#{name}")
      end

      # @return [Integer] number of cached compiled pipelines
      def cache_size
        @cache_mutex.synchronize { @cache.size }
      end

      # Clears the in-process pipeline cache.
      #
      # @return [Hash] the cleared cache
      def clear_cache!
        @cache_mutex.synchronize { @cache.clear }
      end

      private

      def create_pipeline(bridge, src, name)
        bridge.pipeline_create(src, name)
      rescue RuntimeError => e
        raise CompileError, "failed to compile kernel '#{name}':\n#{e.message}"
      end
    end

    attr_reader :name

    def initialize(pipeline, name, bridge, processed_source)
      @pipeline = pipeline
      @name = name
      @bridge = bridge
      @processed_source = processed_source
    end

    # Launches the kernel and waits for its completion.
    #
    # @param grid [Integer, Array<Integer>] one to three positive grid dimensions
    # @param block [Integer, Array<Integer>] one to three positive thread-group dimensions
    # @param args [Array<Buffer, Numeric>] kernel arguments
    # @return [nil]
    # @raise [LaunchError] if arguments or launch dimensions are invalid
    def launch(grid:, block:, args: [])
      dispatch(grid, block, args, wait: true)
    end

    # Launches the kernel without waiting for completion.
    #
    # @param grid [Integer, Array<Integer>]
    # @param block [Integer, Array<Integer>]
    # @param args [Array<Buffer, Numeric>]
    # @return [nil]
    # @note Read a buffer or call {Tilework.synchronize} before consuming results on the host.
    def launch_async(grid:, block:, args: [])
      dispatch(grid, block, args, wait: false)
    end

    # @return [Integer] backend limit for threads in one thread group
    def max_threads_per_block
      @bridge.pipeline_max_threads(@pipeline)
    end

    private

    def dispatch(grid, block, args, wait:)
      recompile_for_current_bridge! unless Tilework.bridge.equal?(@bridge)

      grid = normalize_dim(grid, "grid")
      block = normalize_dim(block, "block")
      validate_block!(block)
      native_args = convert_args(args)
      bytes = args.sum { |a| a.is_a?(Buffer) ? a.bytesize : 0 }
      threads = grid.zip(block).reduce(1) { |acc, (g, b)| acc * g * b }

      begin
        Telemetry.instrument(:dispatch, label: @name, bytes: bytes, threads: threads) do
          @bridge.dispatch(@pipeline, native_args, grid, block, wait)
        end
      rescue RuntimeError => e
        raise LaunchError, "kernel '#{@name}' failed: #{e.message}"
      end
      nil
    end

    # DSL and built-in kernels are memoized above the pipeline cache. A backend
    # change therefore cannot merely reject the old object: refresh its native
    # pipeline lazily so those long-lived Ruby kernel objects stay portable.
    def recompile_for_current_bridge!
      refreshed = self.class.compile(@processed_source, name: @name, raw: true)
      @pipeline = refreshed.instance_variable_get(:@pipeline)
      @bridge = refreshed.instance_variable_get(:@bridge)
    end

    def normalize_dim(dim, label)
      dims = dim.is_a?(Integer) ? [dim, 1, 1] : Array(dim)
      dims += [1] * (3 - dims.length) if dims.length < 3
      unless dims.length == 3 && dims.all? { |d| d.is_a?(Integer) && d.positive? }
        raise LaunchError, "#{label} must be a positive Integer or an Array of up to 3 positive Integers, got #{dim.inspect}"
      end
      dims
    end

    def validate_block!(block)
      total = block.reduce(:*)
      max = max_threads_per_block
      return if total <= max

      raise LaunchError,
            "block size #{block.inspect} (#{total} threads) exceeds the pipeline limit of #{max} threads per threadgroup"
    end

    # A Buffer becomes setBuffer; a Ruby number becomes a setBytes byte string
    # (fixed Float -> float32 / Integer -> int32 for now).
    def convert_args(args)
      if args.length > MAX_KERNEL_ARGS
        raise LaunchError, "too many kernel arguments: #{args.length} (Metal limit is #{MAX_KERNEL_ARGS})"
      end

      args.map.with_index do |arg, i|
        case arg
        when Buffer  then arg.__handle__
        when Float   then [arg].pack("f")
        when Integer then [arg].pack("l")
        else
          raise LaunchError, "unsupported kernel argument at index #{i}: #{arg.class} (expected Tilework::Buffer, Float, or Integer)"
        end
      end
    end
  end
end
