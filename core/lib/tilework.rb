# frozen_string_literal: true

require_relative "tilework/version"
require_relative "tilework/runtime/telemetry"
require_relative "tilework/dtype/half"
require_relative "tilework/dtype/bfloat"
require_relative "tilework/dtype/cast"
require_relative "tilework/math/philox"
require_relative "tilework/math/random"
require_relative "tilework/math/fft"
require_relative "tilework/math/conv"
require_relative "tilework/capabilities"
require_relative "tilework/dialect"
require_relative "tilework/scope"
require_relative "tilework/buffer"
require_relative "tilework/kernel"
require_relative "tilework/dsl"
require_relative "tilework/ops"
require_relative "tilework/math/tensor_core"
require_relative "tilework/runtime/submission"
require_relative "tilework/runtime/offload"
require_relative "tilework/math/nn"
require_relative "tilework/graph"

module Tilework
  class Error < StandardError; end
  class DeviceError < Error; end
  class CompileError < Error; end
  class LaunchError < Error; end
  # Raised when an operation requires a buffer that has been offloaded.
  class NotResidentError < Error; end

  # The maximum number of kernel arguments supported by the Metal backend.
  MAX_KERNEL_ARGS = 31

  class << self
    attr_reader :backend

    # Registers a backend bridge.
    #
    # @api private
    # @param name [Symbol, String] backend identifier
    # @param bridge [Module] bridge implementing Tilework's backend ABI
    # @return [Module] the registered bridge
    def register_backend(name, bridge)
      (@registry ||= {})[name.to_sym] = bridge
    end

    # Selects a backend and loads its gem on demand.
    #
    # @param name [Symbol, String] `:cpu`, `:metal`, `:cuda`, or a registered backend
    # @return [Symbol] the selected backend
    # @raise [DeviceError] if the backend cannot be loaded or registered
    def backend=(name)
      name = name.to_sym
      unless (@registry ||= {}).key?(name)
        begin
          require "tilework/backend/#{name}"
        rescue LoadError => e
          raise DeviceError, "backend #{name.inspect} is not installed; add gem \"tilework-#{name}\" " \
                              "to your Gemfile (#{e.message})"
        end
      end
      next_bridge = @registry[name] or
        raise DeviceError, "tilework/backend/#{name} loaded but never called Tilework.register_backend(#{name.inspect}, ...)"
      release_backend_cache! unless @bridge.equal?(next_bridge)
      @bridge = next_bridge
      @backend = name
    end

    # Returns the active backend bridge.
    #
    # @return [Module]
    # @raise [DeviceError] if no backend has been selected
    def bridge
      @bridge or raise DeviceError, "no backend selected; set Tilework.backend = :metal first"
    end

    # Installs a custom backend bridge.
    #
    # @api private
    # @param mod [Module] a bridge implementing Tilework's backend ABI
    # @return [Module]
    def bridge=(mod)
      release_backend_cache! unless @bridge.equal?(mod)
      @bridge = mod
      @backend = :custom
    end

    # Whether the active bridge prefers direct host execution for supported operations.
    #
    # @api private
    # @return [Boolean]
    def host_fallback?
      bridge.respond_to?(:host_fallback?) && bridge.host_fallback?
    end

    # Returns the active device's display name.
    #
    # @return [String]
    def device_name
      bridge.device_name
    end

    # Describes the active backend's supported features and device properties.
    # Upper layers pick paths from this one value, keeping the branching in one
    # place. Do not probe Tilework.bridge.respond_to? individually — this is
    # more reliable than aligning respond_to? behavior across backends, and it
    # doubles as a user-facing report.
    #
    #   Tilework.capabilities
    #   # => { tensor_core: true, tf32: true, offload: true, offload_async: false,
    #   #      offload_transport: :posix, gemm_library: nil, warp_size: 32,
    #   #      unified_memory: true, memory_budget_bytes: 7_400_000_000,
    #   #      compute_capability: [8, 7] }
    #
    # Unimplemented features honestly answer false / nil; branches rely on that.
    # Capabilities::FIELDS in lib/tilework/capabilities.rb is the single source
    # of the key set, types, defaults and meanings; this only writes the
    # backend's declarations on top of it.
    def capabilities
      base = Capabilities.build(
        offload_async: Offload.async?, # has transfer_begin/poll/end
        offload_transport: Offload.transport,
        host_backing: Offload::HostBacking.available?,
        # HIP's first implementation is a pageable synchronous copy; it does not
        # declare async until the staging ABI is measured.
        host_backing_async: backend && bridge.respond_to?(:host_backing_async_available?) && bridge.host_backing_async_available?,
      )

      case backend
      when :cuda
        base.merge(cuda_capabilities.merge(delegated_library_capabilities,
                                           bf16_storage: true, dtypes: base[:dtypes] + %i[float16 bfloat16]))
      when :metal
        # SIMD group matrix is unsupported, so tensor_core is false. float16
        # storage is verified on Apple Silicon by conformance / metal_test.
        base.merge(delegated_library_capabilities,
                   warp_size: 32, unified_memory: true, bf16_storage: true,
                   dtypes: base[:dtypes] + %i[float16 bfloat16])
      when :hip
        base.merge(hip_capabilities.merge(delegated_library_capabilities,
                                          bf16_storage: true, dtypes: base[:dtypes] + %i[float16 bfloat16]))
      when :cpu
        # :cpu's generators (lib/tilework/backend/cpu/*) accept neither
        # CUDA-flavored dialect strings nor float16 storage (they only emit
        # plain Ruby source).
        base.merge(warp_size: 1, unified_memory: true, dialect: false, fp16_storage: false,
                   bf16_storage: true, dtypes: base[:dtypes] + [:bfloat16])
      else
        base
      end
    end

    # Creates a buffer from scalar values or an Array.
    #
    # @param values [Array<Numeric>, Array<Array<Numeric>>] values to transfer
    # @param dtype [:float32, :float16, :int32, nil] storage dtype; inferred when omitted
    # @return [Buffer]
    def [](*values, dtype: nil)
      array = values.length == 1 && values.first.is_a?(Array) ? values.first : values
      Buffer.from(array, dtype: dtype)
    end

    # Compiles source into a reusable kernel.
    #
    # @param src [String] Tilework dialect source, or native source when `raw: true`
    # @param name [String, Symbol] kernel entry-point name
    # @param raw [Boolean] bypass dialect expansion for backend-native source
    # @return [Kernel]
    # @raise [CompileError] if the active backend cannot compile the source
    def compile(src, name:, raw: false)
      if !capabilities[:dialect] && !raw
        raise CompileError,
              "the #{backend.inspect} backend does not support dialect kernel strings; " \
              "use Tilework.kernel { } (DSL), Buffer operators (Ops), or pass raw: true with backend-native source"
      end
      Kernel.compile(src, name: name, raw: raw)
    end

    # Defines a Ruby-block kernel specialized from its call arguments.
    #
    # @param name [String] generated kernel name
    # @param out [Integer, nil] output argument index; inferred when omitted
    # @param out_dtype [Symbol, nil] output dtype; inferred when omitted
    # @yield kernel body in the supported Tilework DSL
    # @return [DSLKernel]
    def kernel(name = "dsl_kernel", out: nil, out_dtype: nil, &block)
      DSLKernel.new(name, block, out: out, out_dtype: out_dtype)
    end

    # Runs the block with kernel launches enqueued instead of waited on, and
    # returns a {Submission::Job} whose `#wait` synchronizes once.
    #
    #   job = Tilework.submit { (a * 2 + b).relu }
    #   out = job.wait
    #
    # Ordering on the device is unchanged (everything goes to the same queue),
    # so the values match a synchronous eager run. Only the host-side waiting
    # moves: one wait at the boundary instead of one per launch. Operations
    # that must wait anyway — whole-buffer reductions, attention, delegated
    # library calls, and reading back to the host — stay synchronous inside the
    # block; see {Submission}.
    #
    # `Buffer#free` called inside the block is deferred until the job
    # completes, so a temporary buffer freed in an `ensure` is not handed back
    # while a launched kernel still reads it. Nesting is a no-op: an inner
    # `submit` does not introduce a second boundary.
    #
    # @yield code to enqueue
    # @return [Submission::Job]
    def submit(&block)
      Submission.run(&block)
    end

    # Waits for all work submitted to the active backend.
    #
    # @return [nil]
    # @raise [LaunchError] if work submitted asynchronously failed; the call
    #   that enqueued it had already returned, so this is where it surfaces
    def synchronize
      Telemetry.instrument(:synchronize) { bridge.synchronize }
      nil
    rescue RuntimeError => e
      raise LaunchError, "asynchronous work failed: #{e.message}"
    end

    # Collects backend activity while the block runs.
    #
    # @yield code to profile
    # @return [Telemetry::Report]
    # @raise [Error] if called while another profile is active
    def profile
      Telemetry.start! # raises here when nested (finish! is not called yet)
      begin
        yield
      ensure
        events = Telemetry.finish!
      end
      Telemetry::Report.new(events)
    end

    # Runs a block with Array inputs converted to buffers and outputs converted back.
    #
    # Buffers allocated inside the block are freed when it returns, including on errors.
    # Do not retain a buffer except through the block return value.
    #
    # @param inputs [Array, Buffer] input values
    # @yieldparam buffers [Buffer] converted inputs
    # @return [Object] the block result, with buffers converted to nested Arrays
    def run(*inputs)
      scope = Scope.open
      args = inputs.map { |v| v.is_a?(Array) ? Buffer.from(v) : v }
      unwrap(yield(*args))
    ensure
      Scope.close(scope)
    end

    private

    # Compiled pipelines belong to the bridge that created them. Keeping the
    # cache across a backend change both retains the old backend's native
    # modules and risks handing one backend another backend's handle for an
    # identical source string. Synchronize before dropping the cache because
    # an async launch may still be using a cached pipeline.
    def release_backend_cache!
      return unless @bridge && defined?(Kernel) && Kernel.cache_size.positive?

      Telemetry.instrument(:synchronize) { @bridge.synchronize }
      Kernel.clear_cache!
    end

    # Takes raw device facts from the bridge's optional query and translates
    # them to contract keys. Without the query, only warp_size is filled with
    # a known constant (CUDA's warpSize is always 32).
    def cuda_capabilities
      probe = bridge.respond_to?(:device_capabilities) ? bridge.device_capabilities : nil
      return { warp_size: 32 } if probe.nil?

      major, = probe[:compute_capability]
      # The bridge returns raw device facts without knowing the contract's key
      # set; keys outside it are dropped here (Capabilities.build would reject them).
      Capabilities.slice_known(probe).merge(
        # Tensor Cores from Volta (SM 70). TF32 from Ampere (SM 80), taking FP32
        # buffers directly as input.
        tensor_core: major >= 7,
        tf32: major >= 8,
        # FP16 Tensor Cores from Volta (SM 70).
        fp16: major >= 7,
      )
    end

    # HIP's gfx names are not CUDA compute capabilities. Only measured values the
    # bridge returns go into the contract; matrix instructions / TF32 are not
    # declared until an execution probe exists.
    def hip_capabilities
      probe = bridge.respond_to?(:device_capabilities) ? bridge.device_capabilities : nil
      matrix_core = bridge.respond_to?(:matrix_core_available?) && bridge.matrix_core_available?
      return { compute_capability: nil, tf32: false, tensor_core: matrix_core, fp16: matrix_core } if probe.nil?

      Capabilities.slice_known(probe).merge(
        compute_capability: nil,
        tf32: false,
        tensor_core: matrix_core,
        fp16: matrix_core,
      )
    end

    # Delegation (GEMM/RNG/FFT/convolution) is not CUDA-specific like SM
    # generations; it follows from respond_to? queries to the bridge alone, so
    # the same expression is right on :cuda and :metal alike.
    def delegated_library_capabilities
      {
        gemm_library: TensorCore.gemm_library? ? TensorCore.gemm_library_name : nil,
        gemm_batched: TensorCore.gemm_batched?,
        gemm_dtypes: if !TensorCore.gemm_library?
                       []
                     elsif bridge.respond_to?(:gemm_supported_dtypes)
                       bridge.gemm_supported_dtypes
                     else
                       %i[float32 float16]
                     end,
        rng: Random.library,
        fft: FFT.library,
        conv: Conv.library,
        conv_backward: Conv.backward_available?,
        conv_dtypes: if !Conv.library
                       []
                     elsif bridge.respond_to?(:conv_supported_dtypes)
                       bridge.conv_supported_dtypes
                     else
                       %i[float32 float16]
                     end,
      }
    end

    def unwrap(value)
      case value
      when Buffer then value.to_nested_a # same as to_a for a 1D shape
      when Array  then value.map { |v| unwrap(v) }
      else value
      end
    end
  end
end
