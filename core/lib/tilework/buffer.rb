# frozen_string_literal: true

module Tilework
  # A typed, device-resident linear memory allocation.
  #
  # Buffers can have a logical shape without changing their row-major storage layout.
  # Use {Buffer.from} to transfer Ruby Arrays and {#to_a} to read values back.
  class Buffer
    # Element size and host<->bytes conversion per dtype.
    #
    # Not a single pack directive because of float16 — Ruby's Array#pack has
    # no half-precision directive (lib/tilework/dtype/half.rb). Holding
    # encode/decode as procs leaves each dtype free to convert however it needs.
    DTYPES = {
      float32: { size: 4, encode: ->(a) { a.pack("f*") }, decode: ->(s) { s.unpack("f*") } },
      int32:   { size: 4, encode: ->(a) { a.pack("l*") }, decode: ->(s) { s.unpack("l*") } },
      # float16 can be allocated only on backends listing it in
      # capabilities[:dtypes]. Accumulation and outputs stay FP32, so it is
      # used only for storage, matmul and offload.
      float16: { size: 2, encode: ->(a) { Half.encode(a) }, decode: ->(s) { Half.decode(s) } },
      bfloat16: { size: 2, encode: ->(a) { BFloat.encode(a) }, decode: ->(s) { BFloat.decode(s) } },
    }.freeze

    # float16 conversions above this many elements go to the GPU. The pure-Ruby
    # Tilework::Half does per-element bit manipulation and takes 1.3 s to encode
    # a million elements (0.5 ms on the GPU path). Rounding is the same, so the
    # results agree bit for bit. Small arrays are not worth a temporary buffer,
    # so they stay in Ruby.
    GPU_CONVERSION_THRESHOLD = 4096

    attr_reader :dtype, :length, :shape

    # Allocates a zero-initialized buffer.
    #
    # @param dtype [:float32, :float16, :bfloat16, :int32] storage dtype supported by the active backend
    # @param length [Integer] positive element count
    # @param shape [Array<Integer>, nil] logical shape; defaults to `[length]`
    # @return [Buffer]
    # @raise [ArgumentError] if the dtype, length, or shape is invalid
    # @raise [DeviceError] if the backend cannot allocate the dtype
    def initialize(dtype, length, shape: nil)
      info = DTYPES[dtype] or
        raise ArgumentError, "unknown dtype: #{dtype.inspect} (available: #{DTYPES.keys.join(', ')})"
      raise ArgumentError, "length must be a positive Integer" unless length.is_a?(Integer) && length.positive?

      check_dtype_supported!(dtype)
      @dtype = dtype
      @length = length
      @shape = validate_shape(shape || [length])
      @encode = info[:encode]
      @decode = info[:decode]
      @elem_size = info[:size]
      @handle = Tilework.bridge.buffer_alloc(bytesize)
      @freed = false
      Scope.register(self)
      Telemetry::Resources.track_buffer(@handle, bytesize)
      # A safety net for a forgotten explicit free (on some backends @handle is
      # not TypedData). A proc capturing self would never be collected, so a
      # class method builds it.
      ObjectSpace.define_finalizer(self, self.class.finalizer(@handle, Tilework.bridge))
    end

    # --- Views (the same memory seen with another shape / range) ---
    #
    # A view **does not allocate**. It is another Buffer pointing at the
    # parent's device memory, and writes reach the parent. Splitting attention
    # heads, KV caches, slicing batches and channel operations would all be
    # copies without it.
    #
    #   x = Tilework.random([4, 256])
    #   x.rows(0...2)          # the first two rows (no copy)
    #   x.reshape(2, 512)      # the same memory with another shape
    #   x.view(256, 256)       # 256 elements starting at element 256
    #
    # **Freeing the parent invalidates its views**; do not let a view outlive
    # its parent. A view's own #free releases no memory; it only makes that
    # view unusable.

    # Returns a zero-copy element-range view.
    #
    # @param offset [Integer] zero-based element offset
    # @param length [Integer] number of elements
    # @param shape [Array<Integer>, nil] logical shape for the view
    # @return [Buffer] a view that aliases this buffer's storage
    # @raise [ArgumentError] if the requested range is outside this buffer
    def view(offset, length, shape: nil)
      require_resident!("view")
      raise ArgumentError, "offset must be a non-negative Integer" unless offset.is_a?(Integer) && !offset.negative?
      raise ArgumentError, "length must be a positive Integer" unless length.is_a?(Integer) && length.positive?

      unless offset + length <= @length
        raise ArgumentError,
              "view #{offset}...#{offset + length} is out of bounds for a buffer of #{@length} elements"
      end

      build_view(offset, length, shape || [length])
    end

    # Returns a zero-copy view with a different logical shape.
    #
    # @param dims [Array<Integer>] dimensions with at most one inferred `-1`
    # @return [Buffer]
    # @raise [ArgumentError] unless the dimensions contain exactly this buffer's element count
    def reshape(*dims)
      dims = dims.first if dims.length == 1 && dims.first.is_a?(Array)
      build_view(0, @length, resolve_shape(dims))
    end

    # Returns a zero-copy slice along the outermost dimension.
    #
    # @param range [Integer, Range] row or outer-dimension range
    # @return [Buffer]
    # @raise [Error] unless this buffer has at least two dimensions
    #
    # A row range for 2D, a batch slice for NCHW.
    #
    #   x.slice(1..2)   # [4, 256] -> [2, 256]
    #   images.slice(0) # [8, 3, 64, 64] -> [1, 3, 64, 64]
    #
    # (Named differently because `#rows` already returns the row count.)
    def slice(range)
      raise Error, "slice requires a shaped buffer (shape: #{@shape.inspect})" if @shape.length < 2

      first, count = normalize_range(range, @shape[0])
      stride = @shape.drop(1).inject(1, :*)
      view(first * stride, count * stride, shape: [count, *@shape.drop(1)])
    end

    # Whether this buffer is a view; if so #base holds the parent.
    def view? = !@base.nil?

    attr_reader :base

    # --- Internal (views) ---

    def build_view(offset, length, shape)
      v = self.class.allocate
      v.send(:init_view, self, offset, length, shape)
      v
    end
    private :build_view

    # Bypasses initialize (which would allocate).
    def init_view(parent, offset, length, shape)
      @dtype = parent.dtype
      @length = length
      @shape = validate_shape(shape)
      info = DTYPES[@dtype]
      @encode = info[:encode]
      @decode = info[:decode]
      @elem_size = info[:size]
      # In a chain of views the parent's handle already carries its offset, so
      # only this view's share is added here (the C side does the addition).
      @handle = Tilework.bridge.buffer_view(parent.__handle__, offset * @elem_size)
      @base = parent.view? ? parent.base : parent # always points at the owning buffer
      @view_offset = offset
      @freed = false
      # Not registered with Telemetry (nothing was allocated, so it would be
      # double-counted), and no finalizer (nothing to free).
      Scope.register(self)
    end
    private :init_view

    def normalize_range(range, limit)
      case range
      when Integer
        idx = range.negative? ? limit + range : range
        raise ArgumentError, "row #{range} is out of bounds (#{limit} rows)" unless (0...limit).cover?(idx)

        [idx, 1]
      when Range
        first = range.begin || 0
        first += limit if first.negative?
        last = range.end || limit
        last += limit if last.negative?
        last -= 1 if range.exclude_end? && range.end
        count = last - first + 1
        unless first >= 0 && count.positive? && first + count <= limit
          raise ArgumentError, "row range #{range} is out of bounds (#{limit} rows)"
        end

        [first, count]
      else
        raise ArgumentError, "expected an Integer or a Range, got #{range.class}"
      end
    end
    private :normalize_range

    def resolve_shape(dims)
      unless dims.all? { |d| d.is_a?(Integer) }
        raise ArgumentError, "shape dimensions must be Integers, got #{dims.inspect}"
      end

      wild = dims.count(-1)
      raise ArgumentError, "at most one dimension may be -1 (got #{dims.inspect})" if wild > 1

      if wild == 1
        known = dims.reject { |d| d == -1 }.inject(1, :*)
        unless known.positive? && (@length % known).zero?
          raise ArgumentError, "cannot infer -1 in #{dims.inspect} for #{@length} elements"
        end

        dims = dims.map { |d| d == -1 ? @length / known : d }
      end

      unless dims.inject(1, :*) == @length
        raise ArgumentError, "shape #{dims.inspect} has #{dims.inject(1, :*)} elements, but the buffer has #{@length}"
      end

      dims
    end
    private :resolve_shape

    # Offloading swaps the handle, which would break a view's link to its parent.
    def reject_view!(action)
      return unless view?

      raise Error, "cannot #{action} a view; call it on the base buffer instead"
    end
    private :reject_view!

    def require_resident!(action)
      return if resident?

      raise Error, "cannot #{action} a buffer that is not resident (offloaded or freed)"
    end
    private :require_resident!

    # bridge.buffer_free is safe to call twice (idempotent) on every backend,
    # so a GC call after an explicit #free is harmless.
    def self.finalizer(handle, bridge)
      proc do
        bridge.buffer_free(handle)
        Telemetry::Resources.untrack_buffer(handle)
      end
    end

    # Transfers a one- or two-dimensional Ruby Array into a buffer.
    #
    # @param array [Array<Numeric>, Array<Array<Numeric>>] values to transfer
    # @param dtype [:float32, :float16, :int32, nil] storage dtype; inferred when omitted
    # @return [Buffer]
    def self.from(array, dtype: nil)
      raise ArgumentError, "array must not be empty" if array.empty?

      if array.first.is_a?(Array)
        unless array.all? { |e| e.is_a?(Array) }
          raise ArgumentError, "array must not mix nested Arrays with scalar elements"
        end

        return from_matrix(array, dtype: dtype)
      end

      dtype ||= array.all?(Integer) ? :int32 : :float32
      buffer = new(dtype, array.length)
      buffer.write(array)
      buffer
    end

    # Transfers a rectangular nested Array into a row-major two-dimensional buffer.
    #
    # @param rows [Array<Array<Numeric>>]
    # @param dtype [:float32, :float16, :int32, nil]
    # @return [Buffer]
    def self.from_matrix(rows, dtype: nil)
      raise ArgumentError, "rows must not be empty" if rows.empty?
      unless rows.all? { |r| r.is_a?(Array) && r.length == rows.first.length }
        raise ArgumentError, "all rows must be non-empty Arrays of the same length"
      end

      flat = rows.flatten(1)
      dtype ||= flat.all?(Integer) ? :int32 : :float32
      buffer = new(dtype, flat.length, shape: [rows.length, rows.first.length])
      buffer.write(flat)
      buffer
    end

    # @return [Boolean] whether this buffer has exactly two dimensions
    def matrix?
      @shape.length == 2
    end

    # @return [Integer] row count for a two-dimensional buffer
    # @raise [Error] unless this buffer is two-dimensional
    def rows
      raise Error, "buffer is not 2D (shape: #{@shape.inspect})" unless matrix?

      @shape[0]
    end

    # @return [Integer] column count for a two-dimensional buffer
    # @raise [Error] unless this buffer is two-dimensional
    def cols
      raise Error, "buffer is not 2D (shape: #{@shape.inspect})" unless matrix?

      @shape[1]
    end

    # Reads a one- or two-dimensional buffer into a Ruby Array.
    #
    # @return [Array<Numeric>, Array<Array<Numeric>>]
    def to_nested_a
      return to_a unless matrix?

      to_a.each_slice(cols).to_a
    end

    # @return [Integer] allocation size in bytes
    def bytesize
      @length * @elem_size
    end

    # Replaces every element with values from a Ruby Array.
    #
    # This synchronizes the active backend before writing.
    #
    # @param array [Array<Numeric>] exactly {#length} values
    # @return [Buffer] this buffer
    def write(array)
      check_freed!
      check_resident!
      unless array.is_a?(Array) && array.length == @length
        raise ArgumentError, "expected an Array of #{@length} elements"
      end
      if gpu_conversion?
        # Send as float32, then convert on the GPU
        staged = Buffer.new(:float32, @length)
        begin
          staged.write(array)
          Cast.convert_into(staged, self)
        ensure
          staged.free
        end
      else
        # Wait for pending async kernels. On unified-memory backends
        # buffer_write is a raw memcpy from the host, so writing while an
        # in-flight kernel reads is a data race.
        Tilework.synchronize
        Telemetry.instrument(:h2d, bytes: bytesize) { Tilework.bridge.buffer_write(@handle, @encode.call(array), 0) }
      end
      self
    end

    # Reads every element into a Ruby Array.
    #
    # This synchronizes the active backend before reading.
    #
    # @return [Array<Numeric>]
    def to_a
      check_freed!
      check_resident!
      Tilework.synchronize
      if gpu_conversion?
        staged = cast(:float32)
        begin
          return staged.to_a
        ensure
          staged.free
        end
      end
      @decode.call(Telemetry.instrument(:d2h, bytes: bytesize) { Tilework.bridge.buffer_read(@handle, bytesize, 0) })
    end

    # Converts this buffer to a different storage dtype.
    #
    # @param target [:float32, :float16, :bfloat16, :int32]
    # @return [Buffer] a converted buffer, or this buffer when the dtype is unchanged
    #
    # The GPU also rounds to nearest even, so the result matches Tilework::Half
    # bit for bit.
    def cast(target)
      check_freed!
      check_resident!
      return self if target == @dtype

      Tilework::Cast.convert(self, target)
    end

    # Reads raw bytes without packing or unpacking values.
    #
    # @param length [Integer, nil] byte count; defaults to the remaining allocation
    # @param offset [Integer] byte offset
    # @return [String]
    #
    # Unlike `to_a` there is no pack/unpack, so **this works directly for
    # reading and writing weight files**. "Header + raw tensors" formats like
    # safetensors can be implemented in the application with this alone (the
    # library need not know the format).
    #
    #   w = Tilework::Buffer.new(:float32, 1024)
    #   File.open(path) { |f| f.seek(data_offset); w.write_bytes(f.read(w.bytesize)) }
    #
    # offset and length are **bytes, not elements**, for streaming a large file
    # in chunks.
    #
    # The bytes are the backend's native representation (float32 is
    # little-endian IEEE-754 single, float16 is binary16). x86 and ARM are both
    # little-endian, so there is no mismatch with file formats to worry about.
    def read_bytes(length = nil, offset: 0)
      check_freed!
      check_resident!
      length ||= bytesize - offset
      check_byte_range!(offset, length)
      Tilework.synchronize
      Telemetry.instrument(:d2h, bytes: length) { Tilework.bridge.buffer_read(@handle, length, offset) }
    end

    # Writes raw bytes without converting values.
    #
    # @param bytes [String] byte sequence
    # @param offset [Integer] byte offset
    # @return [Buffer] this buffer
    def write_bytes(bytes, offset: 0)
      check_freed!
      raise ArgumentError, "expected a String of bytes, got #{bytes.class}" unless bytes.is_a?(String)

      check_byte_range!(offset, bytes.bytesize)
      # Same as #write: host writes race in-flight kernels.
      Tilework.synchronize
      Telemetry.instrument(:h2d, bytes: bytes.bytesize) { Tilework.bridge.buffer_write(@handle, bytes, offset) }
      self
    end

    # Bytes per element, for read_bytes / write_bytes offset arithmetic.
    attr_reader :elem_size

    def check_byte_range!(offset, length)
      unless offset.is_a?(Integer) && !offset.negative? && length.is_a?(Integer) && !length.negative?
        raise ArgumentError, "offset and length must be non-negative Integers"
      end
      return if offset + length <= bytesize

      raise ArgumentError,
            "byte range #{offset}...#{offset + length} is out of bounds for #{bytesize} bytes"
    end
    private :check_byte_range!

    # --- VRAM <-> SSD offload ---

    # Saves this buffer to disk and releases its device allocation.
    #
    # @param path [String] destination file
    # @param checksum [Boolean] include an integrity checksum
    # @return [Buffer] this non-resident buffer
    def offload!(path, checksum: false)
      reject_view!("offload!")
      check_freed!
      return self unless resident?

      backing = Offload::FileBacking.new(path, checksum: checksum).write(self)
      Tilework.bridge.buffer_free(@handle)
      Telemetry::Resources.untrack_buffer(@handle)
      @handle = nil
      @backing = backing
      @last_backing = backing
      self
    end

    # Moves this buffer into an explicitly allocated volatile CPU-RAM backing.
    # Unlike #offload!, no filesystem path is interpreted or created.
    def offload_to_host!
      reject_view!("offload_to_host!")
      check_freed!
      return self unless resident?
      unless Offload::HostBacking.available?
        raise Error, "pageable host backing is not available on the #{Tilework.backend.inspect} backend"
      end

      backing = Offload::HostBacking.new(dtype: @dtype, length: @length, shape: @shape, bytesize: bytesize)
      begin
        backing.write(self)
      rescue StandardError
        backing.delete
        raise
      end
      Tilework.bridge.buffer_free(@handle)
      Telemetry::Resources.untrack_buffer(@handle)
      @handle = nil
      @backing = backing
      @last_backing = backing
      self
    end

    # Restores an offloaded buffer synchronously.
    #
    # @return [Buffer] this resident buffer
    def restore!
      reject_view!("restore!")
      check_freed!
      return self if resident?
      reject_pending_restore!

      backing = backing!
      meta = backing.validate!(self)

      @handle = Tilework.bridge.buffer_alloc(bytesize)
      Telemetry::Resources.track_buffer(@handle, bytesize)
      ObjectSpace.define_finalizer(self, self.class.finalizer(@handle, Tilework.bridge))
      @backing = nil
      backing.restore_into(self, meta)
      self
    end

    # Starts restoring an offloaded buffer asynchronously.
    #
    # @return [Offload::Transfer] call `#wait` before using the buffer
    #
    # The contents are complete only after Transfer#wait. Until then resident?
    # stays false, so using it without waiting raises NotResidentError.
    def restore_async!(async: true)
      reject_view!("restore_async!")
      check_freed!
      return Offload::Transfer.new(nil, self) if resident?
      reject_pending_restore!
      return Offload::Transfer.new(nil, restore!) unless async

      backing = backing!
      backing.validate!(self)

      handle = Tilework.bridge.buffer_alloc(bytesize)
      token = backing.begin_restore(handle, bytesize)
      # Backings that cannot load asynchronously (such as a checksummed
      # FileBacking) fall back to a synchronous restore. Reallocating without
      # first releasing the handle already allocated would leak, so free it here.
      if token.nil?
        Tilework.bridge.buffer_free(handle)
        return Offload::Transfer.new(nil, restore!)
      end
      kind, token = token if token.is_a?(Array)
      @pending_handle = handle
      @pending_backing = backing
      Offload::Transfer.new(token, self,
                            poll: kind == :host ? :host_backing_transfer_poll : :transfer_poll,
                            finish: kind == :host ? :host_backing_transfer_end : :transfer_end)
    end

    # Called from Transfer#wait; this is where it becomes resident.
    def __finish_restore__
      return self unless @pending_handle

      @handle = @pending_handle
      Telemetry::Resources.track_buffer(@handle, bytesize)
      ObjectSpace.define_finalizer(self, self.class.finalizer(@handle, Tilework.bridge))
      @pending_handle = nil
      @backing = nil
      @pending_backing = nil
      self
    end

    # Called after pipeline leaves the block; writes back to the latest offload target.
    def offload_back!
      reject_view!("offload_back!")
      return self if @freed || !resident?

      backing = backing_for_reoffload
      backing.write(self)
      Tilework.bridge.buffer_free(@handle)
      Telemetry::Resources.untrack_buffer(@handle)
      @handle = nil
      @backing = backing
      self
    end

    # Releases device memory without rewriting the existing offload file.
    #
    # @return [Buffer] this non-resident buffer
    # @note Changes since restoration are discarded.
    #
    # Assumes the offload file exists and is unchanged — when streaming
    # read-only weights, writing back only rewrites the same contents, and as
    # synchronous I/O it cannot overlap compute and defeats hiding.
    #
    # **Changes made in the block are lost.** To keep them use offload_back!
    # (write_back: true for pipeline).
    def evict!
      reject_view!("evict!")
      return self if @freed || !resident?

      backing = backing_for_reoffload
      Tilework.bridge.buffer_free(@handle)
      Telemetry::Resources.untrack_buffer(@handle)
      @handle = nil
      @backing = backing
      self
    end

    # @return [Boolean] whether this buffer can be used by device operations
    def resident?
      !@freed && !@handle.nil?
    end

    # Temporarily restores offloaded buffers for a block.
    #
    # @param buffers [Array<Buffer>]
    # @param write_back [Boolean] re-offload modified buffers when the block exits
    # @yieldparam buffers [Buffer]
    # @return [Object] the block result
    #
    #   Tilework::Buffer.with_resident(w1, w2, w3) { |a, b, c| a.matmul(b).matmul(c) }
    #
    # Restores are **issued in parallel and waited on once**. Nesting
    # with_resident one buffer at a time serializes the transfers and cannot
    # fill NVMe's queue depth (measured: 1.48x at 4, 1.69x at 8).
    #
    # write_back: defaults to true so changes made in the block are not lost.
    # When the data is known to be read-only, false skips rewriting the same
    # bytes entirely.
    #
    # Buffers that were already resident are left alone (neither loaded nor
    # offloaded).
    def self.with_resident(*buffers, write_back: true)
      targets = buffers.reject { |b| b.freed? || b.resident? }
      Offload.wait_all(targets.map(&:restore_async!))

      yield(*buffers)
    ensure
      targets&.each do |b|
        next if b.freed?

        write_back ? b.offload_back! : b.evict!
      end
    end

    # Temporarily restores this buffer for a block.
    #
    # @param write_back [Boolean] re-offload this buffer when the block exits
    # @yieldparam buffer [Buffer]
    # @return [Object] the block result
    #
    # restore! on entry, offload on exit: the explicit sugar in place of
    # restoring offloaded buffers implicitly.
    def with_resident(write_back: true, &block)
      self.class.with_resident(self, write_back: write_back, &block)
    end

    # Releases this buffer's device allocation and any owned offload file.
    #
    # Inside a {Tilework.submit} block the release is **deferred** to the end of
    # the block: launched kernels may still be reading this memory, and handing
    # it back before they finish is undefined behavior. The buffer therefore
    # stays usable — and `#freed?` stays false — until the job completes.
    #
    # @return [nil]
    #
    # An offloaded buffer's file is deleted too (so a Scope ending releases VRAM
    # and deletes offload files alike). Persistence through Offload::Store is
    # untouched, since the Store owns those paths.
    def free
      return if @freed
      return if Ops.defer_free(self)

      reject_pending_restore!

      if @handle
        Tilework.bridge.buffer_free(@handle) # a view releases no real memory (decided on the C side)
        Telemetry::Resources.untrack_buffer(@handle) unless view?
      elsif @backing
        @backing.delete
      end
      # A HostBacking is volatile and the Buffer is its only owner. After
      # restore! @backing is nil but @last_backing remains as the re-offload
      # target, so it must be reclaimed here. A FileBacking keeps the existing
      # Store/path ownership.
      if @last_backing.is_a?(Offload::HostBacking)
        @last_backing.delete
        @last_backing = nil
      end
      @handle = nil
      @freed = true
      nil
    end

    # Graph residency uses this after restoring a plan-owned volatile backing.
    # Durable FileBacking remains owned by its path/store and is intentionally
    # untouched.
    def discard_host_backing!
      return self unless @last_backing.is_a?(Offload::HostBacking)

      @last_backing.delete
      @last_backing = nil
      self
    end

    def freed?
      @freed
    end

    # The native handle passed to the bridge layer; not public API.
    def __handle__
      check_freed!
      check_resident!
      @handle
    end

    private

    # Large float16 conversions are orders of magnitude faster on the GPU (see the constant's comment).
    def gpu_conversion?
      %i[float16 bfloat16].include?(@dtype) && @length >= GPU_CONVERSION_THRESHOLD &&
        Cast.supported?(:float32, @dtype)
    end

    def check_dtype_supported!(dtype)
      supported = Tilework.capabilities[:dtypes]
      return if supported.include?(dtype)

      raise DeviceError,
            "dtype #{dtype.inspect} is not supported on the #{Tilework.backend.inspect} backend " \
            "(supported: #{supported.map(&:inspect).join(' / ')})"
    end

    # Where with_resident's exit writes back. restore! sets @backing to nil, so
    # the latest backing is remembered separately.
    def backing_for_reoffload
      @last_backing or
        raise Error, "this buffer has never been offloaded, so there is nowhere to put it back; " \
                     "call #offload!(path) first (Tilework::Offload.pipeline expects an already-offloaded series)"
    end

    def backing!
      @backing or raise Error, "buffer has no backing to restore"
    end

    # The pending GPU allocation and HostBacking are owned by Transfer until
    # #wait joins it. Starting another restore or freeing them beforehand
    # would race the same backing store.
    def reject_pending_restore!
      return unless @pending_handle

      raise Error, "buffer restore is still in progress; call Transfer#wait first"
    end

    def check_resident!
      return if resident?

      raise NotResidentError,
            "buffer is offloaded to #{backing_for_reoffload.diagnostic}; call #restore! " \
            "(or wrap the use in #with_resident) before operating on it"
    end

    def validate_shape(shape)
      unless shape.is_a?(Array) && shape.all? { |d| d.is_a?(Integer) && d.positive? } && shape.reduce(:*) == @length
        raise ArgumentError, "shape #{shape.inspect} does not match length #{@length}"
      end

      shape
    end

    def check_freed!
      raise Error, "buffer has been freed" if @freed
    end
  end
end
