# frozen_string_literal: true

require "fileutils"

module Tilework
  # Explicit persistence and restoration of device buffers.
  #
  # The purpose is moving what does not fit in VRAM out to SSD, not backup, so
  # offload! always releases VRAM.
  #
  # Computing on an offloaded Buffer raises instead of restoring implicitly:
  # swap-like implicit restores would make performance unpredictable — the
  # same thinking that made to_a an explicit transfer. Buffer#with_resident is
  # the explicit sugar.
  module Offload
    MAGIC = "RMGPUOFF"
    VERSION = 1
    # Header and body both sit on 4096-byte boundaries to suit O_DIRECT. The
    # Ruby fallback writes the same format, so swapping in the C ABI
    # implementation later keeps files compatible.
    HEADER_BYTES = 4096
    ALIGN = 4096

    DTYPE_CODES = { float32: 0, int32: 1, float16: 2, bfloat16: 3 }.freeze
    CODE_DTYPES = DTYPE_CODES.invert.freeze

    MAX_DIMS = 8

    module_function

    # Non-resident Buffer payloads.  The public `Buffer#offload!(path)` API
    # remains path-shaped, but Buffer itself owns a backing object so native
    # host memory can use the identical residency transitions without being
    # disguised as a filesystem path.
    class Backing
      def kind = :backing

      def diagnostic = kind.to_s

      def delete; end
    end

    class FileBacking < Backing
      attr_reader :path, :checksum

      def initialize(path, checksum: false)
        @path = String(path)
        @checksum = checksum
      end

      def kind = :file

      def diagnostic = "file backing at #{@path.inspect}"

      def write(buffer)
        Offload.write_file(@path, buffer, checksum: @checksum)
        self
      end

      def metadata
        Offload.read_header(@path)
      end

      def validate!(buffer)
        meta = metadata
        unless meta[:dtype] == buffer.dtype && meta[:length] == buffer.length
          raise Error, "offload file #{@path} does not match this buffer " \
                       "(#{meta[:dtype]} x#{meta[:length]} vs #{buffer.dtype} x#{buffer.length})"
        end
        meta
      end

      def restore_into(buffer, meta = validate!(buffer))
        Offload.read_body_into(@path, buffer, meta)
      end

      # Returns a bridge transfer token or nil when the durable file requires
      # the synchronous/checksummed path.
      def begin_restore(handle, bytes, meta = validate_metadata)
        return nil unless Offload.async? && meta[:checksum].nil?

        Tilework.bridge.transfer_begin(handle, @path, bytes, 0, 1)
      end

      def delete
        FileUtils.rm_f(@path)
      end

      private

      def validate_metadata = metadata
    end

    # Volatile pageable host storage.  Its opaque native handle is intentionally
    # never represented as a path or exposed in diagnostics.  Only backends
    # implementing the optional host-backing bridge ABI can construct it.
    class HostBacking < Backing
      Ownership = Struct.new(:handle, :bridge, :freed)

      attr_reader :dtype, :length, :shape

      def self.available?
        Tilework.bridge.respond_to?(:host_backing_available?) && Tilework.bridge.host_backing_available?
      end

      def initialize(dtype:, length:, shape:, bytesize:)
        @dtype, @length, @shape = dtype, length, shape.dup
        bridge = Tilework.bridge
        @handle = bridge.host_backing_alloc(bytesize)
        @ownership = Ownership.new(@handle, bridge, false)
        ObjectSpace.define_finalizer(self, self.class.finalizer(@ownership))
      end

      def self.finalizer(ownership)
        proc do
          next if ownership.freed

          ownership.bridge.host_backing_free(ownership.handle)
          ownership.freed = true
        rescue StandardError
          nil # finalizers must never turn GC into an application exception
        end
      end

      def kind = :host

      def diagnostic = "pageable host backing"

      def validate!(buffer)
        return self if buffer.dtype == @dtype && buffer.length == @length && buffer.shape == @shape

        raise Error, "host backing does not match this buffer"
      end

      def write(buffer)
        validate!(buffer)
        Tilework.bridge.host_backing_store(@handle, buffer.__handle__, buffer.bytesize, 0)
        self
      end

      def restore_into(buffer, _meta = self)
        validate!(buffer)
        Tilework.bridge.host_backing_restore(@handle, buffer.__handle__, buffer.bytesize, 0)
        buffer
      end

      def async_available?
        Tilework.bridge.respond_to?(:host_backing_async_available?) && Tilework.bridge.host_backing_async_available?
      end

      def begin_restore(handle, bytes)
        return nil unless async_available?

        [:host, Tilework.bridge.host_backing_transfer_begin(@handle, handle, bytes, 0)]
      end

      def delete
        return if @ownership.freed

        @ownership.bridge.host_backing_free(@handle)
        @ownership.freed = true
      end
    end

    # @return [:staged, :posix] active file-transfer implementation
    #
    #   :staged — the bridge's pinned staging (O_DIRECT where possible)
    #   :posix  — the plain D2H -> Ruby String -> File.write path
    #
    # GPUDirect Storage (:gpudirect) is not implemented; on the test machine
    # cuFile only reaches compatibility mode and gains nothing.
    def transport
      return :posix if ENV["GPU_OFFLOAD_FORCE_POSIX"] == "1"
      return :posix unless Tilework.backend
      return :posix unless Tilework.bridge.respond_to?(:offload_available?) && Tilework.bridge.offload_available?

      :staged
    end

    # @return [Boolean] whether the backend supplies a native staged path
    def native? = transport == :staged

    # @return [Boolean] whether asynchronous restore is supported
    #
    # Whether the async transfer calls (transfer_begin/poll/end) exist. Without
    # them it falls back to the synchronous version and prefetch can no longer
    # hide transfers.
    def async?
      return false if ENV["GPU_OFFLOAD_FORCE_POSIX"] == "1"
      return false unless Tilework.backend

      Tilework.bridge.respond_to?(:transfer_available?) && Tilework.bridge.transfer_available?
    end

    # A pending asynchronous buffer restore.
    class Transfer
      def initialize(token, buffer, poll: :transfer_poll, finish: :transfer_end)
        @token = token
        @buffer = buffer
        @poll = poll
        @finish = finish
        @finished = token.nil?
      end

      # Polls without waiting.
      # @return [Boolean]
      def done?
        return true if @finished

        Tilework.bridge.public_send(@poll, @token) != 0
      end

      # Waits for the transfer and makes its buffer resident.
      # @return [Buffer]
      def wait
        return @buffer if @finished

        Tilework.bridge.public_send(@finish, @token)
        @finished = true
        @buffer.__finish_restore__
      end
    end

    # Builds the 4096-byte header. No endianness conversion — only x86_64 /
    # aarch64 are targeted, so Buffer's internal representation is written as is.
    def pack_header(dtype:, length:, shape:, checksum: nil)
      code = DTYPE_CODES[dtype] or raise ArgumentError, "cannot offload dtype #{dtype.inspect}"
      raise ArgumentError, "shape has too many dimensions: #{shape.length}" if shape.length > MAX_DIMS

      dims = shape + [0] * (MAX_DIMS - shape.length)
      flags = checksum ? 1 : 0
      head = [MAGIC, VERSION, code, length, shape.length, *dims, flags, checksum || 0]
             .pack("a8 L L Q L L#{MAX_DIMS} L Q")
      head + ("\0" * (HEADER_BYTES - head.bytesize))
    end

    def unpack_header(bytes)
      raise Error, "offload file is truncated (#{bytes.bytesize} bytes)" if bytes.bytesize < HEADER_BYTES

      magic, version, code, length, ndim, *rest = bytes.unpack("a8 L L Q L L#{MAX_DIMS} L Q")
      raise Error, "not a tilework offload file (bad magic)" unless magic == MAGIC
      raise Error, "unsupported offload file version #{version}" unless version == VERSION

      dtype = CODE_DTYPES[code] or raise Error, "unknown dtype code #{code} in offload file"
      dims = rest.first(MAX_DIMS)
      flags = rest[MAX_DIMS]
      checksum = rest[MAX_DIMS + 1]
      { dtype: dtype, length: length, shape: dims.first(ndim),
        checksum: flags.anybits?(1) ? checksum : nil }
    end

    # Buffer -> file. Uses the bridge's optional save ABI if implemented,
    # otherwise the plain D2H -> Ruby String -> File.write path. The file
    # contents are the same either way.
    def write_file(path, buffer, checksum: false)
      FileUtils.mkdir_p(File.dirname(path))

      # A checksum requires looking at the body in Ruby, which cancels the
      # native path's advantage (not routing D2H through a Ruby String), so
      # use the posix path in that case.
      if native? && !checksum
        write_file_native(path, buffer)
      else
        write_file_posix(path, buffer, checksum: checksum)
      end
      path
    end

    def write_file_posix(path, buffer, checksum: false)
      body = buffer.read_bytes
      sum = checksum ? checksum_of(body) : nil
      padding = (-body.bytesize) % ALIGN

      File.open(path, "wb") do |f|
        f.write(pack_header(dtype: buffer.dtype, length: buffer.length,
                            shape: buffer.shape, checksum: sum))
        f.write(body)
        f.write("\0" * padding) if padding.positive?
        # A checkpoint lost to a power failure is useless, so make it durable.
        # The native path fsyncs too, so both paths are equally durable.
        f.fsync
      end
    end

    # The header is written first with ordinary I/O and only the body goes
    # through the bridge's O_DIRECT. The file is bit-identical to the posix path's.
    def write_file_native(path, buffer)
      File.open(path, "wb") do |f|
        f.write(pack_header(dtype: buffer.dtype, length: buffer.length, shape: buffer.shape))
        # Reserve the O_DIRECT target (pwrite could extend the file, but
        # truncate explicitly so the size matches the posix path)
        f.truncate(HEADER_BYTES + ((buffer.bytesize + ALIGN - 1) / ALIGN * ALIGN))
      end
      Tilework.bridge.buffer_save(buffer.__handle__, path, buffer.bytesize, 0)
    end

    # File -> new Buffer. dtype/shape are fully restored from the header, so
    # callers need not remember the shape.
    def read_file(path)
      meta = read_header(path)
      buffer = Buffer.new(meta[:dtype], meta[:length], shape: meta[:shape])
      read_body_into(path, buffer, meta)
      buffer
    end

    def read_header(path)
      unpack_header(File.binread(path, HEADER_BYTES).to_s)
    end

    # Reads the body into an existing Buffer. Checksum verification needs the
    # body in Ruby, so that case uses the posix path.
    def read_body_into(path, buffer, meta)
      if native? && meta[:checksum].nil?
        Tilework.bridge.buffer_load(buffer.__handle__, path, buffer.bytesize, 0)
        return buffer
      end

      body = File.open(path, "rb") do |f|
        f.seek(HEADER_BYTES)
        f.read(buffer.bytesize).to_s
      end
      raise Error, "offload file body is truncated (#{body.bytesize} of #{buffer.bytesize} bytes)" if body.bytesize < buffer.bytesize

      if meta[:checksum] && checksum_of(body) != meta[:checksum]
        raise Error, "offload file checksum mismatch for #{path} (corrupted?)"
      end
      buffer.write_bytes(body)
      buffer
    end

    # The checksum is optional: off by default since it is wasted computation
    # for training spills, on for checkpoints. xxh64 is not in the standard
    # library, so FNV-1a folds to 64 bits without adding a dependency.
    def checksum_of(bytes)
      hash = 0xcbf29ce484222325
      bytes.each_byte do |b|
        hash = ((hash ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
      end
      hash
    end

    def store(dir)
      Store.new(dir)
    end

    # An alias of pipeline, for when the name should say the block runs once
    # per element (in the family of each_slice / each_with_index). To load one
    # or N buffers at once, use Tilework::Buffer.with_resident instead.
    def each_resident(buffers, depth: 2, write_back: false, &block)
      pipeline(buffers, depth: depth, write_back: write_back, &block)
    end

    # Makes an offloaded series resident in turn while prefetching ahead.
    #
    #   Tilework::Offload.pipeline(layers, depth: 2) do |buf|
    #     ...  # buf is resident; the next depth-1 are transferring meanwhile
    #   end
    #
    # depth: 2 means one element of prefetch. VRAM use scales with one
    # element's size x depth, so choose depth from the free VRAM. A Buffer
    # leaving the block goes back to its offload target, freeing VRAM.
    #
    # Whether transfers are hidden depends on the ratio of compute time to load
    # time: if per-element compute is shorter than its transfer, raising depth
    # adds no bandwidth and the waits remain.
    #
    # write_back: defaults to false. When streaming read-only weights, the
    # offload file does not change, so writing back only rewrites the same
    # contents, and as synchronous I/O it cannot overlap compute and defeats
    # hiding (measured: raising depth stayed at 1.0x). **Set it true only when
    # changes made in the block must be kept.**
    def pipeline(items, depth: 2, write_back: false)
      raise ArgumentError, "depth must be >= 1" unless depth.is_a?(Integer) && depth >= 1

      # An element may be a single Buffer or an array of Buffers. A real
      # model's "layer" is a group of tensors rather than one weight matrix (in a
      # transformer, heads x {wq,wk,wv,wo} + w1 + w2), and layer-wise streaming
      # needs groups to flow as a unit.
      #
      # Do not use Array() — it would call Buffer#to_a (a read from the device).
      groups = items.map { |item| item.is_a?(Array) ? item : [item] }

      pending = {}
      results = []
      prefetch = lambda do |i|
        next if i >= groups.length || pending.key?(i)

        pending[i] = groups[i].reject { |b| b.freed? || b.resident? }.map(&:restore_async!)
      end

      # At most depth elements occupy VRAM at once (one being computed plus
      # depth-1 prefetching), so depth: 1 means no prefetch and depth: 2 one
      # element ahead. With groups, VRAM is "group size x depth".
      depth.times { |i| prefetch.call(i) }

      begin
        items.each_with_index do |item, i|
          Offload.wait_all(pending.delete(i) || [])
          groups[i].each { |b| b.restore! unless b.resident? } # backstop for environments without async
          begin
            results << yield(item)
          ensure
            groups[i].each do |b|
              next if b.freed?

              write_back ? b.offload_back! : b.evict!
            end
            # Issue the next one only after one element's VRAM is free; issuing
            # earlier would exceed the depth limit.
            prefetch.call(i + depth)
          end
        end
      ensure
        # If the block raises (or breaks), prefetched but unconsumed elements
        # would keep holding VRAM. restore_async! allocates device memory before
        # the wait, so they hold it even while resident? is false. Always
        # collect them here.
        pending.each do |i, transfers|
          begin
            Offload.wait_all(transfers)
          rescue StandardError
            nil # a secondary failure during cleanup must not hide the original exception
          end
          groups[i].each { |b| b.evict! unless b.freed? }
        end
        pending.clear
      end
      results
    end

    # Collects every launched transfer before raising the first exception;
    # leaving early would strand worker threads, so wait on all of them even on failure.
    def wait_all(transfers)
      first_error = nil
      transfers.each do |t|
        t.wait
      rescue StandardError => e
        first_error ||= e
      end
      raise first_error if first_error
    end

    # A container for when file names should not be the caller's concern. A
    # Store is explicit persistence: it survives the end of a Scope and stays
    # until clear is called.
    class Store
      attr_reader :dir

      def initialize(dir)
        @dir = dir
        @seq = 0
        @handles = {}
        FileUtils.mkdir_p(dir)
      end

      # Stores a buffer and returns its generated handle. The buffer's VRAM is
      # released; get it back with the returned handle.
      # @param buffer [Buffer]
      # @param checksum [Boolean]
      # @return [String]
      def put(buffer, checksum: false)
        handle = (@seq += 1)
        path = File.join(@dir, format("%08d.rmgpu", handle))
        buffer.offload!(path, checksum: checksum)
        @handles[handle] = path
        handle
      end

      # Restores a stored buffer by handle.
      # @param handle [String]
      # @return [Buffer]
      def get(handle)
        path = @handles[handle] or raise Error, "unknown offload handle #{handle.inspect}"
        Offload.read_file(path)
      end

      # Deletes a stored buffer.
      # @param handle [String]
      # @return [nil]
      def delete(handle)
        path = @handles.delete(handle) or return false

        FileUtils.rm_f(path)
        true
      end

      # Deletes every buffer stored by this instance.
      # @return [nil]
      def clear
        @handles.each_value { |path| FileUtils.rm_f(path) }
        @handles.clear
        nil
      end

      def size = @handles.size
    end
  end
end
