# frozen_string_literal: true

require_relative "../test_helper"
require "tmpdir"

# VRAM <-> SSD offload. Runs on the :cpu backend, so CI covers it without a
# GPU. The offload layer only uses the bridge's buffer_read/write, so the
# semantics verified here carry over to the device backends.
class OffloadCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
    @dir = Dir.mktmpdir("rmgpu-offload")
  end

  def teardown
    FileUtils.remove_entry(@dir) if @dir && File.directory?(@dir)
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  def path(name = "buf.bin") = File.join(@dir, name)

  # --- Semantics ---

  # offload! releases VRAM. Without that it would just be a backup and would
  # not serve the purpose of offloading.
  def test_offload_releases_the_device_allocation_but_not_the_buffer
    b = Tilework[[1.0, 2.0, 3.0]]
    assert b.resident?
    b.offload!(path)
    refute b.resident?
    refute b.freed?, "the Buffer object itself stays usable (it can be restored)"
    assert File.exist?(path)
  end

  def test_path_offload_uses_file_backing_and_keeps_the_existing_format
    b = Tilework[[1.0, 2.0]]
    target = path("backing.bin")
    b.offload!(target)

    backing = b.instance_variable_get(:@backing)
    assert_instance_of Tilework::Offload::FileBacking, backing
    assert_equal :file, backing.kind
    assert_equal target, backing.path
    assert_equal Tilework::Offload::MAGIC, File.binread(target, 8)
    error = assert_raises(Tilework::NotResidentError) { b.to_a }
    assert_match(/file backing/, error.message)
  end

  def test_host_backing_uses_only_the_generic_opaque_bridge_contract
    bridge = Tilework::CPUBridge
    freed = []
    bridge.define_singleton_method(:host_backing_available?) { true }
    bridge.define_singleton_method(:host_backing_alloc) { |bytes| ("\0" * bytes).b }
    bridge.define_singleton_method(:host_backing_free) { |handle| freed << handle }
    bridge.define_singleton_method(:host_backing_store) do |host, device, bytes, offset|
      host.replace(buffer_read(device, bytes, offset))
    end
    bridge.define_singleton_method(:host_backing_restore) do |host, device, bytes, offset|
      buffer_write(device, host.byteslice(0, bytes), offset)
    end

    buffer = Tilework[[1.0, 2.0, 3.0]]
    buffer.offload_to_host!
    refute buffer.resident?
    assert_equal :host, buffer.instance_variable_get(:@backing).kind
    buffer.restore!
    assert_equal [1.0, 2.0, 3.0], buffer.to_a
    backing = buffer.instance_variable_get(:@last_backing)
    ownership = backing.instance_variable_get(:@ownership)
    backing.delete
    Tilework::Offload::HostBacking.finalizer(ownership).call
    assert_equal 1, freed.length, "explicit delete and GC finalization share exactly one ownership token"
  ensure
    %i[host_backing_available? host_backing_alloc host_backing_free host_backing_store host_backing_restore].each do |name|
      bridge.singleton_class.remove_method(name) if bridge&.singleton_methods(false)&.include?(name)
    end
  end

  # Computing on an offloaded buffer raises instead of restoring implicitly;
  # swap-like implicit restores would make performance unpredictable.
  def test_operating_on_an_offloaded_buffer_raises
    b = Tilework[[1.0, 2.0, 3.0]]
    b.offload!(path)

    assert_raises(Tilework::NotResidentError) { b + 1.0 }
    assert_raises(Tilework::NotResidentError) { b.to_a }
    assert_raises(Tilework::NotResidentError) { b.sum }
  end

  def test_restore_brings_the_values_back
    values = [1.5, -2.25, 3.75, 4.0]
    b = Tilework[values]
    b.offload!(path)
    b.restore!
    assert b.resident?
    assert_equal values, b.to_a
  end

  # An offload round trip must be bit-identical.
  def test_round_trip_is_bit_exact
    values = Array.new(1024) { |i| Math.sin(i) * 1e6 }
    original = Tilework[values]
    before = original.read_bytes

    original.offload!(path)
    original.restore!

    assert_equal before, original.read_bytes, "offload round-trip must be bit-exact"
  end

  def test_int32_buffers_round_trip
    b = Tilework[[1, -2, 3, 2_147_483_647]]
    assert_equal :int32, b.dtype
    b.offload!(path)
    b.restore!
    assert_equal [1, -2, 3, 2_147_483_647], b.to_a
  end

  def test_with_resident_restores_for_the_block_and_offloads_again
    b = Tilework[[1.0, 2.0, 3.0]]
    b.offload!(path)

    result = b.with_resident { |x| (x * 2.0).to_a }

    assert_equal [2.0, 4.0, 6.0], result
    refute b.resident?, "with_resident must put the buffer back after the block"
  end

  def test_with_resident_offloads_again_even_when_the_block_raises
    b = Tilework[[1.0, 2.0]]
    b.offload!(path)

    assert_raises(RuntimeError) { b.with_resident { raise "boom" } }
    refute b.resident?
  end

  # with_resident on an already resident Buffer does not offload it
  def test_with_resident_leaves_a_resident_buffer_alone
    b = Tilework[[1.0, 2.0]]
    b.with_resident { |x| x.to_a }
    assert b.resident?
  end

  # Freeing an offloaded Buffer also deletes its offload file.
  def test_free_deletes_the_offload_file
    b = Tilework[[1.0, 2.0]]
    b.offload!(path)
    assert File.exist?(path)
    b.free
    refute File.exist?(path), "freeing an offloaded buffer must remove its spill file"
  end

  # --- File format ---

  def test_header_is_4096_bytes_and_body_is_aligned
    b = Tilework[[1.0, 2.0, 3.0]] # a 12-byte body
    b.offload!(path)
    # 4096 (header) + 4096 (body rounded up to a 4096 boundary)
    assert_equal 8192, File.size(path)
    assert_equal "RMGPUOFF", File.binread(path, 8)
  end

  def test_shape_is_recovered_from_the_file
    m = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    m.offload!(path)
    restored = Tilework::Offload.read_file(path)
    assert_equal [2, 3], restored.shape
    assert_equal [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], restored.to_nested_a
  end

  def test_rejects_a_file_that_is_not_ours
    File.binwrite(path, "not an offload file" + ("\0" * 5000))
    error = assert_raises(Tilework::Error) { Tilework::Offload.read_file(path) }
    assert_match(/bad magic/, error.message)
  end

  def test_checksum_detects_corruption
    b = Tilework[[1.0, 2.0, 3.0, 4.0]]
    b.offload!(path, checksum: true)

    # Corrupt one byte of the body
    bytes = File.binread(path)
    bytes.setbyte(Tilework::Offload::HEADER_BYTES, bytes.getbyte(Tilework::Offload::HEADER_BYTES) ^ 0xFF)
    File.binwrite(path, bytes)

    error = assert_raises(Tilework::Error) { Tilework::Offload.read_file(path) }
    assert_match(/checksum mismatch/, error.message)
  end

  def test_checksum_is_off_by_default
    b = Tilework[[1.0, 2.0]]
    b.offload!(path)
    assert_nil Tilework::Offload.unpack_header(File.binread(path, Tilework::Offload::HEADER_BYTES))[:checksum]
  end

  # --- Store ---

  def test_store_put_and_get
    store = Tilework::Offload.store(File.join(@dir, "store"))
    m = Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0, 4.0]])

    handle = store.put(m)
    refute m.resident?, "put must free the device allocation"

    got = store.get(handle)
    assert_equal [[1.0, 2.0], [3.0, 4.0]], got.to_nested_a
    assert_equal [2, 2], got.shape
  end

  def test_store_get_returns_an_independent_buffer_each_time
    store = Tilework::Offload.store(File.join(@dir, "store"))
    handle = store.put(Tilework[[1.0, 2.0]])

    a = store.get(handle)
    b = store.get(handle)
    a.write([9.0, 9.0])
    assert_equal [1.0, 2.0], b.to_a
  end

  def test_store_clear_removes_the_files
    store = Tilework::Offload.store(File.join(@dir, "store"))
    store.put(Tilework[[1.0]])
    store.put(Tilework[[2.0]])
    assert_equal 2, store.size

    store.clear
    assert_equal 0, store.size
    assert_empty Dir.glob(File.join(@dir, "store", "*.rmgpu"))
  end

  def test_store_get_with_an_unknown_handle_raises
    store = Tilework::Offload.store(File.join(@dir, "store"))
    assert_raises(Tilework::Error) { store.get(999) }
  end

  # --- Loading several at once (Buffer.with_resident) ---

  def test_class_with_resident_makes_all_of_them_resident_at_once
    a, b, c = 3.times.map { |i| Tilework[[i.to_f, i + 1.0]].tap { |x| x.offload!(path("m#{i}.bin")) } }

    seen = Tilework::Buffer.with_resident(a, b, c) do |x, y, z|
      assert [x, y, z].all?(&:resident?), "all of them must be resident inside the block"
      [x.to_a, y.to_a, z.to_a]
    end

    assert_equal [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], seen
    assert [a, b, c].none?(&:resident?), "all of them must go back afterwards"
  end

  def test_class_with_resident_returns_the_block_value
    a = Tilework[[2.0, 3.0]]
    a.offload!(path)
    assert_equal 5.0, Tilework::Buffer.with_resident(a) { |x| x.to_a.sum }
  end

  # Leaves already-resident buffers alone (neither loads nor offloads them)
  def test_class_with_resident_leaves_already_resident_buffers_alone
    offloaded = Tilework[[1.0]].tap { |b| b.offload!(path("off.bin")) }
    live = Tilework[[2.0]]

    Tilework::Buffer.with_resident(offloaded, live) { |_, _| nil }

    refute offloaded.resident?, "the offloaded one goes back"
    assert live.resident?, "the resident one must be left alone"
  end

  def test_class_with_resident_puts_them_back_even_when_the_block_raises
    a, b = 2.times.map { |i| Tilework[[i.to_f]].tap { |x| x.offload!(path("r#{i}.bin")) } }
    assert_raises(RuntimeError) { Tilework::Buffer.with_resident(a, b) { raise "boom" } }
    assert [a, b].none?(&:resident?)
  end

  # --- write_back: ---

  # Writes back by default (so changes made in the block are not lost)
  def test_with_resident_writes_back_by_default
    b = Tilework[[1.0, 2.0]]
    b.offload!(path)
    b.with_resident { |x| x.write([7.0, 8.0]) }
    b.restore!
    assert_equal [7.0, 8.0], b.to_a
  end

  # Write-back can be skipped when the data is known to be read-only
  def test_with_resident_can_skip_the_write_back
    b = Tilework[[1.0, 2.0]]
    b.offload!(path)
    b.with_resident(write_back: false) { |x| x.write([7.0, 8.0]) }
    b.restore!
    assert_equal [1.0, 2.0], b.to_a, "write_back: false must leave the file untouched"
  end

  def test_class_with_resident_write_back_applies_to_all
    a, b = 2.times.map { |i| Tilework[[i.to_f]].tap { |x| x.offload!(path("w#{i}.bin")) } }
    Tilework::Buffer.with_resident(a, b, write_back: true) { |x, y| x.write([9.0]); y.write([9.0]) }
    a.restore!
    b.restore!
    assert_equal [[9.0], [9.0]], [a.to_a, b.to_a]
  end

  # --- each_resident (an alias of pipeline) ---

  def test_each_resident_is_an_alias_of_pipeline
    bufs = offloaded_series(3)
    assert_equal [0.0, 10.0, 20.0],
                 Tilework::Offload.each_resident(bufs, depth: 2) { |b| b.to_a.first }
    assert bufs.none?(&:resident?)
  end

  # Passing a never-offloaded Buffer to pipeline must give an error that names
  # the cause.
  def test_pipeline_on_a_never_offloaded_buffer_explains_itself
    error = assert_raises(Tilework::Error) do
      Tilework::Offload.pipeline([Tilework[[1.0]]], depth: 1) { |b| b.to_a }
    end
    assert_match(/never been offloaded/, error.message)
  end

  # --- pipeline ---
  #
  # :cpu has no async ABI, so this checks correctness of the synchronous
  # fallback. Whether transfers are actually hidden is checked on hardware.

  def offloaded_series(count)
    count.times.map do |i|
      b = Tilework[[i * 10.0, i * 10.0 + 1, i * 10.0 + 2]]
      b.offload!(path("l#{i}.bin"))
      b
    end
  end

  def test_pipeline_yields_each_buffer_resident_and_in_order
    bufs = offloaded_series(4)
    seen = []
    Tilework::Offload.pipeline(bufs, depth: 2) do |b|
      assert b.resident?, "the buffer must be resident inside the block"
      seen << b.to_a.first
    end
    assert_equal [0.0, 10.0, 20.0, 30.0], seen
  end

  def test_pipeline_evicts_each_buffer_after_its_turn
    bufs = offloaded_series(3)
    Tilework::Offload.pipeline(bufs, depth: 2) { |b| b.to_a }
    assert bufs.none?(&:resident?), "every buffer must be offloaded again afterwards"
  end

  def test_pipeline_returns_the_block_results
    bufs = offloaded_series(3)
    assert_equal [0.0, 10.0, 20.0], Tilework::Offload.pipeline(bufs, depth: 2) { |b| b.to_a.first }
  end

  # depth is how many elements occupy VRAM at once; depth: 1 means no
  # prefetch. The result must not depend on depth.
  def test_pipeline_results_do_not_depend_on_depth
    [1, 2, 3, 10].each do |depth|
      bufs = offloaded_series(4)
      got = Tilework::Offload.pipeline(bufs, depth: depth) { |b| b.to_a }
      assert_equal [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]], got,
                   "depth: #{depth}"
    end
  end

  def test_pipeline_holds_at_most_depth_buffers_resident
    bufs = offloaded_series(6)
    peak = 0
    Tilework::Offload.pipeline(bufs, depth: 2) do |_|
      peak = [peak, bufs.count(&:resident?)].max
    end
    assert_operator peak, :<=, 2, "more than depth buffers were resident at once"
  end

  # No write-back by default. For streaming read-only weights it would only
  # rewrite the same contents, and as synchronous I/O it would defeat hiding.
  def test_pipeline_does_not_write_back_by_default
    bufs = offloaded_series(2)
    Tilework::Offload.pipeline(bufs, depth: 2) { |b| b.write([99.0, 99.0, 99.0]) }
    bufs.first.restore!
    assert_equal [0.0, 1.0, 2.0], bufs.first.to_a, "changes must not be persisted by default"
  end

  def test_pipeline_write_back_persists_changes
    bufs = offloaded_series(2)
    Tilework::Offload.pipeline(bufs, depth: 2, write_back: true) { |b| b.write([99.0, 99.0, 99.0]) }
    bufs.first.restore!
    assert_equal [99.0, 99.0, 99.0], bufs.first.to_a
  end

  def test_pipeline_rejects_a_bad_depth
    assert_raises(ArgumentError) { Tilework::Offload.pipeline([], depth: 0) { } }
  end

  def test_pipeline_evicts_even_when_the_block_raises
    bufs = offloaded_series(2)
    assert_raises(RuntimeError) { Tilework::Offload.pipeline(bufs, depth: 2) { raise "boom" } }
    refute bufs.first.resident?
  end

  # --- Elements that are groups (a real model's "layer") ---
  #
  # A transformer layer is not one weight matrix but a group of tensors
  # (heads x {wq,wk,wv,wo} + w1 + w2). Layer-wise streaming needs groups to
  # flow as a unit, so an element may be a single Buffer or an array.

  def offloaded_layers(count, per_layer)
    count.times.map do |i|
      per_layer.times.map do |j|
        b = Tilework[[(i * 10 + j).to_f, (i * 10 + j + 0.5)]]
        b.offload!(path("l#{i}_#{j}.bin"))
        b
      end
    end
  end

  def test_pipeline_accepts_groups_and_yields_them_whole
    layers = offloaded_layers(3, 4)

    seen = Tilework::Offload.pipeline(layers, depth: 2) do |layer|
      assert_kind_of Array, layer
      assert_equal 4, layer.length
      assert layer.all?(&:resident?), "every tensor in the group must be resident"
      layer.map { |b| b.to_a.first }
    end

    assert_equal [[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]], seen
    assert layers.flatten.none?(&:resident?), "the whole group goes back afterwards"
  end

  def test_pipeline_groups_hold_at_most_depth_groups_resident
    layers = offloaded_layers(5, 3)
    peak = 0
    Tilework::Offload.pipeline(layers, depth: 2) do |_|
      peak = [peak, layers.count { |l| l.any?(&:resident?) }].max
    end
    assert_operator peak, :<=, 2, "more than depth groups were resident at once"
  end

  # Single Buffers and arrays can be mixed (backward compatible)
  def test_pipeline_accepts_a_mix_of_buffers_and_groups
    single = Tilework[[7.0]].tap { |b| b.offload!(path("single.bin")) }
    group = 2.times.map { |j| Tilework[[j.to_f]].tap { |b| b.offload!(path("g#{j}.bin")) } }

    shapes = Tilework::Offload.pipeline([single, group], depth: 2) do |item|
      item.is_a?(Array) ? :group : :buffer
    end
    assert_equal %i[buffer group], shapes
  end

  def test_pipeline_group_write_back_applies_to_every_member
    layers = offloaded_layers(2, 2)
    Tilework::Offload.pipeline(layers, depth: 2, write_back: true) do |layer|
      layer.each { |b| b.write([99.0, 99.0]) }
    end
    layers.flatten.each(&:restore!)
    assert layers.flatten.all? { |b| b.to_a == [99.0, 99.0] }
  end

  def test_pipeline_group_evicts_even_when_the_block_raises
    layers = offloaded_layers(2, 3)
    assert_raises(RuntimeError) { Tilework::Offload.pipeline(layers, depth: 2) { raise "boom" } }
    assert layers.flatten.none?(&:resident?)
  end

  # Leaving the loop with break must not keep prefetched, unconsumed elements resident
  def test_pipeline_releases_unconsumed_prefetches_on_break
    layers = offloaded_layers(6, 2)
    Tilework::Offload.pipeline(layers, depth: 3) { |_| break }
    assert layers.flatten.none?(&:resident?), "prefetched-but-unused groups still hold VRAM"
  end

  # Backends without the async ABI just fall back to the synchronous path.
  def test_async_is_reported_as_unavailable_on_the_cpu_backend
    refute Tilework::Offload.async?
    refute Tilework.capabilities[:offload_async]
  end

  # --- Backends without float16 ---

  # :cpu cannot store float16, so it fails at allocation rather than breaking
  # silently later.
  def test_float16_is_rejected_on_the_cpu_backend
    error = assert_raises(Tilework::DeviceError) { Tilework::Buffer.new(:float16, 4) }
    assert_match(/float16/, error.message)
    assert_match(/supported: :float32 \/ :int32 \/ :bfloat16/, error.message)
  end

  # The offload file's dtype code uses the number reserved from the start.
  def test_float16_has_a_reserved_offload_dtype_code
    assert_equal 2, Tilework::Offload::DTYPE_CODES.fetch(:float16)
    header = Tilework::Offload.pack_header(dtype: :float16, length: 8, shape: [8])
    assert_equal :float16, Tilework::Offload.unpack_header(header)[:dtype]
  end

  def test_bfloat16_round_trips_with_the_reserved_dtype_code
    assert_equal 3, Tilework::Offload::DTYPE_CODES.fetch(:bfloat16)
    values = [1.0, -2.5, 3.25, 65_536.0]
    path = File.join(@dir, "bfloat16.bin")
    buffer = Tilework[values, dtype: :bfloat16]
    expected = buffer.read_bytes
    buffer.offload!(path).restore!
    assert_equal :bfloat16, buffer.dtype
    assert_equal expected, buffer.read_bytes
    assert_equal values, buffer.to_a
  end

  # --- capability ---

  def test_capabilities_report_an_offload_path
    assert Tilework.capabilities[:offload]
    assert_equal :posix, Tilework.capabilities[:offload_transport]
  end
end
