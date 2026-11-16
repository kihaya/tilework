# frozen_string_literal: true

require_relative "test_helper"
require "open3"

# Tests that need a real HIP device (AMD GPU + ROCm); run with rake test:hip.
# Backend-independent Ops/DSL/with_gpu checks belong to conformance_hip_test.rb
# (Tilework::Conformance), so this file covers what is specific to the HIP
# bridge: raw HIP C++, dialect strings, compile errors, resource health.
class HipIntegrationTest < Minitest::Test
  VEC_ADD = <<~KERNEL
    GPU_KERNEL void vec_add(GPU_ARGS,
                            device const float* a,
                            device const float* b,
                            device float* c,
                            constant int& n) {
      int i = GLOBAL_ID_X;
      if (i < n) c[i] = a[i] + b[i];
    }
  KERNEL

  FILL_2D = <<~KERNEL
    GPU_KERNEL void fill_2d(GPU_ARGS,
                            device float* out,
                            constant int& width,
                            constant int& height) {
      int x = GLOBAL_ID_X;
      int y = GLOBAL_ID_Y;
      if (x < width && y < height) out[y * width + x] = (float)(y * 100 + x);
    }
  KERNEL

  FILL_3D = <<~HIP
    extern "C" __global__ void fill_3d(int* out, int width, int height, int depth) {
      int x = blockIdx.x * blockDim.x + threadIdx.x;
      int y = blockIdx.y * blockDim.y + threadIdx.y;
      int z = blockIdx.z * blockDim.z + threadIdx.z;
      if (x < width && y < height && z < depth) {
        out[(z * height + y) * width + x] = z * 100 + y * 10 + x;
      }
    }
  HIP

  def setup
    skip "GPU tests disabled (set GPU_TESTS=1 or run rake test:hip)" unless gpu_tests_enabled?
    skip "not running on the HIP backend" unless gpu_backend == :hip
    Tilework.backend = :hip
    Tilework::Kernel.clear_cache!
  end

  def assert_float_arrays_equal(expected, actual)
    assert_equal expected.length, actual.length
    expected.zip(actual).each_with_index do |(e, a), i|
      assert_in_delta e, a, 1e-6, "element #{i}"
    end
  end

  def test_device_name_is_present
    refute_empty Tilework.device_name
  end

  def test_device_info_reports_runtime_architecture_without_claiming_cuda_capability
    info = Tilework.bridge.device_info
    assert_equal Tilework.device_name, info[:name]
    assert_match(/\Agfx/, info[:architecture])
    assert_equal Integer(ENV.fetch("TILEWORK_DEVICE", "0"), 10), info[:index]
    assert_nil Tilework.capabilities[:compute_capability]
    refute Tilework.capabilities[:tf32]
  end

  def test_invalid_device_selection_never_falls_back_to_zero
    root = File.expand_path("../..", __dir__)
    env = { "TILEWORK_DEVICE" => "99", "RUBYLIB" => "#{root}/core/lib:#{root}/hip/lib" }
    _out, error, status = Open3.capture3(env, RbConfig.ruby, "-e",
                                         'require "tilework"; require "tilework/backend/hip"; Tilework.backend = :hip; Tilework.device_name')
    refute status.success?
    assert_match(/outside the visible HIP device count/, error)
  end

  def test_malformed_device_selection_never_falls_back_to_zero
    root = File.expand_path("../..", __dir__)
    env = { "TILEWORK_DEVICE" => "not-a-number", "RUBYLIB" => "#{root}/core/lib:#{root}/hip/lib" }
    _out, error, status = Open3.capture3(env, RbConfig.ruby, "-e",
                                         'require "tilework"; require "tilework/backend/hip"; Tilework.backend = :hip; Tilework.device_name')
    refute status.success?
    assert_match(/invalid TILEWORK_DEVICE=not-a-number/, error)
  end

  def test_hipblas_execution_probe_is_the_only_basis_for_gemm_capability
    caps = Tilework.capabilities
    assert_equal :hipblas, caps[:gemm_library]
    refute caps[:gemm_batched]

    a = Tilework[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    b = Tilework[[[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]]
    assert_equal [58.0, 64.0, 139.0, 154.0], a.matmul(b).to_a
  end

  def test_hipblas_fp16_inputs_accumulate_into_float32
    a = Tilework[[[1.0, 2.0], [3.0, 4.0]], dtype: :float16]
    b = Tilework[[[5.0, 6.0], [7.0, 8.0]], dtype: :float16]
    out = a.matmul(b)
    assert_equal :float32, out.dtype
    assert_float_arrays_equal [19.0, 22.0, 43.0, 50.0], out.to_a
  end

  def test_rocwmma_probe_and_fp16_matrix_core_path
    caps = Tilework.capabilities
    assert caps[:tensor_core]
    assert caps[:fp16]
    refute caps[:tf32]

    a_values = Array.new(16) { |row| Array.new(16) { |col| ((row + col) % 5 - 2).to_f } }
    b_values = Array.new(16) { |row| Array.new(16) { |col| ((row * 3 + col) % 7 - 3).to_f } }
    expected = a_values.flat_map do |row|
      (0...16).map { |col| row.zip(b_values.map { |r| r[col] }).sum { |x, y| x * y } }
    end
    out = Tilework[a_values, dtype: :float16].matmul(Tilework[b_values, dtype: :float16])
    assert_equal :float32, out.dtype
    assert_float_arrays_equal expected, out.to_a
  end

  def test_rocwmma_ineligible_shape_uses_hipblas_and_remains_correct
    a = Tilework[[[1.0, 2.0], [3.0, 4.0]], dtype: :float16]
    b = Tilework[[[5.0, 6.0], [7.0, 8.0]], dtype: :float16]
    assert_float_arrays_equal [19.0, 22.0, 43.0, 50.0], a.matmul(b).to_a
  end

  def test_hiprand_native_uniform_is_reproducible_and_in_range
    assert_equal :hiprand, Tilework.capabilities[:rng]
    Tilework::Random.native!
    first = Tilework.random(257, seed: 12_345).to_a
    second = Tilework.random(257, seed: 12_345).to_a
    assert_equal first, second
    assert first.all? { |value| value.positive? && value <= 1.0 }
  end

  def test_hiprand_native_normal_handles_odd_lengths
    Tilework::Random.native!
    values = Tilework.randn(10_001, mean: 2.0, stddev: 3.0, seed: 54_321).to_a
    mean = values.sum / values.length
    variance = values.sum { |value| (value - mean)**2 } / values.length
    assert_in_delta 2.0, mean, 0.15
    assert_in_delta 9.0, variance, 0.7
  end

  def test_hipfft_real_round_trip_and_batched_execution
    assert_equal :hipfft, Tilework.capabilities[:fft]
    input = Tilework[[1.0, 0.0, -1.0, 0.0, 2.0, 1.0, 0.0, -1.0]]
    spectrum = Tilework::FFT.forward(input, batch: 2)
    restored = Tilework::FFT.inverse(spectrum, 4, batch: 2)
    assert_float_arrays_equal input.to_a, restored.to_a
  end

  def test_hipfft_complex_forward_inverse_round_trip
    # Four interleaved complex values.
    input = Tilework[[1.0, 0.0, 2.0, -1.0, -3.0, 0.5, 0.0, 2.0]]
    transformed = Tilework::FFT.transform(input)
    restored = Tilework::FFT.transform(transformed, direction: :inverse)
    assert_float_arrays_equal input.to_a, restored.to_a
  end

  def test_miopen_forward_probe_and_nchw_padding_stride
    assert_equal :miopen, Tilework::Conv.library
    input = buffer(:float32, [1, 1, 3, 3], (1..9).map(&:to_f))
    weight = buffer(:float32, [1, 1, 2, 2], [1.0, 0.0, 0.0, 1.0])
    output = Tilework::Conv.forward(input, weight, pad: 1, stride: 2)
    assert_equal [1, 1, 2, 2], output.shape
    assert_float_arrays_equal [1.0, 3.0, 7.0, 14.0], output.to_a
  end

  def test_miopen_grouped_forward_and_float16_storage
    input = buffer(:float32, [1, 2, 2, 2], (1..8).map(&:to_f))
    weight = buffer(:float32, [2, 1, 1, 1], [2.0, 3.0])
    output = Tilework::Conv.forward(input, weight, groups: 2)
    assert_float_arrays_equal [2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0], output.to_a

    half_input = buffer(:float16, [1, 1, 2, 2], [1.0, 2.0, 3.0, 4.0])
    half_weight = buffer(:float16, [1, 1, 1, 1], [0.5])
    half_output = Tilework::Conv.forward(half_input, half_weight)
    assert_equal :float16, half_output.dtype
    assert_float_arrays_equal [0.5, 1.0, 1.5, 2.0], half_output.to_a
  end

  def test_miopen_backward_data_and_filter_follow_forward_plan
    input = buffer(:float32, [1, 1, 3, 3], (1..9).map(&:to_f))
    weight = buffer(:float32, [1, 1, 1, 1], [2.0])
    output = Tilework::Conv.forward(input, weight)
    gradient = buffer(:float32, output.shape, Array.new(output.length, 1.0))
    dx, dw = Tilework::Conv.backward(input, weight, gradient)
    assert Tilework::Conv.backward_available?
    assert_float_arrays_equal Array.new(9, 2.0), dx.to_a
    assert_float_arrays_equal [45.0], dw.to_a
  end

  def test_pageable_host_backing_round_trip_and_write_back
    assert Tilework::HIPBridge.host_backing_available?
    original = Array.new(1025) { |i| (i - 512).to_f / 8 }
    buffer = buffer(:float32, [1025], original)
    buffer.offload_to_host!
    refute buffer.resident?
    assert_equal :host, buffer.instance_variable_get(:@backing).kind
    assert_raises(Tilework::NotResidentError) { buffer.to_a }

    buffer.with_resident { |value| value.write(Array.new(1025, 3.5)) }
    refute buffer.resident?
    buffer.restore!
    assert_float_arrays_equal Array.new(1025, 3.5), buffer.to_a
  end

  def test_pageable_host_backing_free_is_idempotent_after_restore
    buffer = buffer(:float32, [4], [1.0, 2.0, 3.0, 4.0])
    buffer.offload_to_host!
    buffer.restore!
    backing = buffer.instance_variable_get(:@last_backing)
    buffer.free
    buffer.free
    assert backing.instance_variable_get(:@freed)
  end

  def test_pageable_host_backing_crosses_the_pinned_staging_boundary
    # One 1 MiB staging chunk + 3 float32 values: two staged transfers.
    count = (1024 * 1024 / 4) + 3
    values = Array.new(count) { |i| ((i % 97) - 48).to_f }
    buffer = buffer(:float32, [count], values)
    buffer.offload_to_host!
    buffer.restore!
    assert_float_arrays_equal values, buffer.to_a
  end

  def test_pageable_host_backing_async_restore_is_opt_in_and_token_fenced
    values = Array.new(262_147, 1.75)
    buffer = buffer(:float32, [values.length], values)
    buffer.offload_to_host!
    assert Tilework.capabilities[:host_backing_async]
    transfer = buffer.restore_async!
    refute buffer.resident?
    transfer.wait
    assert buffer.resident?
    assert_float_arrays_equal values, buffer.to_a

    buffer.offload_to_host!
    synchronous = buffer.restore_async!(async: false)
    assert synchronous.done?
    assert buffer.resident?
  end

  def test_pending_async_restore_rejects_second_restore_and_free_until_waited
    values = Array.new(262_147, 1.75)
    buffer = buffer(:float32, [values.length], values)
    buffer.offload_to_host!
    transfer = buffer.restore_async!

    assert_raises(Tilework::Error) { buffer.restore_async! }
    assert_raises(Tilework::Error) { buffer.restore! }
    assert_raises(Tilework::Error) { buffer.free }
    refute buffer.resident?

    transfer.wait
    assert buffer.resident?
    assert_float_arrays_equal values, buffer.to_a
    buffer.free
    assert buffer.freed?
  end

  def test_pageable_host_backing_preserves_float16_and_int32_payloads
    half_values = [-3.5, -0.25, 0.0, 1.5, 17.75]
    half = buffer(:float16, [half_values.length], half_values)
    half.offload_to_host!
    half.restore!
    assert_float_arrays_equal half_values, half.to_a

    int_values = [-2_147_483_648, -7, 0, 42, 2_147_483_647]
    integers = buffer(:int32, [int_values.length], int_values)
    integers.offload_to_host!
    integers.restore!
    assert_equal int_values, integers.to_a
  end

  def test_graph_residency_can_use_budgeted_host_backing
    weights = 3.times.map { buffer(:float32, [4, 4], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]) }
    graph = Tilework::Graph.build do |g|
      x = g.input(:x, [4, 4])
      weights.each { |weight| x = x.matmul(g.param(weight)) }
      g.output(x)
    end
    probe = graph.compile(vram: 1 << 30)
    reserved = probe.residency.reserved
    assert_raises(Tilework::Error) { graph.compile(vram: reserved + 128, backing: :host, host_budget: 64) }

    plan = graph.compile(vram: reserved + 128, backing: :host, host_budget: 192)
    result = plan.run(x: buffer(:float32, [4, 4], (1..16).map(&:to_f)))
    assert_float_arrays_equal (1..16).map(&:to_f), result.to_a
    assert_operator plan.offload_stats[:spills], :>, 0
  ensure
    probe&.free_offload!
    plan&.free_offload!
  end

  def test_buffer_roundtrip_on_device
    values = Array.new(1000) { rand(-100.0..100.0) }
    expected = values.pack("f*").unpack("f*") # the reference is the value rounded to float32
    assert_float_arrays_equal expected, Tilework[values].to_a
  end

  def test_vec_add_matches_cpu_reference
    n = 4096
    a = Array.new(n) { rand }
    b = Array.new(n) { rand }
    da = Tilework[a]
    db = Tilework[b]
    dc = Tilework::Buffer.new(:float32, n)

    kernel = Tilework.compile(VEC_ADD, name: "vec_add")
    kernel.launch(grid: (n / 256.0).ceil, block: 256, args: [da, db, dc, n])

    expected = a.zip(b).map { |x, y| x + y }
    assert_float_arrays_equal expected, dc.to_a
  end

  def test_two_dimensional_dispatch_maps_grid_and_block_axes_correctly
    width = 37
    height = 19
    output = Tilework::Buffer.new(:float32, width * height)
    kernel = Tilework.compile(FILL_2D, name: "fill_2d")
    kernel.launch(grid: [(width / 8.0).ceil, (height / 4.0).ceil], block: [8, 4],
                  args: [output, width, height])

    expected = (0...height).flat_map { |y| (0...width).map { |x| (y * 100 + x).to_f } }
    assert_float_arrays_equal expected, output.to_a
  end

  def test_three_dimensional_dispatch_maps_grid_and_block_axes_correctly
    width = height = depth = 4
    output = Tilework::Buffer.new(:int32, width * height * depth)
    kernel = Tilework.compile(FILL_3D, name: "fill_3d", raw: true)
    kernel.launch(grid: [2, 2, 2], block: [2, 2, 2], args: [output, width, height, depth])

    expected = (0...depth).flat_map do |z|
      (0...height).flat_map { |y| (0...width).map { |x| z * 100 + y * 10 + x } }
    end
    assert_equal expected, output.to_a
  end

  def test_view_aliases_device_memory_and_freeing_view_keeps_base_usable
    base = Tilework[(0...12).map(&:to_f)]
    view = base.view(4, 4)
    kernel = Tilework.compile(<<~KERNEL, name: "scale_view")
      GPU_KERNEL void scale_view(GPU_ARGS, device float* values, constant int& n) {
        int i = GLOBAL_ID_X;
        if (i < n) values[i] *= 10.0f;
      }
    KERNEL
    kernel.launch(grid: 1, block: 4, args: [view, 4])
    view.free

    assert_float_arrays_equal [0.0, 1.0, 2.0, 3.0, 40.0, 50.0, 60.0, 70.0, 8.0, 9.0, 10.0, 11.0], base.to_a
  end

  def test_launch_async_and_to_a_waits_for_completion
    n = 4096
    a = Array.new(n) { rand }
    da = Tilework[a]
    db = Tilework[a]
    dc = Tilework::Buffer.new(:float32, n)

    kernel = Tilework.compile(VEC_ADD, name: "vec_add")
    kernel.launch_async(grid: (n / 256.0).ceil, block: 256, args: [da, db, dc, n])

    expected = a.map { |x| x * 2 }
    assert_float_arrays_equal expected, dc.to_a
  end

  def test_int32_kernel_arguments
    n = 256
    src = <<~KERNEL
      GPU_KERNEL void scale(GPU_ARGS, device int* a, constant int& factor, constant int& n) {
        int i = GLOBAL_ID_X;
        if (i < n) a[i] = a[i] * factor;
      }
    KERNEL
    buffer = Tilework[(1..n).to_a]
    Tilework.compile(src, name: "scale").launch(grid: 1, block: n, args: [buffer, 3, n])
    assert_equal (1..n).map { |v| v * 3 }, buffer.to_a
  end

  def test_compile_error_includes_hiprtc_compiler_output
    error = assert_raises(Tilework::CompileError) do
      Tilework.compile("GPU_KERNEL void broken(GPU_ARGS) { undefined_fn(); }", name: "broken")
    end
    assert_match(/error/i, error.message)
  end

  def test_raw_hip_escape_hatch
    src = <<~HIP
      extern "C" __global__ void fill(float* out) {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        out[gid] = 42.0f;
      }
    HIP
    buffer = Tilework::Buffer.new(:float32, 64)
    Tilework.compile(src, name: "fill", raw: true).launch(grid: 1, block: 64, args: [buffer])
    assert buffer.to_a.all? { |v| (v - 42.0).abs < 1e-6 }
  end

  def test_block_limit_validation_raises_before_dispatch
    kernel = Tilework.compile(VEC_ADD, name: "vec_add")
    assert_raises(Tilework::LaunchError) do
      kernel.launch(grid: 1, block: 100_000, args: [])
    end
  end

  def test_buffer_alloc_free_cycles_do_not_grow_rss
    # Warm up first. RSS must not grow monotonically over 2000 cycles of 1 KB.
    50.times { Tilework::Buffer.new(:float32, 256).free }
    GC.start
    before = rss_kb
    2000.times { Tilework::Buffer.new(:float32, 256).free }
    GC.start
    after = rss_kb
    assert_operator after - before, :<, 20_000, "RSS grew from #{before}KB to #{after}KB"
  end

  def test_hiprtc_compile_dispatch_and_module_teardown_repeat_stably
    run_hiprtc_cycle = lambda do
      input = Tilework[[1.0, 2.0, 3.0, 4.0]]
      output = Tilework::Buffer.new(:float32, 4)
      Tilework.compile(VEC_ADD, name: "vec_add").launch(grid: 1, block: 4, args: [input, input, output, 4])
      assert_float_arrays_equal [2.0, 4.0, 6.0, 8.0], output.to_a
      Tilework::Kernel.clear_cache!
      input.free
      output.free
      GC.start
    end
    # ROCm's first HIPRTC compilation materializes a sizeable process-global
    # JIT cache. Measure Tilework module ownership after that one-time cost.
    3.times { run_hiprtc_cycle.call }
    baseline = rss_kb
    10.times { run_hiprtc_cycle.call }
    assert_operator rss_kb - baseline, :<, 30_000, "steady-state HIPRTC loop grew RSS from #{baseline}KB"
  end

  private

  def rss_kb
    `ps -o rss= -p #{Process.pid}`.to_i
  end

  def buffer(dtype, shape, values)
    Tilework::Buffer.new(dtype, values.length, shape: shape).tap { |value| value.write(values) }
  end
end
