# frozen_string_literal: true

require_relative "test_helper"

# Tests that need a real CUDA device (NVIDIA GPU + CUDA Toolkit); run with
# rake test:cuda. Backend-independent Ops/DSL/with_gpu checks belong to
# conformance_cuda_test.rb (Tilework::Conformance), so this file covers what is
# specific to the CUDA bridge: raw CUDA C++, dialect strings, compile errors,
# resource health.
class CudaIntegrationTest < Minitest::Test
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

  def setup
    skip "GPU tests disabled (set GPU_TESTS=1 or run rake test:cuda)" unless gpu_tests_enabled?
    skip "not running on the CUDA backend" unless gpu_backend == :cuda
    Tilework.backend = :cuda
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

  def test_compile_error_includes_nvrtc_compiler_output
    error = assert_raises(Tilework::CompileError) do
      Tilework.compile("GPU_KERNEL void broken(GPU_ARGS) { undefined_fn(); }", name: "broken")
    end
    assert_match(/error/i, error.message)
  end

  def test_raw_cuda_escape_hatch
    src = <<~CUDA
      extern "C" __global__ void fill(float* out) {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        out[gid] = 42.0f;
      }
    CUDA
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

  private

  def rss_kb
    `ps -o rss= -p #{Process.pid}`.to_i
  end
end
