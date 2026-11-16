# frozen_string_literal: true

require_relative "../test_helper"

class KernelTest < Minitest::Test
  include BridgeTestSetup

  SRC = "GPU_KERNEL void f(GPU_ARGS, device float* a) {}"

  def compile(src = SRC, name: "f", raw: false)
    Tilework.compile(src, name: name, raw: raw)
  end

  def test_compile_injects_compat_header
    compile
    assert_includes @bridge.compilations.last[:src], "#define GPU_KERNEL"
  end

  def test_compile_raw_skips_header_injection
    compile(raw: true)
    refute_includes @bridge.compilations.last[:src], "#define GPU_KERNEL"
    assert_equal SRC, @bridge.compilations.last[:src]
  end

  def test_pipeline_cache_hits_for_identical_source_and_name
    compile
    compile
    assert_equal 1, @bridge.compilations.size
  end

  def test_pipeline_cache_misses_for_different_name
    compile(name: "f")
    @bridge.compile_error = nil
    compile("GPU_KERNEL void g(GPU_ARGS) {}", name: "g")
    assert_equal 2, @bridge.compilations.size
  end

  def test_changing_bridge_drains_work_and_drops_the_old_pipeline_cache
    compile
    old_bridge = @bridge
    replacement = MockBridge.new

    Tilework.bridge = replacement

    assert_equal 1, old_bridge.synchronize_count
    assert_equal 0, Tilework::Kernel.cache_size
    compile
    assert_equal 1, replacement.compilations.size
  end

  def test_kernel_compiled_before_bridge_change_recompiles_for_the_new_bridge
    kernel = compile
    replacement = MockBridge.new
    Tilework.bridge = replacement

    kernel.launch(grid: 1, block: 1)

    assert_equal 1, replacement.compilations.size
    assert_equal 1, replacement.dispatches.size
  end

  def test_cache_key_is_sha256_of_source_and_name
    key = Tilework::Kernel.cache_key("src", "fn")
    assert_equal Digest::SHA256.hexdigest("src\0fn"), key
  end

  def test_compile_error_is_normalized_and_keeps_compiler_output
    @bridge.compile_error = "program_source:3:5: error: use of undeclared identifier 'x'"
    error = assert_raises(Tilework::CompileError) { compile }
    assert_includes error.message, "undeclared identifier 'x'"
  end

  def test_launch_converts_scalar_args_to_packed_bytes
    kernel = compile
    kernel.launch(grid: 4, block: 256, args: [1.5, 7])
    args = @bridge.dispatches.last[:args]
    assert_equal [1.5].pack("f"), args[0]
    assert_equal [7].pack("l"), args[1]
  end

  def test_launch_passes_buffer_handles
    kernel = compile
    buffer = Tilework::Buffer.new(:float32, 8)
    kernel.launch(grid: 1, block: 8, args: [buffer])
    assert_instance_of MockBridge::Handle, @bridge.dispatches.last[:args][0]
  end

  def test_launch_normalizes_integer_grid_and_block_to_3d
    compile.launch(grid: 4, block: 256, args: [])
    dispatch = @bridge.dispatches.last
    assert_equal [4, 1, 1], dispatch[:grid]
    assert_equal [256, 1, 1], dispatch[:block]
    assert dispatch[:wait]
  end

  def test_launch_async_does_not_wait
    compile.launch_async(grid: 1, block: 1, args: [])
    refute @bridge.dispatches.last[:wait]
  end

  def test_launch_rejects_block_exceeding_pipeline_limit
    error = assert_raises(Tilework::LaunchError) { compile.launch(grid: 1, block: 2048, args: []) }
    assert_includes error.message, "1024"
  end

  def test_launch_rejects_nonpositive_dimensions
    assert_raises(Tilework::LaunchError) { compile.launch(grid: 0, block: 1, args: []) }
    assert_raises(Tilework::LaunchError) { compile.launch(grid: 1, block: [1, -1], args: []) }
  end

  def test_launch_rejects_too_many_args
    args = Array.new(Tilework::MAX_KERNEL_ARGS + 1, 1)
    error = assert_raises(Tilework::LaunchError) { compile.launch(grid: 1, block: 1, args: args) }
    assert_includes error.message, "too many"
  end

  def test_launch_rejects_unsupported_arg_types
    error = assert_raises(Tilework::LaunchError) { compile.launch(grid: 1, block: 1, args: ["nope"]) }
    assert_includes error.message, "index 0"
  end
end
