# frozen_string_literal: true

require_relative "../test_helper"

class DialectTest < Minitest::Test
  KERNEL = "GPU_KERNEL void f(GPU_ARGS) {}"

  def test_expand_prepends_compat_header
    expanded = Tilework::Dialect.expand(KERNEL)
    assert expanded.start_with?("// tilework compat header")
    assert expanded.end_with?(KERNEL)
  end

  def test_expand_resets_line_numbers_for_user_source
    assert_includes Tilework::Dialect.expand(KERNEL), "#line 1 \"kernel\""
  end

  MACROS = %w[GPU_KERNEL GPU_ARGS GLOBAL_ID_X GLOBAL_ID_Y LOCAL_ID_X GROUP_ID_X GPU_SHARED GPU_BARRIER].freeze

  def test_compat_header_defines_the_dialect_macro_set
    header = Tilework::Dialect.compat_header
    MACROS.each { |macro| assert_includes header, "#define #{macro}" }
  end

  # --- Per-backend header selection and the CUDA rewrite ---
  #
  # Swaps the Tilework.backend ivar directly to exercise the branch without
  # loading a native extension.

  def with_backend(name)
    previous = Tilework.backend
    Tilework.instance_variable_set(:@backend, name)
    yield
  ensure
    Tilework.instance_variable_set(:@backend, previous)
  end

  def test_each_backend_gets_its_own_compat_header
    metal = with_backend(:metal) { Tilework::Dialect.compat_header }
    cuda  = with_backend(:cuda)  { Tilework::Dialect.compat_header }
    hip   = with_backend(:hip)   { Tilework::Dialect.compat_header }

    assert_includes metal, "#include <metal_stdlib>"
    assert_includes cuda, "__global__"
    refute_equal metal, cuda, "the header must not be memoized across backends"
    assert_includes hip, "__global__"
  end

  def test_cuda_header_defines_the_same_macro_set
    header = with_backend(:cuda) { Tilework::Dialect.compat_header }
    MACROS.each { |macro| assert_includes header, "#define #{macro}" }
  end

  def test_hip_header_defines_the_same_macro_set
    header = with_backend(:hip) { Tilework::Dialect.compat_header }
    MACROS.each { |macro| assert_includes header, "#define #{macro}" }
    assert_includes header, "hip/hip_fp16.h"
  end

  # Making `device` an empty macro also collapses CUDA's host_defines.h
  # `__device__` -> `__location__(device)`, leaving every later function
  # without an execution-space annotation so nothing compiles. Qualifiers must
  # be stripped from the dialect source (CUDA_REWRITES) instead.
  def test_cuda_header_does_not_erase_address_space_keywords_as_macros
    header = with_backend(:cuda) { Tilework::Dialect.compat_header }
    refute_match(/^#define device\s*$/, header)
    refute_match(/^#define constant\s*$/, header)
  end

  def test_unknown_backend_falls_back_to_the_metal_header
    header = with_backend(:custom) { Tilework::Dialect.compat_header }
    assert_includes header, "#include <metal_stdlib>"
  end

  def test_metal_source_is_not_rewritten
    src = "GPU_KERNEL void f(GPU_ARGS, device const float* a, constant int& n) {}"
    assert_equal src, with_backend(:metal) { Tilework::Dialect.rewrite(src) }
  end

  def test_cuda_rewrite_drops_gpu_args_and_its_comma
    src = "GPU_KERNEL void f(GPU_ARGS, device float* a) {}"
    got = with_backend(:cuda) { Tilework::Dialect.rewrite(src) }
    assert_equal "GPU_KERNEL void f(float* a) {}", got
  end

  def test_hip_uses_the_cpp_dialect_rewrites
    src = "GPU_KERNEL void f(GPU_ARGS, device float* a, constant int& n) {}"
    got = with_backend(:hip) { Tilework::Dialect.rewrite(src) }
    assert_equal "GPU_KERNEL void f(float* a, const int n) {}", got
  end

  def test_cuda_rewrite_keeps_bare_gpu_args_for_the_header_macro
    src = "GPU_KERNEL void f(GPU_ARGS) {}"
    assert_equal src, with_backend(:cuda) { Tilework::Dialect.rewrite(src) }
  end

  # Passed by reference it still compiles, and the value bytes dispatch passes
  # are silently read as a pointer. It must be by value.
  def test_cuda_rewrite_turns_reference_scalars_into_by_value
    src = "GPU_KERNEL void f(GPU_ARGS, constant int& n, constant float& s) {}"
    got = with_backend(:cuda) { Tilework::Dialect.rewrite(src) }
    assert_equal "GPU_KERNEL void f(const int n, const float s) {}", got
    refute_includes got, "&"
  end

  def test_cuda_rewrite_strips_address_space_qualifiers
    src = "GPU_KERNEL void f(GPU_ARGS, device const float* a, device float* c) {}"
    got = with_backend(:cuda) { Tilework::Dialect.rewrite(src) }
    assert_equal "GPU_KERNEL void f(const float* a, float* c) {}", got
  end

  def test_cuda_expand_prepends_the_cuda_header
    expanded = with_backend(:cuda) { Tilework::Dialect.expand(KERNEL) }
    assert_includes expanded, "__global__"
    assert expanded.end_with?(KERNEL)
  end
end
