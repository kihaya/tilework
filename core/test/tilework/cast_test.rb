# frozen_string_literal: true

require_relative "../test_helper"

# Regression: when the cast kernel was CUDA-only (#include / __global__),
# Buffer#cast(:float16) and fp16 #write of 4096+ elements raised CompileError
# on :metal. The source is now dialect and supported? is gated on capabilities.
class CastTest < Minitest::Test
  include BridgeTestSetup

  def with_backend(name)
    Tilework.instance_variable_set(:@backend, name)
    yield
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  def test_source_is_dialect_not_backend_native
    src = Tilework::Cast.source(Tilework::Cast::CONVERSIONS.fetch(%i[float32 float16]))
    assert_includes src, "GPU_KERNEL"
    assert_includes src, "GPU_ARGS"
    assert_includes src, "GLOBAL_ID_X"
    assert_includes src, "GPU_HALF"
    refute_includes src, "#include"
    refute_includes src, "__global__"
  end

  # Being dialect, the CUDA rewrite alone turns it into C++ for NVRTC.
  def test_source_expands_for_both_dialect_backends
    src = Tilework::Cast.source(Tilework::Cast::CONVERSIONS.fetch(%i[float16 float32]))
    with_backend(:cuda) do
      expanded = Tilework::Dialect.expand(src)
      assert_includes expanded, "extern \"C\" __global__"
      assert_includes expanded, "const int n" # constant int& -> by value
      # The implicit-argument comma goes too, leaving `f(const GPU_HALF* src, ...)`.
      assert_includes expanded, "void cast_kernel(const GPU_HALF* src"
    end
    with_backend(:metal) do
      expanded = Tilework::Dialect.expand(src)
      assert_includes expanded, "thread_position_in_grid"
      assert_includes expanded, "#define GPU_HALF    half"
    end
  end

  def test_bfloat16_cast_uses_the_portable_dialect_type
    src = Tilework::Cast.source(Tilework::Cast::CONVERSIONS.fetch(%i[float32 bfloat16]))
    assert_includes src, "GPU_BFLOAT"
    with_backend(:cuda) do
      expanded = Tilework::Dialect.expand(src)
      assert_includes expanded, "cuda_bf16.h"
      assert_includes expanded, "__nv_bfloat16"
    end
    with_backend(:hip) do
      expanded = Tilework::Dialect.expand(src)
      assert_includes expanded, "hip_bf16.h"
      assert_includes expanded, "__hip_bfloat16"
    end
    with_backend(:metal) do
      assert_includes Tilework::Dialect.expand(src), "#define GPU_BFLOAT  bfloat"
    end
  end

  # Being in the table does not mean the backend can actually convert.
  def test_supported_requires_the_dialect_and_the_dtypes
    with_backend(:metal) { assert Tilework::Cast.supported?(:float32, :float16) }
    with_backend(:cuda)  { assert Tilework::Cast.supported?(:float16, :float32) }
    with_backend(:metal) { assert Tilework::Cast.supported?(:float32, :bfloat16) }
    with_backend(:hip)   { assert Tilework::Cast.supported?(:bfloat16, :float32) }
    # :cpu accepts neither the dialect nor float16
    with_backend(:cpu)   { refute Tilework::Cast.supported?(:float32, :float16) }
    # :custom (MockBridge) accepts the dialect but cannot allocate float16
    refute Tilework::Cast.supported?(:float32, :float16)
  end

  def test_unsupported_pairs_stay_unsupported
    with_backend(:cuda) { refute Tilework::Cast.supported?(:float32, :int32) }
  end

  def test_convert_into_refuses_a_backend_that_cannot_run_it
    with_backend(:cpu) do
      src = Object.new
      def src.dtype = :float32
      dst = Object.new
      def dst.dtype = :float16
      error = assert_raises(Tilework::Error) { Tilework::Cast.convert_into(src, dst) }
      assert_match(/cannot convert/, error.message)
    end
  end
end
