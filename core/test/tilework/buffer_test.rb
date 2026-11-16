# frozen_string_literal: true

require_relative "../test_helper"

class BufferTest < Minitest::Test
  include BridgeTestSetup

  def test_from_infers_int32_for_all_integers
    assert_equal :int32, Tilework[[1, 2, 3]].dtype
  end

  def test_from_infers_float32_for_mixed_numerics
    assert_equal :float32, Tilework[[1, 2.5]].dtype
  end

  def test_roundtrip_float32
    values = [1.0, -2.5, 3.25]
    assert_equal values, Tilework[values].to_a
  end

  def test_roundtrip_int32
    values = [1, -2, 2_000_000_000]
    assert_equal values, Tilework[values].to_a
  end

  def test_roundtrip_bfloat16_on_cpu
    Tilework.backend = :cpu
    values = [1.0, -2.5, 3.25, 65_536.0]
    buffer = Tilework[values, dtype: :bfloat16]
    assert_equal :bfloat16, buffer.dtype
    assert_equal values.length * 2, buffer.bytesize
    assert_equal values, buffer.to_a
    assert_equal Tilework::BFloat.encode(values), buffer.read_bytes
  ensure
    Tilework.bridge = @bridge
  end

  def test_to_a_synchronizes_pending_work_first
    buffer = Tilework[[1.0]] # the write at creation synchronizes once
    before = @bridge.synchronize_count
    buffer.to_a
    assert_equal before + 1, @bridge.synchronize_count
  end

  def test_bytesize
    assert_equal 40, Tilework::Buffer.new(:float32, 10).bytesize
  end

  def test_rejects_unknown_dtype
    assert_raises(ArgumentError) { Tilework::Buffer.new(:float64, 4) }
  end

  def test_rejects_empty_array
    assert_raises(ArgumentError) { Tilework[[]] }
  end

  def test_explicit_free_marks_buffer_unusable
    buffer = Tilework[[1.0]]
    buffer.free
    assert buffer.freed?
    assert_raises(Tilework::Error) { buffer.to_a }
  end

  def test_free_is_idempotent
    buffer = Tilework[[1.0]]
    buffer.free
    buffer.free
  end

  def test_default_shape_is_1d
    assert_equal [3], Tilework[[1.0, 2.0, 3.0]].shape
    refute Tilework[[1.0, 2.0, 3.0]].matrix?
  end

  def test_from_matrix_builds_2d_shape_row_major
    m = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_equal [2, 3], m.shape
    assert m.matrix?
    assert_equal 2, m.rows
    assert_equal 3, m.cols
    assert_equal [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], m.to_a
    assert_equal [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], m.to_nested_a
  end

  def test_from_matrix_infers_int32
    assert_equal :int32, Tilework::Buffer.from_matrix([[1, 2], [3, 4]]).dtype
  end

  def test_from_matrix_rejects_ragged_rows
    assert_raises(ArgumentError) { Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0]]) }
  end

  def test_reshape_via_shape_kwarg
    buf = Tilework::Buffer.new(:float32, 6, shape: [2, 3])
    assert_equal [2, 3], buf.shape
  end

  def test_shape_mismatched_with_length_raises
    assert_raises(ArgumentError) { Tilework::Buffer.new(:float32, 6, shape: [2, 4]) }
  end

  def test_rows_cols_raise_on_non_matrix
    buf = Tilework[[1.0, 2.0]]
    assert_raises(Tilework::Error) { buf.rows }
    assert_raises(Tilework::Error) { buf.cols }
  end

  def test_to_nested_a_on_1d_buffer_returns_flat_array
    assert_equal [1.0, 2.0], Tilework[[1.0, 2.0]].to_nested_a
  end

  def test_from_auto_detects_nested_array_as_matrix
    m = Tilework::Buffer.from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_equal [2, 3], m.shape
    assert m.matrix?
    assert_equal [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], m.to_nested_a
  end

  def test_from_rejects_mixed_nested_and_scalar_elements
    assert_raises(ArgumentError) { Tilework::Buffer.from([[1.0, 2.0], 3.0]) }
  end

  # initialize registers ObjectSpace.define_finalizer(self,
  # self.class.finalizer(...)) so bridge.buffer_free runs even when a Buffer
  # is collected without #free. Actual GC timing is unreliable under minitest
  # (conservative stack scanning, references held by the runner), so this
  # calls the registered finalizer directly instead of going through GC.
  def test_finalizer_calls_bridge_buffer_free
    buf = Tilework::Buffer.new(:float32, 4)
    handle = buf.instance_variable_get(:@handle)
    refute handle.freed

    Tilework::Buffer.finalizer(handle, @bridge).call

    assert handle.freed
  end

  def test_finalizer_after_explicit_free_does_not_raise
    buf = Tilework::Buffer.new(:float32, 4)
    handle = buf.instance_variable_get(:@handle)
    buf.free
    assert handle.freed

    Tilework::Buffer.finalizer(handle, @bridge).call # must be safe if GC calls it again

    assert handle.freed
  end

  # Host writes race in-flight kernels (on unified-memory backends
  # buffer_write is a raw memcpy). Reads already waited; writes did not.
  def test_write_waits_for_outstanding_kernels
    buf = Tilework::Buffer.new(:float32, 4)
    before = @bridge.synchronize_count
    buf.write([1.0, 2.0, 3.0, 4.0])
    assert_operator @bridge.synchronize_count, :>, before
  end

  def test_write_bytes_waits_for_outstanding_kernels
    buf = Tilework::Buffer.new(:float32, 4)
    before = @bridge.synchronize_count
    buf.write_bytes([1.0, 2.0, 3.0, 4.0].pack("f*"))
    assert_operator @bridge.synchronize_count, :>, before
  end
end
