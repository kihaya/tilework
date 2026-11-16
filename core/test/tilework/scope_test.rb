# frozen_string_literal: true

require_relative "../test_helper"

class ScopeTest < Minitest::Test
  include BridgeTestSetup

  def test_arrays_are_passed_as_buffers
    Tilework.run([1.0, 2.0], [3, 4], 5) do |a, b, c|
      assert_instance_of Tilework::Buffer, a
      assert_equal :float32, a.dtype
      assert_equal :int32, b.dtype
      assert_equal 5, c
      nil
    end
  end

  def test_buffer_result_is_unwrapped_to_array
    assert_equal [1.0, 2.0], Tilework.run { Tilework[1.0, 2.0] }
  end

  def test_nested_and_scalar_results_are_unwrapped
    result = Tilework.run { [Tilework[1.0], [Tilework[2, 3], 4], "x"] }
    assert_equal [[1.0], [[2, 3], 4], "x"], result
  end

  def test_buffers_created_inside_are_freed
    input = intermediate = nil
    Tilework.run([1.0, 2.0]) do |a|
      input = a
      intermediate = a + a
      intermediate
    end
    assert input.freed?
    assert intermediate.freed?
  end

  def test_buffers_created_outside_are_not_freed
    outside = Tilework[1.0, 2.0]
    Tilework.run { outside + outside }
    refute outside.freed?
  end

  def test_buffers_are_freed_on_exception
    captured = nil
    assert_raises(RuntimeError) do
      Tilework.run do
        captured = Tilework[1.0]
        raise "boom"
      end
    end
    assert captured.freed?
  end

  def test_nested_scopes_are_independent
    outer_buf = nil
    Tilework.run do
      outer_buf = Tilework[1.0]
      inner_buf = nil
      Tilework.run { inner_buf = Tilework[2.0] }
      assert inner_buf.freed?
      refute outer_buf.freed?
      nil
    end
    assert outer_buf.freed?
  end

  def test_core_ext_provides_top_level_with_gpu
    require "tilework/core_ext"
    result = with_gpu([1.0, 2.0]) { |a| a }
    assert_equal [1.0, 2.0], result
  end

  def test_nested_array_input_becomes_2d_buffer
    Tilework.run([[1.0, 2.0], [3.0, 4.0]]) do |m|
      assert_instance_of Tilework::Buffer, m
      assert_equal [2, 2], m.shape
      nil
    end
  end

  def test_2d_buffer_result_is_unwrapped_to_nested_array
    result = Tilework.run { Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0, 4.0]]) }
    assert_equal [[1.0, 2.0], [3.0, 4.0]], result
  end
end
