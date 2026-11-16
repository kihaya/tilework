# frozen_string_literal: true

require_relative "../test_helper"

# A view is a Buffer that sees the same device memory with another shape/range.
# Properties to keep: it does not allocate, writes reach the parent, and it
# must not outlive the parent.
class BufferViewTest < Minitest::Test
  include BridgeTestSetup

  def setup
    super
    @x = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0, 4.0],
                                  [5.0, 6.0, 7.0, 8.0],
                                  [9.0, 10.0, 11.0, 12.0]])
  end

  def test_view_reads_the_requested_element_range
    assert_equal [5.0, 6.0, 7.0, 8.0], @x.view(4, 4).to_a
  end

  def test_slice_takes_a_range_of_the_outermost_dimension
    v = @x.slice(1..2)

    assert_equal [2, 4], v.shape
    assert_equal [5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], v.to_a
  end

  def test_slice_accepts_a_single_index
    assert_equal [1, 4], @x.slice(1).shape
    assert_equal [5.0, 6.0, 7.0, 8.0], @x.slice(1).to_a
  end

  def test_slice_counts_from_the_end_for_a_negative_index
    assert_equal [9.0, 10.0, 11.0, 12.0], @x.slice(-1).to_a
  end

  def test_slice_works_on_rank_4_buffers
    images = Tilework::Buffer.new(:float32, 2 * 3 * 4 * 4, shape: [2, 3, 4, 4])

    assert_equal [1, 3, 4, 4], images.slice(0).shape
    assert_equal 48, images.slice(0).length
  end

  def test_reshape_keeps_the_elements_and_changes_the_shape
    r = @x.reshape(2, 6)

    assert_equal [2, 6], r.shape
    assert_equal @x.to_a, r.to_a
  end

  def test_reshape_infers_a_single_negative_one
    assert_equal [2, 6], @x.reshape(2, -1).shape
    assert_equal [3, 4], @x.reshape(-1, 4).shape
  end

  # A view does not allocate, so writing through to the parent is the whole
  # point; KV caches and head splitting depend on it.
  def test_writing_through_a_view_is_visible_in_the_base
    @x.slice(1).write([99.0, 98.0, 97.0, 96.0])

    assert_equal [1.0, 2.0, 3.0, 4.0, 99.0, 98.0, 97.0, 96.0, 9.0, 10.0, 11.0, 12.0], @x.to_a
  end

  def test_a_view_of_a_view_is_relative_to_the_outer_view
    assert_equal [7.0, 8.0], @x.slice(1).view(2, 2).to_a
  end

  def test_a_nested_view_reports_the_original_buffer_as_its_base
    assert_same @x, @x.slice(1..2).slice(0).base
  end

  def test_view_predicates
    refute_predicate @x, :view?
    assert_predicate @x.slice(0), :view?
    assert_nil @x.base
  end

  # Freeing a view does not release memory; the parent stays alive.
  def test_freeing_a_view_leaves_the_base_usable
    v = @x.slice(0)
    v.free

    assert_predicate v, :freed?
    assert_equal 12, @x.to_a.length
  end

  def test_view_rejects_an_out_of_bounds_range
    assert_raises(ArgumentError) { @x.view(10, 4) }
    assert_raises(ArgumentError) { @x.slice(1..5) }
    assert_raises(ArgumentError) { @x.slice(3) }
  end

  def test_reshape_rejects_a_shape_with_a_different_element_count
    assert_raises(ArgumentError) { @x.reshape(5, 5) }
    assert_raises(ArgumentError) { @x.reshape(-1, -1) }
  end

  def test_slice_requires_a_shaped_buffer
    assert_raises(Tilework::Error) { Tilework::Buffer.new(:float32, 8).slice(0) }
  end

  def test_viewing_a_freed_buffer_is_rejected
    @x.free

    assert_raises(Tilework::Error) { @x.view(0, 4) }
  end
end
