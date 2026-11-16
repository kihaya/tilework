# frozen_string_literal: true

require_relative "../test_helper"

class CapabilitiesTest < Minitest::Test
  Capabilities = Tilework::Capabilities

  def test_build_fills_every_field_with_its_default
    caps = Capabilities.build
    assert_equal Capabilities::NAMES.sort, caps.to_h.keys.sort
    refute caps[:tensor_core]
    assert caps[:dialect]
    assert_equal %i[float32 int32], caps[:dtypes]
    assert_nil caps[:warp_size]
  end

  # The reason for leaving plain Hashes: a typo must not read as "unsupported".
  def test_unknown_key_raises_instead_of_returning_nil
    caps = Capabilities.build
    error = assert_raises(KeyError) { caps[:tensro_core] }
    assert_match(/unknown capability :tensro_core/, error.message)
  end

  def test_build_rejects_an_unknown_key
    error = assert_raises(ArgumentError) { Capabilities.build(tensro_core: true) }
    assert_match(/unknown capability :tensro_core/, error.message)
  end

  def test_build_rejects_a_value_of_the_wrong_type
    error = assert_raises(ArgumentError) { Capabilities.build(warp_size: "32") }
    assert_match(/expects integer_or_nil/, error.message)
    assert_raises(ArgumentError) { Capabilities.build(tensor_core: nil) }
  end

  def test_reader_methods_agree_with_bracket_access
    caps = Capabilities.build(warp_size: 32, gemm_library: :cublas)
    assert_equal caps[:warp_size], caps.warp_size
    assert_equal :cublas, caps.gemm_library
  end

  def test_merge_validates_like_build_and_leaves_the_receiver_alone
    caps = Capabilities.build
    merged = caps.merge(warp_size: 32)
    assert_equal 32, merged[:warp_size]
    assert_nil caps[:warp_size]
    assert_raises(ArgumentError) { caps.merge(warp_siz: 32) }
  end

  # Takes several Hashes like Hash#merge; the :metal branch in tilework.rb
  # calls it that way.
  def test_merge_takes_several_hashes_like_hash_merge
    caps = Capabilities.build.merge({ warp_size: 32 }, { unified_memory: true, warp_size: 64 })
    assert_equal 64, caps[:warp_size] # last one wins
    assert caps[:unified_memory]
  end

  # A bridge's optional device_capabilities does not know the contract's key
  # set, so unknown keys must be droppable before they reach build.
  def test_slice_known_drops_keys_outside_the_contract
    sliced = Capabilities.slice_known(warp_size: 32, sm_count: 16)
    assert_equal({ warp_size: 32 }, sliced)
  end

  def test_every_field_documents_itself
    Capabilities::FIELDS.each do |field|
      refute_empty field.doc.to_s, "#{field.name} has no doc"
    end
  end
end
