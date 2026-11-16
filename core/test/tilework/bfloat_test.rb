# frozen_string_literal: true

require_relative "../test_helper"

class BFloatTest < Minitest::Test
  BF = Tilework::BFloat

  def test_known_ieee_bit_patterns
    [[0.0, 0x0000], [-0.0, 0x8000], [1.0, 0x3F80], [-1.0, 0xBF80],
     [Float::INFINITY, 0x7F80], [-Float::INFINITY, 0xFF80]].each do |value, bits|
      assert_equal bits, BF.from_f(value)
    end
  end

  def test_rounds_to_nearest_even
    lower = BF.to_f(0x3F80)
    upper = BF.to_f(0x3F81)
    assert_equal 0x3F80, BF.from_f((lower + upper) / 2.0)

    lower = BF.to_f(0x3F81)
    upper = BF.to_f(0x3F82)
    assert_equal 0x3F82, BF.from_f((lower + upper) / 2.0)
  end

  def test_nan_signed_zero_and_array_round_trip
    assert_predicate BF.to_f(BF.from_f(Float::NAN)), :nan?
    assert_equal(-Float::INFINITY, 1.0 / BF.to_f(BF.from_f(-0.0)))
    values = [0.0, 1.0, -2.5, 3.25]
    assert_equal values, BF.decode(BF.encode(values))
    assert_equal values.length * 2, BF.encode(values).bytesize
  end
end
