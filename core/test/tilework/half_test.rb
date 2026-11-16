# frozen_string_literal: true

require_relative "../test_helper"

# IEEE-754 binary16 conversion (core/lib/tilework/dtype/half.rb). Needs no GPU,
# so it always runs in CI.
#
# This is the reference implementation of the conversion; the GPU's
# __float2half is expected to match it bit for bit.
class HalfTest < Minitest::Test
  H = Tilework::Half

  # Known bit patterns, written as raw 16 bits so they can be checked against
  # the spec. An Array rather than a Hash because 0.0 and -0.0 collide as Hash
  # keys (Float#eql? is true).
  KNOWN = [
    [0.0, 0x0000],
    [-0.0, 0x8000],
    [1.0, 0x3C00],
    [-1.0, 0xBC00],
    [2.0, 0x4000],
    [3.0, 0x4200],
    [0.5, 0x3800],
    [0.1, 0x2E66],                    # not exactly representable in half (see below)
    [65_504.0, 0x7BFF],               # largest finite value
    [-65_504.0, 0xFBFF],
    [6.103515625e-05, 0x0400],        # smallest normal 2^-14
    [5.960464477539063e-08, 0x0001],  # smallest subnormal 2^-24
    [Float::INFINITY, 0x7C00],
    [-Float::INFINITY, 0xFC00],
  ].freeze

  # Only values exactly representable in half; only for these can decode be
  # expected to return the original Float (0.1 rounds to 0.0999755859375).
  EXACT = [0.0, -0.0, 1.0, -1.0, 2.0, 3.0, 0.5, 65_504.0, -65_504.0,
           6.103515625e-05, 5.960464477539063e-08,
           Float::INFINITY, -Float::INFINITY].freeze

  def test_known_bit_patterns
    KNOWN.each do |value, bits|
      assert_equal bits, H.from_f(value), format("from_f(%p) should be 0x%04X", value, bits)
    end
  end

  def test_exactly_representable_values_round_trip_unchanged
    EXACT.each do |value|
      assert_equal value, H.to_f(H.from_f(value)), format("%p must survive the round trip", value)
    end
  end

  # Unrepresentable values round to the nearest half: 0.1 -> 0.0999755859375 (relative 2.4e-4).
  def test_inexact_values_round_to_the_nearest_representable
    assert_in_delta 0.1, H.to_f(H.from_f(0.1)), 0.1 * 2.0**-11
    refute_equal 0.1, H.to_f(H.from_f(0.1)), "0.1 is not representable in binary16"
  end

  def test_signed_zero_is_preserved
    # -0.0 and 0.0 are equal under ==, so compare the sign of 1/x
    assert_equal(-Float::INFINITY, 1 / H.to_f(H.from_f(-0.0)))
    assert_equal Float::INFINITY, 1 / H.to_f(H.from_f(0.0))
  end

  def test_nan_survives_the_round_trip
    assert_predicate H.to_f(H.from_f(Float::NAN)), :nan?
  end

  # Above the half range is Inf; below it is a signed zero
  def test_overflow_becomes_infinity
    assert_equal Float::INFINITY, H.to_f(H.from_f(70_000.0))
    assert_equal(-Float::INFINITY, H.to_f(H.from_f(-70_000.0)))
  end

  def test_underflow_becomes_signed_zero
    assert_equal 0.0, H.to_f(H.from_f(1e-10))
    assert_equal(-Float::INFINITY, 1 / H.to_f(H.from_f(-1e-10)))
  end

  # Round to nearest even: from 2048 the step is 2, so 2049 sits exactly
  # between 2048 and 2050 and rounds to the even side (2048).
  def test_rounds_half_to_even
    assert_equal 2048.0, H.to_f(H.from_f(2049.0)), "exact tie should round down to even"
    assert_equal 2052.0, H.to_f(H.from_f(2051.0)), "exact tie should round up to even"
    assert_equal 2050.0, H.to_f(H.from_f(2050.0)), "representable value must not move"
  end

  def test_subnormals_are_monotonic
    step = 5.960464477539063e-08
    values = (1..2000).map { |i| i * step * 0.7 }
    decoded = H.decode(H.encode(values))
    assert decoded.all?(&:finite?)
    assert decoded.each_cons(2).all? { |a, b| a <= b }, "subnormal encoding must stay monotonic"
  end

  # --- Array API ---

  def test_encode_produces_two_bytes_per_element
    assert_equal 2 * 3, H.encode([1.0, 2.0, 3.0]).bytesize
  end

  def test_round_trip_stays_within_half_precision_step
    # Normal range only. Subnormals have fixed absolute precision, so relative
    # error grows without bound toward 0 and is not a useful measure.
    values = 5000.times.map { |i| Math.sin(i) * 1000 }
                  .select { |v| v.abs >= Tilework::Half::MIN_NORMAL }
    refute_empty values
    decoded = H.decode(H.encode(values))
    worst = values.zip(decoded).map { |a, b| ((a - b) / a).abs }.max
    # Half has a 10-bit mantissa, so the relative error is at most 2^-11
    assert_operator worst, :<=, 2.0**-11
  end

  # Subnormals are checked with absolute precision (the step is always 2^-24).
  #
  # The bound is one step rather than half a step because building the
  # expected (i+1) * MIN_SUBNORMAL * 0.7 in double already rounds, and an
  # element landing exactly on a midpoint can exceed half a step by 1 ULP.
  # That is error in the test's construction, not the conversion, so the
  # bound gets some slack.
  def test_subnormal_round_trip_is_within_one_step
    values = 500.times.map { |i| (i + 1) * Tilework::Half::MIN_SUBNORMAL * 0.7 }
    decoded = H.decode(H.encode(values))
    worst = values.zip(decoded).map { |a, b| (a - b).abs }.max
    assert_operator worst, :<=, Tilework::Half::MIN_SUBNORMAL
  end

  def test_encode_accepts_integers
    assert_equal H.encode([1.0, 2.0]), H.encode([1, 2])
  end

  def test_decode_is_the_inverse_of_encode_for_representable_values
    exact = [0.0, 0.5, 1.0, -1.0, 2.0, 1024.0, 65_504.0]
    assert_equal exact, H.decode(H.encode(exact))
  end
end
