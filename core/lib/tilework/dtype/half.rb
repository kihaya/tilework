# frozen_string_literal: true

module Tilework
  # Conversion between IEEE-754 binary16 (half precision) and Ruby Float.
  #
  # Ruby's Array#pack has no half-precision directive (f/e/g are 4 bytes,
  # d/E/G are 8), so Buffer's "one pack directive per dtype" design does not
  # work for float16. This is the only implementation of the conversion and is
  # referenced from Buffer::DTYPES as encode/decode.
  #
  # **This is the reference implementation of the conversion** and runs without
  # a GPU, so CI verifies it. Large arrays are converted by Buffer#cast with the
  # GPU's __float2half / __half2float; both round to nearest even, so **the two
  # must agree bit for bit**.
  module Half
    BYTES = 2

    # binary16: 1 sign bit / 5 exponent bits (bias 15) / 10 mantissa bits
    MANT_BITS = 10
    EXP_BIAS = 15
    # binary32: 8 exponent bits (bias 127) / 23 mantissa bits
    F32_MANT_BITS = 23
    F32_EXP_BIAS = 127

    MAX_FINITE = 65_504.0 # largest finite half value
    MIN_NORMAL = 2.0**-14 # smallest normal (about 6.104e-5)
    MIN_SUBNORMAL = 2.0**-24 # smallest subnormal (about 5.96e-8)

    module_function

    # Array<Numeric> -> String (2 bytes per element, little-endian)
    def encode(array)
      array.map { |v| from_f(v) }.pack("S<*")
    end

    # String -> Array<Float>
    def decode(bytes)
      bytes.unpack("S<*").map { |bits| to_f(bits) }
    end

    # Float -> binary16 bit pattern (Integer), rounding to nearest even.
    def from_f(value)
      # "e" / "L<" pin the endianness (independent of the host)
      bits = [value.to_f].pack("e").unpack1("L<")
      sign = (bits >> 31) << 15
      exp = (bits >> F32_MANT_BITS) & 0xFF
      mant = bits & ((1 << F32_MANT_BITS) - 1)

      # Inf / NaN. NaN keeps a nonzero mantissa (zero would turn it into Inf)
      return sign | 0x7C00 | (mant.zero? ? 0 : 0x0200) if exp == 0xFF

      new_exp = exp - F32_EXP_BIAS + EXP_BIAS # rebias to binary16

      if new_exp >= 0x1F
        sign | 0x7C00 # overflow -> Inf
      elsif new_exp > 0
        # Normal. Round the mantissa from 23 to 10 bits. If rounding up
        # overflows it to 1<<10, the addition carries into the exponent (and
        # reaching 0x1F there correctly gives Inf).
        sign | ((new_exp << MANT_BITS) + round_shift(mant, F32_MANT_BITS - MANT_BITS))
      elsif new_exp > -MANT_BITS - 1
        # Subnormal. Restore the implicit 1 and round mant16 = full >> (14 - new_exp).
        # Rounding up to 1<<10 yields the smallest normal as is.
        full = mant | (1 << F32_MANT_BITS)
        sign | round_shift(full, F32_MANT_BITS - MANT_BITS + 1 - new_exp)
      else
        sign # too small -> signed zero
      end
    end

    # binary16 bit pattern -> Float
    def to_f(bits)
      sign = (bits >> 15).zero? ? 1.0 : -1.0
      exp = (bits >> MANT_BITS) & 0x1F
      mant = bits & ((1 << MANT_BITS) - 1)

      case exp
      when 0x1F then mant.zero? ? sign * Float::INFINITY : Float::NAN
      when 0    then sign * mant * MIN_SUBNORMAL # subnormal (and ±0)
      else           sign * (1.0 + mant.fdiv(1 << MANT_BITS)) * 2.0**(exp - EXP_BIAS)
      end
    end

    # Drops the low `shift` bits with round-to-nearest-even. Ties go to the
    # even side; that is the "-even" part.
    def round_shift(mant, shift)
      return 0 if shift > F32_MANT_BITS + 1 # every bit is dropped

      dropped = mant & ((1 << shift) - 1)
      kept = mant >> shift
      halfway = 1 << (shift - 1)

      kept += 1 if dropped > halfway || (dropped == halfway && kept.odd?)
      kept
    end
    private_class_method :round_shift
  end
end
