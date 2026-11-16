# frozen_string_literal: true

module Tilework
  # IEEE-754 bfloat16 and Ruby Float conversion. BF16 is the upper 16 bits of
  # binary32, rounded to nearest with ties to even.
  module BFloat
    BYTES = 2

    module_function

    def encode(array)
      array.map { |value| from_f(value) }.pack("S<*")
    end

    def decode(bytes)
      bytes.unpack("S<*").map { |bits| to_f(bits) }
    end

    def from_f(value)
      bits = [value.to_f].pack("e").unpack1("L<")
      # Canonicalize NaNs so rounding cannot turn a signaling NaN into Inf.
      return ((bits >> 16) | 0x0040) & 0xFFFF if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0

      ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    end

    def to_f(bits)
      [Integer(bits) << 16].pack("L<").unpack1("e")
    end
  end
end
