# frozen_string_literal: true

module Tilework
  # Portable random number generator (Philox 4x32-10).
  #
  # Native RNGs (cuRAND / MPSMatrixRandomPhilox) may both be called "Philox",
  # but counter/key layout and the normal transform are implementation-
  # specific, so **the same seed gives different sequences across backends**.
  # This is the single definition shared by every backend that closes that gap.
  #
  # It uses only plain integer arithmetic, so **the same formula** can be
  # written both as a dialect kernel (within the compat macros) and in plain
  # Ruby. Being counter-based it is deterministic in parallel — element i
  # depends on no other element, and changing the request size does not move
  # the value at position i.
  #
  # The high half of a 32x32 -> 64-bit multiply is built from 16-bit halves
  # (so it works with today's int32-only dtype set; this asks less of a backend
  # than adding 64-bit integers to the compat macros).
  #
  # ## How closely it matches
  #
  # - `uniform` is **bit-identical**. Integer-to-float conversion uses only 24
  #   bits plus powers of two, so neither GPU float nor Ruby double arithmetic
  #   rounds.
  # - `normal` goes through Box-Muller's sqrt/log/cos and matches within their
  #   precision (about 1e-6 relative). Bit identity is not guaranteed — the last
  #   ulp of transcendentals depends on the backend's math library.
  module Philox
    ROUNDS = 10
    M0 = 0xD2511F53
    M1 = 0xCD9E8D57
    W0 = 0x9E3779B9 # golden ratio
    W1 = 0xBB67AE85 # sqrt(3) - 1
    MASK = 0xFFFFFFFF

    module_function

    # Four 32-bit randoms from counter n / key (the low and high 32 bits of the
    # seed). Keep this 1:1 with the expression kernel_source generates below.
    def bijection(counter, key_lo, key_hi)
      c0 = counter & MASK
      c1 = 0
      c2 = 0
      c3 = 0
      k0 = key_lo & MASK
      k1 = key_hi & MASK

      ROUNDS.times do
        hi0, lo0 = mulhilo(M0, c0)
        hi1, lo1 = mulhilo(M1, c2)
        c0, c1, c2, c3 = (hi1 ^ c1 ^ k0) & MASK, lo1, (hi0 ^ c3 ^ k1) & MASK, lo0
        k0 = (k0 + W0) & MASK
        k1 = (k1 + W1) & MASK
      end
      [c0, c1, c2, c3]
    end

    # 32x32 -> [high 32 bits, low 32 bits]. Long multiplication in 16-bit
    # halves gets the high half using only 32-bit products.
    def mulhilo(a, b)
      a0 = a & 0xFFFF
      a1 = (a >> 16) & 0xFFFF
      b0 = b & 0xFFFF
      b1 = (b >> 16) & 0xFFFF

      t = (a0 * b0) & MASK
      k = (t >> 16) & 0xFFFF
      t = ((a1 * b0) + k) & MASK
      w1 = t & 0xFFFF
      w2 = (t >> 16) & 0xFFFF
      t = ((a0 * b1) + w1) & MASK
      hi = ((a1 * b1) + w2 + ((t >> 16) & 0xFFFF)) & MASK
      [hi, (a * b) & MASK]
    end

    # 32-bit random -> (0, 1]. Only the top 24 bits are used, so neither float
    # nor double rounds and GPU and Ruby agree bit for bit.
    def to_unit(x) = (((x >> 8) & 0xFFFFFF) + 1) * (1.0 / (1 << 24))

    # Uniform element i. One bijection serves four elements.
    def uniform_at(index, key_lo, key_hi)
      to_unit(bijection(index / 4, key_lo, key_hi)[index % 4])
    end

    # Normal element i. Box-Muller makes one value from two uniforms, so one
    # bijection serves two elements.
    def normal_at(index, key_lo, key_hi, mean, stddev)
      out = bijection(index / 2, key_lo, key_hi)
      pair = (index % 2) * 2
      u1 = to_unit(out[pair])
      u2 = to_unit(out[pair + 1])
      mean + (stddev * Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2 * Math::PI * u2))
    end

    def host_values(length, kind, key_lo, key_hi, mean, stddev)
      if kind == Random::UNIFORM
        Array.new(length) { |i| uniform_at(i, key_lo, key_hi) }
      else
        Array.new(length) { |i| normal_at(i, key_lo, key_hi, mean, stddev) }
      end
    end

    # seed (64-bit) -> two 32-bit kernel arguments. Dispatch packs scalars as
    # signed 32-bit (kernel.rb#convert_args), so convert to values that fit and
    # reinterpret them as GPU_UINT in the kernel.
    def key_parts(seed)
      seed &= 0xFFFFFFFFFFFFFFFF
      [seed & MASK, (seed >> 32) & MASK]
    end

    def signed32(value) = value >= 0x80000000 ? value - 0x100000000 : value

    # The dialect kernel; its expressions are 1:1 with the Ruby implementation above.
    #
    # Helpers exchange scalars only — MSL address-space qualifiers
    # (`thread T*`) are not in the dialect's macro set and have no CUDA
    # counterpart, so the four result words stay in locals c0..c3 in the kernel body.
    def kernel_source(kind)
      uniform = kind == Random::UNIFORM
      <<~KERNEL
        #{HELPERS}
        GPU_KERNEL void philox_fill(GPU_ARGS,
                                    device float* out,
                                    constant int& key_lo,
                                    constant int& key_hi,
                                    constant float& mean,
                                    constant float& stddev,
                                    constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) {
            GPU_UINT k0 = (GPU_UINT)key_lo;
            GPU_UINT k1 = (GPU_UINT)key_hi;
            GPU_UINT c0 = (GPU_UINT)(i / #{uniform ? 4 : 2}), c1 = 0u, c2 = 0u, c3 = 0u;
        #{ROUND_LOOP}
        #{uniform ? UNIFORM_BODY : NORMAL_BODY}
          }
        }
      KERNEL
    end

    HELPERS = <<~KERNEL
      // High half of 32x32 -> 64 bits from 16-bit halves (same formula as philox.rb#mulhilo)
      GPU_FUNC GPU_UINT philox_mulhi(GPU_UINT a, GPU_UINT b) {
        GPU_UINT a0 = a & 0xFFFFu, a1 = a >> 16;
        GPU_UINT b0 = b & 0xFFFFu, b1 = b >> 16;
        GPU_UINT t = a0 * b0;
        GPU_UINT k = t >> 16;
        t = a1 * b0 + k;
        GPU_UINT w1 = t & 0xFFFFu;
        GPU_UINT w2 = t >> 16;
        t = a0 * b1 + w1;
        return a1 * b1 + w2 + (t >> 16);
      }

      // 32-bit random -> (0, 1]. Only the top 24 bits are used, so float does not round.
      GPU_FUNC float philox_unit(GPU_UINT x) {
        return (float)((x >> 8) + 1u) * (1.0f / 16777216.0f);
      }
    KERNEL

    # Philox 4x32-10's ten rounds (same formula as philox.rb#bijection).
    ROUND_LOOP = <<~KERNEL
          for (int r = 0; r < #{ROUNDS}; r++) {
            GPU_UINT hi0 = philox_mulhi(#{format('0x%08Xu', M0)}, c0);
            GPU_UINT lo0 = #{format('0x%08Xu', M0)} * c0;
            GPU_UINT hi1 = philox_mulhi(#{format('0x%08Xu', M1)}, c2);
            GPU_UINT lo1 = #{format('0x%08Xu', M1)} * c2;
            GPU_UINT n0 = hi1 ^ c1 ^ k0;
            GPU_UINT n2 = hi0 ^ c3 ^ k1;
            c0 = n0; c1 = lo1; c2 = n2; c3 = lo0;
            k0 = k0 + #{format('0x%08Xu', W0)};
            k1 = k1 + #{format('0x%08Xu', W1)};
          }
    KERNEL

    UNIFORM_BODY = <<~KERNEL
          int j = i % 4;
          GPU_UINT w = j == 0 ? c0 : (j == 1 ? c1 : (j == 2 ? c2 : c3));
          out[i] = philox_unit(w);
    KERNEL

    # Box-Muller makes one value from two uniforms, so one call serves two
    # elements (even ones use c0/c1, odd ones c2/c3 — the same assignment as
    # philox.rb#normal_at).
    NORMAL_BODY = <<~KERNEL
          GPU_UINT wa = (i % 2) == 0 ? c0 : c2;
          GPU_UINT wb = (i % 2) == 0 ? c1 : c3;
          float u1 = philox_unit(wa);
          float u2 = philox_unit(wb);
          out[i] = mean + stddev * sqrt(-2.0f * log(u1)) * cos(6.28318530717958648f * u2);
    KERNEL
  end
end
