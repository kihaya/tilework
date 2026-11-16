# frozen_string_literal: true

require_relative "../test_helper"

# Portable RNG (Philox 4x32-10). It underpins the promise of the same sequence
# on every backend, so this pins **the values themselves**, not statistics.
class PhiloxTest < Minitest::Test
  include BridgeTestSetup

  def teardown
    Tilework::Random.native!
    Tilework::Random.seed!(nil)
    super if defined?(super)
  end

  # Random123's known-answer test. As long as this holds, the implementation
  # computes the standard philox4x32-10.
  def test_bijection_matches_the_reference_known_answer
    assert_equal %w[6627e8d5 e169c58d bc57ac4c 9b00dbd8],
                 Tilework::Philox.bijection(0, 0, 0).map { |x| format("%08x", x) }
  end

  # The high 32 bits are built from 16-bit halves; check that part on its own.
  def test_mulhilo_agrees_with_64bit_multiplication
    [[0, 0], [1, 1], [0xFFFFFFFF, 0xFFFFFFFF], [0xD2511F53, 0x12345678], [0x80000000, 2]].each do |a, b|
      product = a * b
      assert_equal [product >> 32, product & 0xFFFFFFFF], Tilework::Philox.mulhilo(a, b), "#{a} * #{b}"
    end
  end

  def test_uniform_lies_in_the_documented_half_open_range
    values = Array.new(2000) { |i| Tilework::Philox.uniform_at(i, 99, 0) }
    assert_operator values.min, :>, 0.0
    assert_operator values.max, :<=, 1.0
  end

  # Only 24 bits and powers of two are used, so rounding to float changes
  # nothing. That is what the bit-identity guarantee rests on.
  def test_uniform_values_survive_the_round_trip_through_float32
    values = Array.new(500) { |i| Tilework::Philox.uniform_at(i, 5, 7) }
    assert_equal values, values.pack("f*").unpack("f*")
  end

  # The point of counter-based: changing the request size does not move value i.
  def test_values_do_not_depend_on_the_request_size
    long = Tilework::Philox.host_values(64, Tilework::Random::UNIFORM, 3, 0, 0.0, 1.0)
    short = Tilework::Philox.host_values(5, Tilework::Random::UNIFORM, 3, 0, 0.0, 1.0)
    assert_equal long.first(5), short
  end

  def test_normal_applies_mean_and_stddev
    plain = Tilework::Philox.normal_at(0, 1, 0, 0.0, 1.0)
    shifted = Tilework::Philox.normal_at(0, 1, 0, 10.0, 2.0)
    assert_in_delta 10.0 + (2.0 * plain), shifted, 1e-12
  end

  # --- Dialect kernel ---
  #
  # Values need hardware, but CI can still guarantee that no raw CUDA / MSL is
  # written and that the source expands under every backend's header.

  def with_backend(name)
    Tilework.instance_variable_set(:@backend, name)
    yield
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  def test_kernel_source_is_written_in_the_dialect
    src = Tilework::Philox.kernel_source(Tilework::Random::UNIFORM)
    assert_includes src, "GPU_KERNEL void philox_fill"
    assert_includes src, "GPU_UINT"
    assert_includes src, "GPU_FUNC"
    refute_includes src, "__global__"
    refute_includes src, "unsigned int"
    # Pointers with address-space qualifiers are not in the dialect's macro set
    refute_includes src, "thread "
  end

  def test_both_compat_headers_define_the_macros_the_kernel_needs
    %i[metal cuda].each do |backend|
      header = with_backend(backend) { Tilework::Dialect.compat_header }
      assert_includes header, "#define GPU_UINT", "#{backend} must define GPU_UINT"
      assert_includes header, "#define GPU_FUNC", "#{backend} must define GPU_FUNC"
    end
  end

  def test_cuda_rewrite_leaves_a_valid_signature
    src = Tilework::Philox.kernel_source(Tilework::Random::NORMAL)
    rewritten = with_backend(:cuda) { Tilework::Dialect.rewrite(src) }
    assert_includes rewritten, "void philox_fill(float* out"
    assert_includes rewritten, "const int n" # constant int& -> by value
  end

  # --- Mode switching on the Random side ---

  def test_mode_defaults_to_native_and_round_trips
    assert_equal :native, Tilework::Random.mode
    refute Tilework::Random.portable?
    Tilework::Random.portable!
    assert Tilework::Random.portable?
    Tilework::Random.native!
    assert_equal :native, Tilework::Random.mode
  end

  def test_unknown_mode_is_rejected
    error = assert_raises(ArgumentError) { Tilework::Random.mode = :fast }
    assert_match(/unknown random mode/, error.message)
  end

  # On a dialect backend (MockBridge is :custom, which accepts the dialect) it
  # launches a kernel instead of generating on the host.
  def test_portable_mode_launches_the_kernel_on_a_dialect_backend
    Tilework::Random.portable!
    Tilework.random(16, seed: 42)
    assert_equal 1, @bridge.compilations.length
    assert_includes @bridge.compilations.first[:src], "philox_fill"
    assert_equal 1, @bridge.dispatches.length
  end
end
