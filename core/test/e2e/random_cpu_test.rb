# frozen_string_literal: true

require_relative "../test_helper"

# Random generation (core/lib/tilework/math/random.rb) on a backend without a
# native path. Runs on :cpu, so CI covers it without a GPU.
class RandomCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def teardown
    Tilework::Random.seed!(nil)
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  def test_falls_back_when_no_rng_library_is_available
    refute Tilework::Random.native?
    assert_nil Tilework.capabilities[:rng]
    assert_equal 100, Tilework.random(100).to_a.length # still works
  end

  def test_uniform_stays_in_the_documented_range
    # (0, 1] to match cuRAND — excludes 0, includes 1; the opposite ends from Ruby's rand.
    values = Tilework.random(5000, seed: 1).to_a
    assert_operator values.min, :>, 0.0
    assert_operator values.max, :<=, 1.0
  end

  def test_uniform_is_roughly_uniform
    values = Tilework.random(20_000, seed: 2).to_a
    assert_in_delta 0.5, values.sum / values.size, 0.02
  end

  def test_normal_has_the_requested_moments
    values = Tilework.randn(20_000, seed: 3).to_a
    mean = values.sum / values.size
    sd = Math.sqrt(values.sum { |v| (v - mean)**2 } / values.size)
    assert_in_delta 0.0, mean, 0.05
    assert_in_delta 1.0, sd, 0.05
  end

  def test_normal_honours_mean_and_stddev
    values = Tilework.randn(20_000, mean: 5.0, stddev: 2.0, seed: 4).to_a
    mean = values.sum / values.size
    sd = Math.sqrt(values.sum { |v| (v - mean)**2 } / values.size)
    assert_in_delta 5.0, mean, 0.1
    assert_in_delta 2.0, sd, 0.1
  end

  # Same seed, same sequence. Not necessarily across backends (different
  # algorithms), but reproducible within a process.
  def test_the_same_seed_reproduces_the_same_values
    assert_equal Tilework.random(500, seed: 42).to_a, Tilework.random(500, seed: 42).to_a
    assert_equal Tilework.randn(500, seed: 42).to_a, Tilework.randn(500, seed: 42).to_a
  end

  def test_different_seeds_give_different_values
    refute_equal Tilework.random(500, seed: 1).to_a, Tilework.random(500, seed: 2).to_a
  end

  def test_without_a_seed_each_call_differs
    refute_equal Tilework.random(200).to_a, Tilework.random(200).to_a
  end

  # seed! makes the whole process deterministic (for tests and reproductions).
  # The point is resetting the call counter as well as the seed; otherwise the
  # same seed would continue the previous sequence.
  def test_seed_bang_makes_the_whole_process_deterministic
    Tilework::Random.seed!(99)
    first = [Tilework.random(50).to_a, Tilework.random(50).to_a]
    Tilework::Random.seed!(99)
    second = [Tilework.random(50).to_a, Tilework.random(50).to_a]

    assert_equal first, second
    refute_equal first[0], first[1], "consecutive draws must still differ"
  end

  def test_seed_bang_nil_restores_non_deterministic_draws
    Tilework::Random.seed!(nil)
    refute_equal Tilework.random(50).to_a, Tilework.random(50).to_a
  end

  # --- Backends without FFT ---

  # Unlike cuRAND, FFT has no host fallback (a DFT in Ruby is not practical),
  # so the error must make the missing support obvious.
  def test_fft_reports_unavailable_and_raises
    refute Tilework::FFT.available?
    assert_nil Tilework.capabilities[:fft]

    error = assert_raises(Tilework::Error) { Tilework::FFT.forward(Tilework[[1.0, 2.0, 3.0, 4.0]]) }
    assert_match(/FFT/, error.message)
    assert_match(/cuda/i, error.message)
  end

  # Like FFT, convolution has no host fallback
  def test_conv_reports_unavailable_and_raises
    refute Tilework::Conv.available?
    assert_nil Tilework.capabilities[:conv]

    x = Tilework::Buffer.new(:float32, 16, shape: [1, 1, 4, 4])
    w = Tilework::Buffer.new(:float32, 9, shape: [1, 1, 3, 3])
    error = assert_raises(Tilework::Error) { Tilework::Conv.forward(x, w) }
    assert_match(/convolution/i, error.message)
  end

  # --- shape ---

  def test_accepts_a_shape_array
    buf = Tilework.randn([16, 8], seed: 5)
    assert_equal [16, 8], buf.shape
    assert_equal 128, buf.length
  end

  def test_rejects_a_bad_shape
    assert_raises(ArgumentError) { Tilework.random(0) }
    assert_raises(ArgumentError) { Tilework.random([4, -1]) }
    assert_raises(ArgumentError) { Tilework.random("nope") }
  end

  def test_only_float32_is_supported
    error = assert_raises(Tilework::Error) { Tilework::Random.uniform!(Tilework[[1, 2, 3]]) }
    assert_match(/float32/, error.message)
  end

  # An odd length still returns the right length (cuRAND's normal generator
  # works in pairs, so the native side must handle the remainder — this pins
  # that contract)
  def test_odd_lengths_work
    assert_equal 999, Tilework.randn(999, seed: 6).to_a.length
    assert_equal 1, Tilework.randn(1, seed: 7).to_a.length
  end

  # --- Portable path (Random.portable!) ---
  #
  # :cpu does not accept the dialect, so the Ruby Philox implementation fills
  # it. It is the same formula as the GPU dialect kernel, so the values pinned
  # here are the cross-backend expectation (the same numbers as conformance.rb's
  # PORTABLE_UNIFORM_SEED_42).
  def test_portable_uniform_is_the_documented_sequence
    Tilework::Random.portable!
    assert_equal [0.6129599213600159, 0.4685865640640259, 0.0732317566871643].pack("f*").unpack("f*"),
                 Tilework.random(3, seed: 42).to_a
  ensure
    Tilework::Random.native!
  end

  def test_portable_normal_is_the_documented_sequence
    Tilework::Random.portable!
    got = Tilework.randn(2, seed: 42).to_a
    [-0.9701898097991943, -1.2356207370758057].zip(got) { |e, a| assert_in_delta e, a, 1e-6 }
  ensure
    Tilework::Random.native!
  end

  # Counter-based, so taking a sub-range does not shift the sequence.
  def test_portable_values_do_not_depend_on_the_request_size
    Tilework::Random.portable!
    assert_equal Tilework.random(50, seed: 8).to_a.first(4), Tilework.random(4, seed: 8).to_a
  ensure
    Tilework::Random.native!
  end

  def test_portable_normal_has_the_requested_moments
    Tilework::Random.portable!
    values = Tilework.randn(20_000, seed: 3).to_a
    mean = values.sum / values.size
    stddev = Math.sqrt(values.sum { |v| (v - mean)**2 } / values.size)
    assert_in_delta 0.0, mean, 0.03
    assert_in_delta 1.0, stddev, 0.03
  ensure
    Tilework::Random.native!
  end

  # seed! does not reset the mode (a separate axis); teardown restores it explicitly.
  def test_mode_is_independent_of_the_seed
    Tilework::Random.portable!
    Tilework::Random.seed!(1)
    assert Tilework::Random.portable?
  ensure
    Tilework::Random.native!
  end
end
