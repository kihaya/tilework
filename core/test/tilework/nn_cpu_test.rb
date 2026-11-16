# frozen_string_literal: true

require_relative "../test_helper"

# Tilework::NN on :cpu. The central "same numbers" checks live in
# Tilework::Conformance (lib/tilework/conformance.rb, run on :cpu by
# test/tilework/conformance_test.rb); what remains here are auxiliary
# numerical properties and argument validation (ArgumentError).
class NNCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def matrix(rows, cols, values)
    buf = Tilework::Buffer.new(:float32, rows * cols, shape: [rows, cols])
    buf.write(values)
    buf
  end

  def assert_close(expected, actual, tol = 1e-6)
    assert_equal expected.length, actual.length
    expected.zip(actual).each_with_index { |(e, a), i| assert_in_delta e, a, tol, "element #{i}" }
  end

  # --- softmax ---

  def test_softmax_rows_sum_to_one
    out = Tilework::NN.softmax(matrix(2, 3, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])).to_a

    assert_in_delta 1.0, out[0, 3].sum, 1e-6
    assert_in_delta 1.0, out[3, 3].sum, 1e-6
    assert_close [1 / 3.0, 1 / 3.0, 1 / 3.0], out[3, 3]
  end

  def test_softmax_is_shift_invariant
    a = Tilework::NN.softmax(matrix(1, 3, [1.0, 2.0, 3.0])).to_a
    b = Tilework::NN.softmax(matrix(1, 3, [101.0, 102.0, 103.0])).to_a

    assert_close a, b, 1e-6
  end

  # --- layernorm ---

  def test_layernorm_rejects_a_mismatched_affine_length
    x = matrix(1, 4, [1.0, 2.0, 3.0, 4.0])
    gamma = Tilework::Buffer.new(:float32, 3)

    assert_raises(ArgumentError) { Tilework::NN.layernorm(x, gamma: gamma) }
  end

  # --- groupnorm ---

  def test_groupnorm_with_one_group_matches_layernorm_over_the_whole_image
    x = Tilework::Buffer.new(:float32, 8, shape: [1, 2, 2, 2])
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x.write(values)
    flat = Tilework::Buffer.new(:float32, 8, shape: [1, 8])
    flat.write(values)

    assert_close Tilework::NN.layernorm(flat).to_a, Tilework::NN.groupnorm(x, groups: 1).to_a, 1e-5
  end

  def test_groupnorm_rejects_groups_that_do_not_divide_the_channels
    x = Tilework::Buffer.new(:float32, 12, shape: [1, 3, 2, 2])

    assert_raises(ArgumentError) { Tilework::NN.groupnorm(x, groups: 2) }
  end

  # --- Activations ---

  # A naive 1/(1+exp(-x)) overflows exp when x is very negative.
  def test_sigmoid_survives_large_negative_inputs
    x = Tilework::Buffer.new(:float32, 2)
    x.write([-800.0, 800.0])
    out = Tilework::NN.sigmoid(x).to_a

    assert(out.none? { |v| v.nan? }, "sigmoid overflowed to NaN")
    assert_in_delta 0.0, out[0], 1e-6
    assert_in_delta 1.0, out[1], 1e-6
  end

  # --- transpose ---

  def test_transpose_twice_is_the_identity
    x = matrix(3, 5, (0...15).map(&:to_f))

    assert_equal x.to_a, Tilework::NN.transpose(Tilework::NN.transpose(x)).to_a
  end

  # --- multi_head_attention ---

  # The T5 relative-position-bias path: the bias is added before softmax, and
  # keys set to -1e9 drop out of the result.
  def test_multi_head_attention_bias_is_added_before_the_softmax
    heads, dh, n, m = 2, 3, 4, 5
    d = heads * dh
    q = matrix(n, d, (0...(n * d)).map { |i| Math.sin(i * 0.7) })
    k = matrix(m, d, (0...(m * d)).map { |i| Math.cos(i * 0.3) })
    # Each key gets its own v, so the output shows which keys contributed
    v = matrix(m, d, (0...(m * d)).map { |i| ((i / d) + 1).to_f })

    # Mask the last two keys
    bias_values = Array.new(heads * n * m, 0.0)
    heads.times do |h|
      n.times { |r| (3...m).each { |c| bias_values[(((h * n) + r) * m) + c] = -1.0e9 } }
    end
    bias = matrix(heads * n, m, bias_values)

    got = Tilework::NN.multi_head_attention(q, k, v, heads: heads, bias: bias).to_a
    # The remaining keys have v = 1, 2, 3, so a convex combination stays within 1..3
    assert got.all? { |x| x >= 1.0 - 1e-4 && x <= 3.0 + 1e-4 },
           "隠したキー (v=4,5) が結果に混ざっている: #{got.minmax.inspect}"
  end

  def test_multi_head_attention_rejects_a_wrongly_shaped_bias
    q = matrix(4, 6, Array.new(24, 0.0))
    k = matrix(5, 6, Array.new(30, 0.0))
    v = matrix(5, 6, Array.new(30, 0.0))
    bad = matrix(4, 5, Array.new(20, 0.0)) # [n, m], not [heads*n, m]

    assert_raises(ArgumentError) do
      Tilework::NN.multi_head_attention(q, k, v, heads: 2, bias: bad)
    end
  end

  # heads must divide d. Letting it through would produce plausible-looking
  # values with misaligned strides.
  def test_multi_head_attention_rejects_shapes_that_do_not_line_up
    q = matrix(4, 6, Array.new(24, 0.0))
    k = matrix(5, 6, Array.new(30, 0.0))
    v = matrix(5, 6, Array.new(30, 0.0))

    assert_raises(ArgumentError) { Tilework::NN.multi_head_attention(q, k, v, heads: 4) }
    # k and v disagree on the number of rows
    assert_raises(ArgumentError) do
      Tilework::NN.multi_head_attention(q, k, matrix(4, 6, Array.new(24, 0.0)), heads: 2)
    end
    # width differs from q
    assert_raises(ArgumentError) do
      Tilework::NN.multi_head_attention(q, matrix(5, 4, Array.new(20, 0.0)), v, heads: 2)
    end
  end

  # --- rmsnorm ---

  def test_rmsnorm_applies_gamma
    x = matrix(1, 4, [1.0, -2.0, 3.0, -4.0])
    g = Tilework::Buffer.new(:float32, 4); g.write([2.0, 0.5, 1.0, -1.0])

    plain = Tilework::NN.rmsnorm(x, eps: 1e-6).to_a
    scaled = Tilework::NN.rmsnorm(x, gamma: g, eps: 1e-6).to_a

    assert_close plain.each_with_index.map { |v, i| v * [2.0, 0.5, 1.0, -1.0][i] }, scaled, 1e-5
  end

  # Loss argument validation. Conformance checks the values; this only checks
  # that a wrong call does not pass silently.
  def test_losses_reject_mismatched_arguments
    pred = matrix(2, 3, [0.0] * 6)
    labels = Tilework::Buffer.new(:int32, 2, shape: [2])
    labels.write([0, 1])

    assert_raises(ArgumentError) { Tilework::NN.mse(pred, matrix(3, 2, [0.0] * 6)) }
    assert_raises(ArgumentError) { Tilework::NN.mse(pred, pred, reduction: :average) }
    # Labels are :int32 class indices, not a probability (one-hot) shape.
    assert_raises(ArgumentError) { Tilework::NN.cross_entropy(pred, pred) }
    assert_raises(ArgumentError) do
      Tilework::NN.cross_entropy(pred, Tilework::Buffer.new(:int32, 5, shape: [5]))
    end
    assert_raises(ArgumentError) { Tilework::NN.cross_entropy(Tilework::Buffer.new(:float32, 6), labels) }
  end

  # Out-of-range labels are clamped to the ends (the same rule as gather). An
  # out-of-range read on a GPU is undefined, so clamp rather than break silently.
  def test_cross_entropy_clamps_out_of_range_labels
    logits = matrix(2, 3, [0.1, 0.5, -0.3, 0.1, 0.5, -0.3])
    labels = Tilework::Buffer.new(:int32, 2, shape: [2])
    labels.write([-1, 7])
    edges = matrix(2, 3, [0.1, 0.5, -0.3, 0.1, 0.5, -0.3])
    want = Tilework::Buffer.new(:int32, 2, shape: [2])
    want.write([0, 2])

    assert_close Tilework::NN.cross_entropy(edges, want, reduction: :none).to_a,
                 Tilework::NN.cross_entropy(logits, labels, reduction: :none).to_a, 1e-6
  end

  def test_operations_reject_wrongly_shaped_buffers
    flat = Tilework::Buffer.new(:float32, 6)

    assert_raises(ArgumentError) { Tilework::NN.softmax(flat) }
    assert_raises(ArgumentError) { Tilework::NN.layernorm(flat) }
    assert_raises(ArgumentError) { Tilework::NN.transpose(flat) }
    assert_raises(ArgumentError) { Tilework::NN.groupnorm(flat, groups: 1) }
  end
end
