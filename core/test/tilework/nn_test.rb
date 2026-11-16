# frozen_string_literal: true

require_relative "../test_helper"

# Checks GPU attention path selection without hardware. MockBridge does not run
# kernels, but the generated dialect source and dispatch arguments are enough to
# pin that we never fall back to the old path that builds a huge scores Buffer.
class NNTest < Minitest::Test
  include BridgeTestSetup

  def matrix(rows, cols, value = 0.0)
    Tilework::Buffer.new(:float32, rows * cols, shape: [rows, cols]).tap do |buffer|
      buffer.write(Array.new(rows * cols, value))
    end
  end

  def test_large_attention_uses_online_softmax_kernel_without_a_scores_buffer
    q = matrix(64, 8, 0.1)
    k = matrix(64, 8, 0.2)
    v = matrix(64, 8, 0.3)

    out = Tilework::NN.multi_head_attention(q, k, v, heads: 2)

    assert_equal [64, 8], out.shape
    assert_equal 1, @bridge.dispatches.length
    dispatch = @bridge.dispatches.last
    assert_equal "nn_flash_attention", dispatch[:pipeline][:name]
    assert_equal [128, 1, 1], dispatch[:grid]
    assert_equal [256, 1, 1], dispatch[:block]

    src = dispatch[:pipeline][:src]
    assert_includes src, "GPU_SHARED float ktile[2048];"
    assert_includes src, "GPU_SHARED float vtile[2048];"
    assert_includes src, "state[1] = state[1] * state[2] + state[3];"
    refute_includes src, "device float* scores"
  end

  def test_flash_attention_applies_bias_inside_the_online_softmax
    q = matrix(64, 8)
    k = matrix(64, 8)
    v = matrix(64, 8)
    bias = matrix(128, 64, -1.0e9)

    Tilework::NN.multi_head_attention(q, k, v, heads: 2, bias: bias)

    src = @bridge.dispatches.last[:pipeline][:src]
    assert_includes src, "device const float* bias"
    assert_includes src, "score = dot[0] * scale + bias[(head * n + query) * m + key]"
  end

  def test_affine_norms_free_only_their_missing_parameter_fillers
    x = matrix(2, 4)
    gamma = Tilework::Buffer.new(:float32, 4)
    Tilework::NN.layernorm(x, gamma: gamma)
    layer_args = @bridge.dispatches.last[:args]
    refute gamma.freed?
    assert layer_args[3].freed, "欠けた beta の zero buffer は起動後に解放する"

    image = Tilework::Buffer.new(:float32, 8, shape: [1, 2, 2, 2])
    beta = Tilework::Buffer.new(:float32, 2)
    Tilework::NN.groupnorm(image, groups: 1, beta: beta)
    group_args = @bridge.dispatches.last[:args]
    assert group_args[2].freed, "欠けた gamma の one buffer は起動後に解放する"
    refute beta.freed?
  end
end
