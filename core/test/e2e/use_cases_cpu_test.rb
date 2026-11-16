# frozen_string_literal: true

require_relative "../test_helper"

# Small end-to-end use cases on the :cpu backend. No GPU is needed, so this
# always runs in CI (.github/workflows/ci.yml) without the GPU_TESTS gate.
# Kernels are written with Tilework.kernel {} (raw dialect strings are not
# supported on :cpu) or with operators alone.
class UseCasesCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def teardown
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  def f32(array)
    array.pack("f*").unpack("f*")
  end

  # --- Linear-algebra primitives written purely as operator chains ---

  def test_dot_product_and_l2_norm_via_operators
    n = 2_000
    a = f32(Array.new(n) { rand * 2 - 1 })
    b = f32(Array.new(n) { rand * 2 - 1 })
    da = Tilework[a]
    db = Tilework[b]

    dot = (da * db).sum
    norm = Math.sqrt((da * da).sum)

    cpu_dot = a.zip(b).sum { |x, y| x * y }
    cpu_norm = Math.sqrt(a.sum { |x| x * x })
    assert_in_delta cpu_dot, dot, [cpu_dot.abs, 1.0].max * 1e-3
    assert_in_delta cpu_norm, norm, cpu_norm * 1e-4
  end

  # --- Monte Carlo estimate of pi ---

  INSIDE_CIRCLE = Tilework.kernel("inside_circle_cpu") do |x, y, hit, n|
    i = global_id_x
    hit[i] = (x[i] * x[i] + y[i] * y[i] <= 1.0) ? 1 : 0 if i < n
  end

  def test_monte_carlo_pi_estimation
    n = 5_000
    rng = Random.new(42)
    xs = Array.new(n) { rng.rand }
    ys = Array.new(n) { rng.rand }

    hits = Tilework::Buffer.new(:int32, n)
    INSIDE_CIRCLE.(Tilework[xs], Tilework[ys], hits, n)

    pi = 4.0 * hits.sum / n
    assert_in_delta Math::PI, pi, 0.05
  end

  # --- Matrix multiply: 2D grid/block and global_id_x/y ---

  MATMUL_2D = Tilework.kernel("matmul_2d_cpu") do |a, b, c, size|
    col = global_id_x
    row = global_id_y
    if row < size && col < size
      acc = 0.0
      size.times { |k| acc = acc + a[row * size + k] * b[k * size + col] }
      c[row * size + col] = acc
    end
  end

  def test_matrix_multiply_with_2d_grid
    size = 16
    a = f32(Array.new(size * size) { rand * 2 - 1 })
    b = f32(Array.new(size * size) { rand * 2 - 1 })

    dc = Tilework::Buffer.new(:float32, size * size)
    MATMUL_2D.launch(grid: [1, 1], block: [size, size], args: [Tilework[a], Tilework[b], dc, size])

    result = dc.to_a
    [0, size / 2, size - 1].each do |row|
      size.times do |col|
        expected = (0...size).sum { |k| a[row * size + k] * b[k * size + col] }
        assert_in_delta expected, result[row * size + col], 1e-3, "c[#{row}][#{col}]"
      end
    end
  end

  # --- 1D heat diffusion: repeated launches over ping-pong buffers ---

  DIFFUSE = Tilework.kernel("diffuse_cpu") do |cur, nxt, alpha, n|
    i = global_id_x
    if i < n
      left = i > 0 ? cur[i - 1] : cur[i]
      right = i < n - 1 ? cur[i + 1] : cur[i]
      nxt[i] = cur[i] + alpha * (left - 2.0 * cur[i] + right)
    end
  end

  def test_heat_diffusion_simulation
    n = 128
    steps = 30
    alpha = 0.2
    initial = Array.new(n, 0.0)
    (n / 2 - 5..n / 2 + 5).each { |i| initial[i] = 100.0 }

    cur = Tilework[initial]
    nxt = Tilework::Buffer.new(:float32, n)
    steps.times do
      DIFFUSE.(cur, nxt, alpha, n)
      cur, nxt = nxt, cur
    end
    result = cur.to_a

    ref = initial.dup
    steps.times do
      ref = Array.new(n) do |i|
        left = i.positive? ? ref[i - 1] : ref[i]
        right = i < n - 1 ? ref[i + 1] : ref[i]
        ref[i] + alpha * (left - 2.0 * ref[i] + right)
      end
    end

    n.times { |i| assert_in_delta ref[i], result[i], 1e-2, "cell #{i}" }
    assert_in_delta initial.sum, result.sum, initial.sum * 1e-4
  end

  # --- Two-layer MLP forward pass: with_gpu + a DSL kernel + operators ---

  DENSE = Tilework.kernel("dense_cpu") do |x, w, b, out, batch, din, dout|
    i = global_id_x
    if i < batch * dout
      row = i / dout
      col = i % dout
      acc = 0.0
      din.times { |k| acc = acc + x[row * din + k] * w[k * dout + col] }
      out[i] = acc + b[col]
    end
  end

  SOFTMAX_ROWS = Tilework.kernel("softmax_rows_cpu") do |logits, probs, rows, cols|
    r = global_id_x
    if r < rows
      m = logits[r * cols]
      cols.times do |j|
        v = logits[r * cols + j]
        m = v > m ? v : m
      end
      s = 0.0
      cols.times { |j| s = s + Math.exp(logits[r * cols + j] - m) }
      cols.times { |j| probs[r * cols + j] = Math.exp(logits[r * cols + j] - m) / s }
    end
  end

  def test_mlp_forward_pass
    batch = 16
    d_in = 12
    d_hidden = 16
    d_out = 5
    rng = Random.new(3)
    x  = f32(Array.new(batch * d_in)      { rng.rand * 2 - 1 })
    w1 = f32(Array.new(d_in * d_hidden)   { rng.rand * 0.4 - 0.2 })
    b1 = f32(Array.new(d_hidden)          { rng.rand * 0.1 })
    w2 = f32(Array.new(d_hidden * d_out)  { rng.rand * 0.4 - 0.2 })
    b2 = f32(Array.new(d_out)             { rng.rand * 0.1 })

    probs = Tilework.run(x, w1, b1, w2, b2) do |dx, dw1, db1, dw2, db2|
      hidden = Tilework::Buffer.new(:float32, batch * d_hidden)
      DENSE.(dx, dw1, db1, hidden, batch, d_in, d_hidden)
      hidden = hidden.maximum(0.0) # ReLU

      logits = Tilework::Buffer.new(:float32, batch * d_out)
      DENSE.(hidden, dw2, db2, logits, batch, d_hidden, d_out)

      out = Tilework::Buffer.new(:float32, batch * d_out)
      SOFTMAX_ROWS.(logits, out, batch, d_out)
      out
    end

    dense_cpu = lambda do |input, w, b, rows, din, dout|
      Array.new(rows * dout) do |i|
        r, c = i.divmod(dout)
        b[c] + (0...din).sum { |k| input[r * din + k] * w[k * dout + c] }
      end
    end
    h = dense_cpu.(x, w1, b1, batch, d_in, d_hidden).map { |v| [v, 0.0].max }
    logits = dense_cpu.(h, w2, b2, batch, d_hidden, d_out)
    expected = logits.each_slice(d_out).flat_map do |row|
      m = row.max
      exps = row.map { |v| Math.exp(v - m) }
      total = exps.sum
      exps.map { |e| e / total }
    end

    assert_equal batch * d_out, probs.length
    probs.each_slice(d_out) { |row| assert_in_delta 1.0, row.sum, 1e-4 }
    expected.zip(probs).each_with_index do |(e, p), i|
      assert_in_delta e, p, 1e-4, "prob #{i}"
    end
  end

  # --- softmax: a DSL kernel plus sum / scalar division inside with_gpu ---

  EXP_SHIFTED = Tilework.kernel("exp_shifted_cpu") do |x, out, max_val, n|
    i = global_id_x
    out[i] = Math.exp(x[i] - max_val) if i < n
  end

  def test_softmax_pipeline
    n = 512
    logits = f32(Array.new(n) { rand * 8 - 4 })
    max = logits.max

    probs = Tilework.run(logits) do |dx|
      exps = Tilework::Buffer.new(:float32, n)
      EXP_SHIFTED.(dx, exps, max, n)
      exps / exps.sum
    end

    cpu_exps = logits.map { |x| Math.exp(x - max) }
    cpu_total = cpu_exps.sum
    assert_in_delta 1.0, probs.sum, 1e-3
    probs.each_with_index do |p, i|
      assert_in_delta cpu_exps[i] / cpu_total, p, 1e-6, "prob #{i}"
    end
  end
end
