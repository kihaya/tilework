# frozen_string_literal: true

require_relative "../test_helper"

# Small ML workloads on the :cpu backend. No GPU is needed, so this always
# runs in CI (.github/workflows/ci.yml) without the GPU_TESTS gate. Kernels
# are written with Tilework.kernel {}, which :cpu supports (raw dialect
# strings are not); the host-side error/gradient/update logic is plain Ruby.
class MLCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def teardown
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  # --- Logistic regression: a gradient-descent training loop on :cpu ---

  PREDICT = Tilework.kernel("predict_cpu") do |x, w, b, p, n, d|
    i = global_id_x
    if i < n
      z = b
      d.times { |j| z = z + w[j] * x[j * n + i] }
      p[i] = 1.0 / (1.0 + Math.exp(-z))
    end
  end

  def test_logistic_regression_training
    n = 300
    d = 4
    rng = Random.new(7)
    w_true = [2.0, -3.0, 1.0, 0.5]
    b_true = 0.5

    xcols = Array.new(d) { Array.new(n) { rng.rand * 2 - 1 } }
    labels = Array.new(n) do |i|
      z = b_true + d.times.sum { |j| w_true[j] * xcols[j][i] }
      z > 0 ? 1.0 : 0.0
    end

    dx = Tilework[xcols.flatten]
    dxcols = xcols.map { |col| Tilework[col] }
    dy = Tilework[labels]
    dw = Tilework::Buffer.new(:float32, d)
    dp = Tilework::Buffer.new(:float32, n)

    w = Array.new(d, 0.0)
    b = 0.0
    lr = 1.0
    losses = []

    run_predict = lambda do
      dw.write(w)
      PREDICT.(dx, dw, b, dp, n, d)
    end

    30.times do
      run_predict.call
      err = dp - dy
      d.times { |j| w[j] -= lr * (err * dxcols[j]).sum / n }
      b -= lr * err.sum / n

      probs = dp.to_a
      losses << -labels.zip(probs).sum { |y, p| y * Math.log(p.clamp(1e-7, 1)) + (1 - y) * Math.log((1 - p).clamp(1e-7, 1)) } / n
    end

    assert_operator losses.last, :<, losses.first * 0.5,
                    "loss did not decrease: #{losses.first} -> #{losses.last}"

    run_predict.call
    predictions = dp.to_a.map { |p| p > 0.5 ? 1.0 : 0.0 }
    accuracy = predictions.zip(labels).count { |p, y| p == y } / n.to_f
    assert_operator accuracy, :>, 0.9, "training accuracy too low: #{accuracy}"

    dot = w.zip(w_true).sum { |a, c| a * c }
    cos = dot / (Math.sqrt(w.sum { |v| v * v }) * Math.sqrt(w_true.sum { |v| v * v }))
    assert_operator cos, :>, 0.95, "learned weights point the wrong way: #{w.inspect}"
  end

  # --- k-means: assignment in a DSL kernel, centroid update on the host ---

  ASSIGN = Tilework.kernel("assign_cpu") do |px, py, centroids, out, n, k|
    i = global_id_x
    if i < n
      best = 1e30
      best_c = 0
      k.times do |c|
        dx = px[i] - centroids[c * 2]
        dy = py[i] - centroids[c * 2 + 1]
        dist = dx * dx + dy * dy
        if dist < best
          best = dist
          best_c = c
        end
      end
      out[i] = best_c
    end
  end

  def test_kmeans_clustering
    k = 3
    per_cluster = 100
    rng = Random.new(11)
    true_centers = [[0.0, 0.0], [5.0, 5.0], [-4.0, 6.0]]

    points = true_centers.flat_map do |cx, cy|
      Array.new(per_cluster) { [cx + gauss(rng) * 0.8, cy + gauss(rng) * 0.8] }
    end
    n = points.length

    dpx = Tilework[points.map(&:first)]
    dpy = Tilework[points.map(&:last)]
    dassign = Tilework::Buffer.new(:int32, n)
    dcentroids = Tilework::Buffer.new(:float32, k * 2)

    centroids = [[1.0, 1.0], [4.0, 4.0], [-1.0, 4.0]]

    10.times do
      dcentroids.write(centroids.flatten)
      ASSIGN.(dpx, dpy, dcentroids, dassign, n, k)
      assignments = dassign.to_a
      centroids = k.times.map do |c|
        members = points.each_index.select { |i| assignments[i] == c }
        next centroids[c] if members.empty?

        [members.sum { |i| points[i][0] } / members.size,
         members.sum { |i| points[i][1] } / members.size]
      end
    end

    true_centers.each do |tx, ty|
      nearest = centroids.map { |cx, cy| Math.sqrt((cx - tx)**2 + (cy - ty)**2) }.min
      assert_operator nearest, :<, 0.3,
                      "no centroid found near (#{tx}, #{ty}): #{centroids.inspect}"
    end
  end

  private

  def gauss(rng)
    Math.sqrt(-2 * Math.log(1 - rng.rand)) * Math.cos(2 * Math::PI * rng.rand)
  end
end
