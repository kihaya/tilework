# frozen_string_literal: true

require_relative "../test_helper"

# Graph-layer numbers on a real backend (:cpu). Cross-backend agreement is
# Tilework::Conformance's job (the graph_* checks); this file holds run-time
# behavior worth confirming on :cpu — state advances across runs and freeing
# is not broken.
class GraphCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def teardown
    Tilework::Graph.strict_state = :view
    Tilework::Graph.async = true
    Tilework::Graph.buffer_pool = true
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  def test_a_plan_computes_the_same_values_as_the_eager_path
    a = Tilework::Buffer.from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = Tilework::Buffer.from([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3])
      g.output((x.matmul(g.param(w)) * 2.0) + 1.0)
    end.compile

    expected = ((a.matmul(w) * 2.0) + 1.0).to_nested_a
    assert_equal expected, plan.run(x: a).to_nested_a
  end

  # The KV cache itself: rerun the same plan while advancing pos.
  def test_a_state_advances_across_runs
    cache = Tilework::Buffer.new(:float32, 8, shape: [4, 2])
    cache.write([0.0] * 8)
    plan = Tilework::Graph.build do |g|
      kv0 = g.state(cache, name: :kv)
      k = g.input(:k, [1, 2])
      kv1 = g.write_into(kv0, at: g.index(:pos), value: k)
      g.output(g.read(kv1) * 1.0)
    end.compile

    3.times do |step|
      value = Tilework::Buffer.from([[(step + 1).to_f, (step + 1) * 10.0]])
      out = plan.run(k: value, pos: step)
      assert_equal (step + 1).to_f, out.to_nested_a[step][0]
    end
    assert_equal [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [0.0, 0.0]], cache.to_nested_a
  end

  # RNG is a pure value, not mutable state. The key comes from (plan seed,
  # node index), so the same seed yields the same sequence on every run.
  def test_random_draws_are_pure_in_a_seed
    plan = Tilework::Graph.build do |g|
      g.output(g.random([16]))
      g.output(g.random([16]))
    end.compile

    first, second = plan.run(seed: 42).map(&:to_a)
    again = plan.run(seed: 42).map(&:to_a)
    advanced = plan.run(seed: plan.next_seed(42)).map(&:to_a)

    assert_equal [first, second], again, "同じ seed なら同じ列"
    refute_equal first, second, "同じプランの 2 本の draw は別の列"
    refute_equal [first, second], advanced, "seed を進めれば変わる"
  end

  # A decode loop with placed writes is numerically right too (the copy
  # disappears, so this takes the path where the write node does nothing).
  def test_a_placed_write_advances_the_state_correctly
    cache = Tilework::Buffer.new(:float32, 8, shape: [4, 2])
    cache.write([0.0] * 8)
    weight = Tilework::Buffer.from([[2.0, 0.0], [0.0, 3.0]])
    plan = Tilework::Graph.build do |g|
      kv0 = g.state(cache, name: :kv)
      k = g.input(:k, [1, 2])
      kv1 = g.write_into(kv0, at: g.index(:pos), value: k.matmul(g.param(weight)))
      g.output(g.read(kv1) * 1.0)
    end.compile

    3.times { |step| plan.run(k: Tilework::Buffer.from([[step + 1.0, 1.0]]), pos: step) }

    assert_equal [[2.0, 3.0], [4.0, 3.0], [6.0, 3.0], [0.0, 0.0]], cache.to_nested_a
  end

  # State buffers are live-in and live-out of the plan, so they are never
  # freed or reused as intermediates (that would wipe them on the next run).
  def test_state_buffers_survive_the_plan
    state = Tilework::Buffer.from([[1.0, 2.0], [3.0, 4.0]])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :s)
      g.output(g.read(s0) * 3.0)
    end.compile

    assert_equal [[3.0, 6.0], [9.0, 12.0]], plan.run.to_nested_a
    assert_equal [[3.0, 6.0], [9.0, 12.0]], plan.run.to_nested_a
    assert_equal [[1.0, 2.0], [3.0, 4.0]], state.to_nested_a
  end

  def test_reads_of_a_strided_region_are_materialized
    state = Tilework::Buffer.from([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :s)
      g.output(g.read(s0, s0.axis(1, 1...3)) * 1.0)
    end.compile

    assert_equal [[2.0, 3.0], [6.0, 7.0]], plan.run.to_nested_a
  end

  # A read used as an output is materialized; returned as a view, later
  # writes in the same plan would change it.
  def test_a_read_used_as_an_output_is_not_clobbered_by_a_later_write
    state = Tilework::Buffer.from([[1.0, 1.0], [2.0, 2.0]])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :s)
      before = g.read(s0, 0)
      g.write_into(s0, at: 0, value: g.param(Tilework::Buffer.from([[9.0, 9.0]])))
      g.output(before)
    end.compile

    assert_equal [[1.0, 1.0]], plan.run.to_nested_a
    assert_equal [[9.0, 9.0], [2.0, 2.0]], state.to_nested_a
  end

  # Recycling intermediates does not change values; runs with and without the
  # pool agree (broken reuse would leak previous values in here).
  def test_reusing_intermediates_does_not_change_the_values
    x = Tilework::Buffer.from(Array.new(64) { |i| i.to_f })
    build = lambda do
      Tilework::Graph.build do |g|
        value = g.input(:x, [64])
        10.times { value = (value * 1.5) + 0.5 }
        g.output(value)
      end.compile
    end

    plan = build.call
    expected = (0...64).map { |i| (0...10).inject(i.to_f) { |acc, _| (acc * 1.5) + 0.5 } }
    first = plan.run(x: x).to_a
    second = plan.run(x: x).to_a
    assert_equal first, second, "同じプランを走らせ直しても同じ値"
    first.each_with_index { |value, i| assert_in_delta expected[i], value, 1e-2 }

    Tilework::Graph.buffer_pool = false
    Tilework::Graph.async = false
    assert_equal first, build.call.run(x: x).to_a, "プールと非同期を切っても同じ値"
  ensure
    Tilework::Graph.buffer_pool = true
    Tilework::Graph.async = true
  end

  def test_opaque_nodes_run_their_block_and_are_checked_against_the_declaration
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [4])
      g.output(g.opaque(x, shape: [4], name: "double") { |buffer| buffer * 2.0 } + 1.0)
    end.compile
    assert_equal [3.0, 5.0, 7.0, 9.0], plan.run(x: Tilework::Buffer.from([1.0, 2.0, 3.0, 4.0])).to_a

    lying = Tilework::Graph.build do |g|
      x = g.input(:x, [4])
      g.output(g.opaque(x, shape: [8], name: "wrong") { |buffer| buffer * 2.0 })
    end.compile
    error = assert_raises(Tilework::Error) { lying.run(x: Tilework::Buffer.from([1.0, 2.0, 3.0, 4.0])) }
    assert_includes error.message, "but returned"
  end

  # Reductions stay inside the graph, so a loss can be a plan output.
  def test_a_loss_can_be_expressed_end_to_end
    plan = Tilework::Graph.build do |g|
      diff = g.input(:pred, [8]) - g.input(:target, [8])
      g.output((diff * diff).mean)
    end.compile

    pred = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    target = [1.5, 2.5, 2.0, 4.0, 5.5, 5.0, 7.0, 9.0]
    expected = pred.zip(target).sum { |a, b| (a - b)**2 } / pred.length
    result = plan.run(pred: Tilework::Buffer.from(pred), target: Tilework::Buffer.from(target))

    assert_equal 1, result.length
    assert_in_delta expected, result.to_a.first, 1e-5
  end

  def test_argmax_returns_the_index_as_a_buffer
    plan = Tilework::Graph.build { |g| g.output(g.input(:x, [5]).argmax) }.compile
    result = plan.run(x: Tilework::Buffer.from([1.0, 9.0, 3.0, 2.0, 4.0]))

    assert_equal :int32, result.dtype
    assert_equal [1], result.to_a
  end

  # A state read (a view) can be reduced directly.
  def test_reductions_run_over_a_state_read
    cache = Tilework::Buffer.from([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(cache, name: :kv)
      g.output(g.read(s0, 0..1).sum)
    end.compile

    assert_in_delta 10.0, plan.run.to_a.first, 1e-5
  end


  # --- Broadcasting and axis reductions (docs/graph-ir.md Phase B) ---

  def test_a_row_vector_is_broadcast_across_rows
    x = Tilework::Buffer.from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    plan = Tilework::Graph.build do |g|
      g.output(g.input(:x, [2, 3]) + g.param(Tilework::Buffer.from([10.0, 20.0, 30.0])))
    end.compile

    assert_equal [[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]], plan.run(x: x).to_nested_a
  end

  def test_a_column_vector_is_broadcast_across_columns
    x = Tilework::Buffer.from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    col = Tilework::Buffer.new(:float32, 2, shape: [2, 1])
    col.write([10.0, 100.0])
    plan = Tilework::Graph.build do |g|
      g.output(g.input(:x, [2, 3]) * g.param(col))
    end.compile

    assert_equal [[10.0, 20.0, 30.0], [400.0, 500.0, 600.0]], plan.run(x: x).to_nested_a
  end

  # Both sides stretch; 3D follows the same rule.
  def test_broadcasting_expands_both_operands_and_more_than_two_dimensions
    rows = nd([2, 1, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    cols = nd([1, 2, 1], [10.0, 100.0])
    plan = Tilework::Graph.build do |g|
      g.output(g.param(rows) * g.param(cols))
    end.compile

    result = plan.run
    assert_equal [2, 2, 3], result.shape
    assert_equal [10.0, 20.0, 30.0, 100.0, 200.0, 300.0,
                  40.0, 50.0, 60.0, 400.0, 500.0, 600.0], result.to_a
  end

  def test_axis_reductions_fold_one_axis_at_a_time
    x = nd([2, 3, 2], (1..12).map(&:to_f))
    plan = Tilework::Graph.build do |g|
      t = g.param(x)
      g.output(t.sum(axis: 0))
      g.output(t.sum(axis: 1))
      g.output(t.sum(axis: 2))
      g.output(t.max(axis: 1))
      g.output(t.min(axis: 1))
      g.output(t.mean(axis: 1))
    end.compile

    by_first, by_middle, by_last, maxes, mins, means = plan.run
    # Axis 0: add the first half [1..6] to the second [7..12]
    assert_equal [8.0, 10.0, 12.0, 14.0, 16.0, 18.0], by_first.to_a
    assert_equal [2, 3, 2], x.shape
    assert_equal [3, 2], by_first.shape
    # Axis 1: sum the 3 rows of each batch (per column)
    assert_equal [[9.0, 12.0], [27.0, 30.0]], by_middle.to_nested_a
    # Axis 2: sum adjacent pairs
    assert_equal [[3.0, 7.0, 11.0], [15.0, 19.0, 23.0]], by_last.to_nested_a
    assert_equal [[5.0, 6.0], [11.0, 12.0]], maxes.to_nested_a
    assert_equal [[1.0, 2.0], [7.0, 8.0]], mins.to_nested_a
    assert_equal [[3.0, 4.0], [9.0, 10.0]], means.to_nested_a
  end

  def test_axis_arg_reductions_return_positions_along_the_axis
    x = Tilework::Buffer.from([[1.0, 9.0, 3.0], [7.0, 2.0, 5.0]])
    plan = Tilework::Graph.build do |g|
      t = g.input(:x, [2, 3])
      g.output(t.argmax(axis: 1))
      g.output(t.argmin(axis: 1))
      g.output(t.argmax(axis: 0))
    end.compile

    by_row_max, by_row_min, by_col_max = plan.run(x: x)
    assert_equal :int32, by_row_max.dtype
    assert_equal [1, 0], by_row_max.to_a
    assert_equal [0, 1], by_row_min.to_a
    assert_equal [1, 0, 1], by_col_max.to_a
  end

  # softmax can be written in primitives alone — the minimal case of Phase B's
  # goal in docs/graph-ir.md (high-level ops decompose).
  def test_softmax_is_expressible_in_primitives_and_matches_nn
    x = Tilework::Buffer.from([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
    plan = Tilework::Graph.build do |g|
      t = g.input(:x, [2, 3])
      e = (t - t.max(axis: -1, keepdims: true)).exp
      g.output(e / e.sum(axis: -1, keepdims: true))
    end.compile

    expected = Tilework::NN.softmax(x).to_nested_a
    plan.run(x: x).to_nested_a.each_with_index do |row, i|
      row.each_with_index { |value, j| assert_in_delta expected[i][j], value, 1e-6 }
    end
  end


  def test_where_selects_between_tensors_and_scalars
    x = Tilework::Buffer.from([[-2.0, 3.5], [0.0, -1.25]])
    plan = Tilework::Graph.build do |g|
      t = g.input(:x, [2, 2])
      g.output(g.where(t > 0, t, 0.0))                      # relu
      g.output(g.where(t.eq(0.0), t + 100.0, t))            # both sides Tensors
      g.output(g.where(t > 0, 1, 0))                        # an int32 mask
    end.compile

    relu, patched, mask = plan.run(x: x)
    assert_equal [[0.0, 3.5], [0.0, 0.0]], relu.to_nested_a
    assert_equal [[-2.0, 3.5], [100.0, -1.25]], patched.to_nested_a
    assert_equal :int32, mask.dtype
    assert_equal [0, 1, 0, 0], mask.to_a
  end

  # float -> int truncates toward zero (like a C cast).
  def test_cast_truncates_toward_zero
    x = Tilework::Buffer.from([-2.75, -0.5, 0.5, 3.9])
    plan = Tilework::Graph.build do |g|
      t = g.input(:x, [4])
      g.output(t.cast(:int32))
      g.output(t.cast(:int32).cast(:float32))
    end.compile

    ints, back = plan.run(x: x)
    assert_equal [-2, 0, 0, 3], ints.to_a
    assert_equal [-2.0, 0.0, 0.0, 3.0], back.to_a
  end

  def test_iota_writes_the_index_along_its_axis
    plan = Tilework::Graph.build do |g|
      g.output(g.iota([2, 3], axis: 0))
      g.output(g.iota([2, 3], axis: 1))
      g.output(g.iota([2, 2, 2], axis: 1, dtype: :float32))
    end.compile

    rows, cols, middle = plan.run
    assert_equal [0, 0, 0, 1, 1, 1], rows.to_a
    assert_equal [0, 1, 2, 0, 1, 2], cols.to_a
    assert_equal [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], middle.to_a
  end

  # A causal mask from primitives alone (no constant brought in from the host).
  def test_a_causal_mask_is_expressible_in_primitives
    scores = nd([3, 3], Array.new(9) { |i| i.to_f })
    plan = Tilework::Graph.build do |g|
      s = g.input(:s, [3, 3])
      keep = g.iota([3, 3], axis: 1) <= g.iota([3, 3], axis: 0)
      g.output(g.where(keep, s, -1e9))
    end.compile

    assert_equal [[0.0, -1e9, -1e9], [3.0, 4.0, -1e9], [6.0, 7.0, 8.0]],
                 plan.run(s: scores).to_nested_a
  end

  # reshape is a view, so the source buffer must live until the view's readers
  # finish — returning it to the pool early corrupts values.
  def test_a_reshaped_intermediate_stays_alive_until_its_readers_finish
    x = Tilework::Buffer.from([[1.0, 2.0], [3.0, 4.0]])
    y = Tilework::Buffer.from([10.0, 20.0, 30.0, 40.0])
    plan = Tilework::Graph.build do |g|
      a = g.input(:x, [2, 2]) * 2.0
      g.output((a.reshape(4) + g.input(:y, [4])) * 0.5)
      g.output(a.reshape(1, 4))
    end.compile

    3.times do
      summed, viewed = plan.run(x: x, y: y)
      assert_equal [6.0, 12.0, 18.0, 24.0], summed.to_a
      assert_equal [[2.0, 4.0, 6.0, 8.0]], viewed.to_nested_a
    end
  end


  # A folded broadcast reads **the same values** as reading it materialized;
  # the arithmetic is unchanged, so :cpu agrees bit for bit.
  def test_folding_a_broadcast_does_not_change_the_values
    x = nd([3, 4], (1..12).map { |v| (v * 0.25) - 1.0 })
    bias = Tilework::Buffer.from([0.5, -0.5, 1.0, 0.0])
    col = nd([3, 1], [2.0, -1.0, 0.5])
    build = lambda do
      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [3, 4])
        g.output(((t + g.param(bias)) * g.param(col)).tanh)
      end.compile
      [plan, plan.run(x: x).to_a]
    end

    folded, folded_values = build.call
    Tilework::Graph.fusion = false
    plain, plain_values = build.call

    assert_equal 0, folded.explain.lines.count { |l| l.include?("broadcast") && !l.include?("(fused)") }
    assert_equal 2, plain.explain.lines.count { |l| l.include?("broadcast") && !l.include?("(fused)") }
    assert_equal plain_values, folded_values
  ensure
    Tilework::Graph.fusion = true
  end


  # Row-wise fusion agrees with the unfused path **bit for bit**. Both fold
  # from the start of the row, and the fused side reproduces the rounding of
  # writing back the reduction (the same trick as elementwise fusion).
  def test_row_wise_fusion_is_bit_identical_on_the_cpu_backend
    rows = 256
    cols = 8
    x = nd([rows, cols], Array.new(rows * cols) { |i| ((i % 13) - 6) * 0.5 })
    build = lambda do
      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [rows, cols])
        centered = t - t.mean(axis: -1, keepdims: true)
        g.output(centered / (t.max(axis: -1, keepdims: true) + 8.0))
      end.compile
      [plan, plan.run(x: x).to_a]
    end

    fused, fused_values = build.call
    Tilework::Graph.row_fusion = false
    plain, plain_values = build.call

    assert fused.fusion_stats[:nodes] < plain.fusion_stats[:nodes],
           "row fusion should launch fewer kernels"
    assert_equal plain_values, fused_values
  ensure
    Tilework::Graph.row_fusion = true
  end


  # :cpu has no convolution, so the graph runs it decomposed into primitives —
  # a concrete case of "convolution is portable in a graph".
  def test_conv2d_runs_on_the_cpu_backend_through_its_decomposition
    x = nd([1, 1, 3, 3], (1..9).map(&:to_f))
    weight = nd([1, 1, 2, 2], [1.0, 0.0, 0.0, 1.0])           # picks the diagonal only
    plan = Tilework::Graph.build do |g|
      g.output(Tilework::Conv.forward(g.input(:x, [1, 1, 3, 3]), g.param(weight)))
    end.compile

    # Decomposed, so no conv node remains and nothing is required.
    assert_equal 0, plan.nodes.count { |n| n.op == :conv2d }
    assert_empty plan.requires
    out = plan.run(x: x)
    assert_equal [1, 1, 2, 2], out.shape
    # Diagonal sums of 2x2: [1+5, 2+6, 4+8, 5+9]
    assert_equal [6.0, 8.0, 12.0, 14.0], out.to_a
  end

  # Eager Conv still raises on :cpu (only the graph has the decomposition).
  def test_eager_conv2d_still_raises_on_the_cpu_backend
    x = nd([1, 1, 3, 3], (1..9).map(&:to_f))
    weight = nd([1, 1, 2, 2], [1.0, 0.0, 0.0, 1.0])
    error = assert_raises(Tilework::Error) { Tilework::Conv.forward(x, weight) }
    assert_includes error.message, "convolution"
  end

  # Buffer.from reads nesting two levels deep only, so 3D is built by hand.
  def nd(shape, values)
    buffer = Tilework::Buffer.new(:float32, values.length, shape: shape)
    buffer.write(values)
    buffer
  end

  # [b, m, k] x [b, k, n]. Eager has no such shape, so per-batch 2D matmuls
  # are the reference.
  def test_batched_matmul_matches_per_batch_matmuls
    a = nd([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    b = nd([2, 2, 3], [1.0, 0.0, 2.0, 0.0, 1.0, 3.0, 2.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    plan = Tilework::Graph.build do |g|
      g.output(g.input(:a, [2, 2, 2]).matmul(g.param(b)))
    end.compile

    expected = 2.times.flat_map do |i|
      a.slice(i).reshape(2, 2).matmul(b.slice(i).reshape(2, 3)).to_a
    end
    result = plan.run(a: a)

    assert_equal [2, 2, 3], result.shape
    assert_equal expected, result.to_a
  end

  def test_concat_joins_along_the_outermost_dimension
    a = Tilework::Buffer.from([[1.0, 2.0], [3.0, 4.0]])
    b = Tilework::Buffer.from([[5.0, 6.0]])
    plan = Tilework::Graph.build do |g|
      g.output(g.concat(g.input(:a, [2, 2]), g.param(b)) * 2.0)
    end.compile

    assert_equal [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], plan.run(a: a).to_nested_a
  end

  # Row-wise reductions look at the last dim, so 3D and up pass through without losing dims.
  def test_row_wise_nn_ops_accept_more_than_two_dimensions
    x = nd([2, 2, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    plan = Tilework::Graph.build do |g|
      g.output(Tilework::NN.softmax(g.input(:x, [2, 2, 2])))
    end.compile

    result = plan.run(x: x)
    assert_equal [2, 2, 2], result.shape
    # softmax of each row [v, v+1] is the same for every row.
    assert_equal Tilework::NN.softmax(x.reshape(4, 2)).to_a, result.to_a
  end

  # On :cpu fused and unfused agree **bit for bit**. conformance allows a
  # tolerance only for shader-compiler FMA contraction, which cannot happen
  # here, so this checks the rounding reproduction exactly.
  def test_fusion_is_bit_identical_on_the_cpu_backend
    x = Tilework::Buffer.from(Array.new(64) { |i| ((i % 9) - 4) * 0.3 })
    build = lambda do
      plan = Tilework::Graph.build do |g|
        v = g.input(:x, [64])
        5.times { v = ((v * 1.5) + 0.25).tanh }
        # cast and where are expression-tree nodes too, so they are held to bit identity.
        v = g.where(v > 0.0, v, v.cast(:int32).cast(:float32) - 0.5)
        g.output(v.abs)
      end.compile
      [plan, plan.run(x: x).to_a]
    end

    fused, fused_values = build.call
    Tilework::Graph.fusion = false
    plain, plain_values = build.call

    # Two clusters because all three of where's operands read v — a value read
    # in three places is materialized, so the chain breaks there (the diamond rule).
    assert_equal 2, fused.fusion_stats[:clusters]
    assert_equal 0, plain.fusion_stats[:clusters]
    assert_equal plain_values, fused_values
  ensure
    Tilework::Graph.fusion = true
  end

  # The shape #17's case study wanted in a graph. NN accepts Tensors, so a
  # pre-norm block reads exactly as it does eager, without g.opaque.
  def test_a_transformer_block_is_expressible_without_opaque_nodes
    x = Tilework::Buffer.from([[0.5, -1.0, 2.0, 0.25], [1.0, 0.0, -0.5, 1.5]])
    gamma = Tilework::Buffer.from([1.0, 1.1, 0.9, 1.0])
    wq = Tilework::Buffer.from([[1.0, 0.0, 0.5, 0.0], [0.0, 1.0, 0.0, 0.5],
                                [0.5, 0.0, 1.0, 0.0], [0.0, 0.5, 0.0, 1.0]])
    w_up = Tilework::Buffer.from([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, -1.0]])

    block = lambda do |input, matmul|
      h = Tilework::NN.rmsnorm(input, gamma: gamma)
      a = Tilework::NN.multi_head_attention(matmul.call(h, wq), matmul.call(h, wq), h, heads: 2)
      Tilework::NN.silu(matmul.call(input + a, w_up))
    end

    plan = Tilework::Graph.build do |g|
      t = g.input(:x, [2, 4])
      g.output(block.call(t, ->(a, b) { a.matmul(g.param(b)) }))
    end.compile

    expected = block.call(x, ->(a, b) { a.matmul(b) })
    assert_equal :nn, plan.order.last.op
    assert_equal expected.to_nested_a, plan.run(x: x).to_nested_a
  end

  # NN outputs are treated like any other intermediate (returned to the pool
  # and reused next run); otherwise a decode loop's allocations keep growing.
  def test_nn_outputs_do_not_grow_the_pool_across_runs
    x = Tilework::Buffer.from([[1.0, 2.0], [3.0, 4.0]])
    plan = Tilework::Graph.build do |g|
      g.output(Tilework::NN.softmax(g.input(:x, [2, 2])) * 2.0)
    end.compile

    3.times { plan.run(x: x) }
    steady = plan.pool_stats[:allocated]
    3.times { plan.run(x: x) }

    assert_equal 3, plan.pool_stats[:allocated] - steady, "毎回増えるのは外に出る出力だけ"
  end

  def test_input_shape_is_checked_against_the_declaration
    plan = Tilework::Graph.build { |g| g.output(g.input(:x, [2, 2]) * 1.0) }.compile
    error = assert_raises(Tilework::Error) { plan.run(x: Tilework::Buffer.from([1.0, 2.0, 3.0, 4.0])) }
    assert_includes error.message, "was declared as [2, 2]"
  end
end
