# frozen_string_literal: true

require_relative "../test_helper"

# **Compile-time** behavior of the graph layer: shape/type inference, CSE, the
# linearity checks on mutable state, and ordering (WAR edges). None of it
# touches a device, so it runs on MockBridge. Values are checked by
# graph_cpu_test.rb and Conformance.
class GraphTest < Minitest::Test
  include BridgeTestSetup

  def f32(shape) = Tilework::Buffer.new(:float32, shape.inject(:*), shape: shape)

  # --- Shape and type inference ---

  def test_infers_matmul_shape_and_rejects_mismatches
    Tilework::Graph.build do |g|
      a = g.input(:a, [2, 3])
      b = g.param(f32([3, 4]))
      assert_equal [2, 4], a.matmul(b).shape

      c = g.param(f32([5, 2]))
      error = assert_raises(ArgumentError) { a.matmul(c) }
      assert_includes error.message, "matmul shape mismatch"
    end
  end

  def test_dtype_follows_the_ops_promotion_rules
    Tilework::Graph.build do |g|
      i = g.input(:i, [4], dtype: :int32)
      f = g.input(:f, [4])
      assert_equal :int32, (i * 2).dtype       # stays int32 only when both sides are int32
      assert_equal :float32, (i * 2.0).dtype
      assert_equal :float32, (i + f).dtype
      assert_equal :int32, i.eq(f).dtype       # comparisons give a 0/1 mask
      assert_equal :float32, i.exp.dtype       # unary math widens to float
      assert_equal :int32, i.abs.dtype
    end
  end

  # Unlike eager, reductions return a shape-[1] Tensor rather than a host
  # number (dropping to the host midway would end the plan).
  def test_reductions_collapse_to_a_one_element_tensor
    Tilework::Graph.build do |g|
      x = g.input(:x, [4, 8])
      assert_equal [1], x.sum.shape
      assert_equal :float32, x.sum.dtype
      assert_equal :float32, x.mean.dtype
      assert_equal :int32, g.input(:i, [4], dtype: :int32).sum.dtype
      assert_equal :float32, g.input(:j, [4], dtype: :int32).mean.dtype, "mean は要素数で割るので float"
      assert_equal :int32, x.argmax.dtype, "argmax が返すのは添字"
      assert_equal [1], x.argmax.shape
      # A reduction's result is usable as a value directly
      assert_equal [1], (x.sum * 2.0).shape
    end
  end

  def test_reductions_take_part_in_cse
    Tilework::Graph.build do |g|
      x = g.input(:x, [4])
      assert_equal x.sum.node.id, x.sum.node.id
      refute_equal x.sum.node.id, x.mean.node.id
      refute_equal x.max.node.id, x.argmax.node.id
    end
  end

  def test_rejects_shape_mismatch_between_tensors
    Tilework::Graph.build do |g|
      a = g.input(:a, [2, 3])
      b = g.input(:b, [3, 2])
      error = assert_raises(ArgumentError) { a + b }
      assert_includes error.message, "tensor shape mismatch"
    end
  end

  def test_rejects_dtypes_the_graph_layer_does_not_handle_yet
    Tilework::Graph.build do |g|
      error = assert_raises(Tilework::Error) { g.input(:x, [4], dtype: :float64) }
      assert_includes error.message, "float32 / int32 / float16"
    end
  end

  # Type checking (Builder) and allocatability (backend) are separate. The
  # latter is reported **all at once** at compile time — failing at the first
  # Buffer.new mid-run would not say which node is to blame.
  def test_dtypes_the_backend_cannot_allocate_are_rejected_at_compile_time
    builder = Tilework::Graph.build do |g|
      g.output(g.input(:x, [4], dtype: :float16) * 2.0)
    end
    error = assert_raises(Tilework::Error) { builder.compile }

    assert_includes error.message, ":float16"
    assert_includes error.message, "can only allocate"
  end

  # float16 is a storage type, as in Ops. It never becomes a compute type, so
  # the rule of dropping to float32 per operation is mirrored as is.
  def test_float16_follows_the_ops_storage_rules
    Tilework::Graph.build do |g|
      h = g.input(:h, [4], dtype: :float16)
      f = g.input(:f, [4])

      assert_equal :float16, (h * 2.0).dtype, "半精度とスカラーは半精度のまま"
      assert_equal :float16, (h + h).dtype
      assert_equal :float32, (h + f).dtype, "片方が float32 なら広がる"
      assert_equal :float16, h.exp.dtype, "単項数学関数も半精度を保つ"
      assert_equal :float32, h.sum.dtype, "累算は半精度でやらない"
      assert_equal :int32, h.eq(h).dtype
    end
  end

  def test_bfloat16_follows_the_ops_storage_rules_and_matmul_output_contract
    graph = Tilework::Graph.build do |g|
      x = g.input(:x, [2, 2], dtype: :bfloat16)
      assert_equal :bfloat16, (x * 2.0).dtype
      assert_equal :bfloat16, (x + x).dtype
      assert_equal :float32, x.sum.dtype
      assert_equal :float32, x.matmul(x).dtype
      assert_equal :bfloat16, x.matmul(x, out_dtype: :bfloat16).dtype
      g.output(x)
    end
    assert graph
  end

  def test_bfloat16_training_fails_with_an_actionable_message
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build(differentiable: true) do |g|
        x = g.input(:x, [2], dtype: :bfloat16)
        g.grad(x.cast(:float32).sum, wrt: x)
      end
    end
    assert_match(/bfloat16 training is not supported/, error.message)
    assert_match(/:float32/, error.message)
  end

  # --- CSE ---

  def test_identical_expressions_collapse
    Tilework::Graph.build do |g|
      x = g.input(:x, [4])
      assert_equal (x * 2.0).node.id, (x * 2.0).node.id
      refute_equal (x * 2.0).node.id, (x * 3.0).node.id
      # Different scalar types make different nodes (2 and 2.0 promote differently in Ops)
      refute_equal (x * 2).node.id, (x * 2.0).node.id
    end
  end

  def test_state_version_is_part_of_the_value_identity
    Tilework::Graph.build do |g|
      s0 = g.state(f32([2, 2]), name: :s)
      before_a = g.read(s0, 0)
      before_b = g.read(s0, 0)
      s1 = g.write_into(s0, at: 1, value: g.param(f32([1, 2])))
      after = g.read(s1, 0)

      assert_equal before_a.node.id, before_b.node.id, "同じ版の同じ領域は 1 つに畳んでよい"
      refute_equal before_a.node.id, after.node.id, "書き込みを挟んだ読みは別物"
      g.output(after)
    end
  end

  # --- Batched matmul and concat ---

  def test_infers_batched_matmul_shape
    Tilework::Graph.build do |g|
      a = g.input(:a, [3, 2, 4])
      b = g.param(f32([3, 4, 5]))
      assert_equal [3, 2, 5], a.matmul(b).shape
      assert_equal :batched, a.matmul(b).node.attrs[:mode]
      assert_equal :plain, g.input(:m, [2, 4]).matmul(g.param(f32([4, 5]))).node.attrs[:mode]
    end
  end

  def test_rejects_batched_matmul_shape_mismatches
    Tilework::Graph.build do |g|
      a = g.input(:a, [3, 2, 4])
      error = assert_raises(ArgumentError) { a.matmul(g.param(f32([2, 4, 5]))) }
      assert_includes error.message, "same batch"

      error = assert_raises(ArgumentError) { a.matmul(g.param(f32([3, 5, 5]))) }
      assert_includes error.message, "matmul shape mismatch"

      # Mixed ranks are not accepted yet (write shared weights folded to 2D).
      error = assert_raises(ArgumentError) { a.matmul(g.param(f32([4, 5]))) }
      assert_includes error.message, "[b, m, k] x [b, k, n]"
    end
  end

  def test_concat_extends_the_outermost_dimension
    Tilework::Graph.build do |g|
      a = g.input(:a, [2, 3])
      b = g.input(:b, [4, 3])
      assert_equal [6, 3], g.concat(a, b).shape
      assert_equal [8, 3], g.concat(a, b, a).shape
      assert_equal [6], g.concat(g.input(:u, [2]), g.input(:v, [4])).shape
    end
  end

  def test_concat_rejects_mismatched_trailing_dims_or_dtypes
    Tilework::Graph.build do |g|
      a = g.input(:a, [2, 3])
      error = assert_raises(ArgumentError) { g.concat(a, g.input(:b, [2, 4])) }
      assert_includes error.message, "the same dims outside axis 0"

      error = assert_raises(ArgumentError) { g.concat(a, g.input(:c, [2, 3], dtype: :int32)) }
      assert_includes error.message, "the same dims outside axis 0"

      assert_raises(ArgumentError) { g.concat(a) }
    end
  end

  # --- Built-in NN nodes ---

  def test_nn_functions_record_a_node_and_infer_its_shape
    Tilework::Graph.build do |g|
      x = g.input(:x, [4, 6])
      assert_equal [4, 6], Tilework::NN.softmax(x).shape
      assert_equal [4, 6], Tilework::NN.gelu(x).shape
      assert_equal [6, 4], Tilework::NN.transpose(x).shape
      assert_equal :float32, Tilework::NN.rmsnorm(x).dtype
      assert_equal :nn, Tilework::NN.softmax(x).node.op
    end
  end

  def test_nn_attention_infers_its_output_shape
    Tilework::Graph.build do |g|
      q = g.input(:q, [3, 8])
      k = g.input(:k, [5, 8])
      v = g.input(:v, [5, 8])
      assert_equal [3, 8], Tilework::NN.multi_head_attention(q, k, v, heads: 2).shape
    end
  end

  # Weights can be passed as Buffers (no need to rewrap them with g.param).
  def test_nn_weights_given_as_buffers_are_lifted_to_constants
    gamma = f32([6])
    Tilework::Graph.build do |g|
      x = g.input(:x, [4, 6])
      node = Tilework::NN.rmsnorm(x, gamma: gamma).node

      assert_equal 2, node.inputs.length
      assert_equal :const, node.inputs[1].op
      assert_equal [:gamma], node.attrs[:keys]
      # The same Buffer becomes the same const node (one even if 32 layers share it).
      assert_equal node.inputs[1].id, g.param(gamma).node.id
    end
  end

  def test_nn_nodes_take_part_in_cse
    Tilework::Graph.build do |g|
      x = g.input(:x, [4, 6])
      assert_equal Tilework::NN.softmax(x).node.id, Tilework::NN.softmax(x).node.id
      refute_equal Tilework::NN.silu(x).node.id, Tilework::NN.relu(x).node.id
      refute_equal Tilework::NN.rmsnorm(x, eps: 1e-6).node.id,
                   Tilework::NN.rmsnorm(x, eps: 1e-5).node.id
    end
  end

  # Validation goes through the same function as eager, so it fails here before touching a device.
  def test_nn_shape_rules_are_checked_while_recording
    Tilework::Graph.build do |g|
      x = g.input(:x, [24])
      error = assert_raises(ArgumentError) { Tilework::NN.softmax(x) }
      assert_includes error.message, "2D"

      matrix = g.input(:m, [4, 6])
      error = assert_raises(ArgumentError) { Tilework::NN.rmsnorm(matrix, gamma: g.param(f32([5]))) }
      assert_includes error.message, "gamma must have 6 elements"
    end
  end

  # NN frees its intermediates mid-call; the context now takes over that free
  # (#86), so it stays queued.
  def test_nn_nodes_run_asynchronously_because_their_frees_are_deferred
    plan = Tilework::Graph.build do |g|
      g.output(Tilework::NN.silu(g.input(:x, [4, 6])))
    end.compile
    plan.run(x: f32([4, 6]))

    assert_equal [false], @bridge.dispatches.map { |d| d[:wait] }.uniq
    assert_equal 0, @bridge.synchronize_count, "委譲ライブラリを使わない NN に囲いは要らない"
  end

  # Outputs come from the pool too, so rerunning the same plan does not keep allocating.
  def test_nn_outputs_are_pooled_like_any_other_intermediate
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [4, 6])
      g.output(Tilework::NN.silu(x) * 2.0)
    end.compile
    input = f32([4, 6])

    plan.run(x: input)
    first = plan.pool_stats
    plan.run(x: input)
    second = plan.pool_stats

    assert_equal 1, second[:allocated] - first[:allocated], "2 回目に増える確保は出力の 1 本だけ"
  end

  # --- Linearity of mutable state ---

  def test_a_version_may_not_be_written_twice_by_overlapping_regions
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build do |g|
        s0 = g.state(f32([2, 2]), name: :kv)
        g.write_into(s0, at: 0, value: g.param(f32([1, 2])))
        g.write_into(s0, at: 0, value: g.param(f32([1, 2])))
      end
    end
    assert_includes error.message, "written twice"
  end

  def test_disjoint_writes_fork_and_must_be_merged
    unmerged = assert_raises(Tilework::Error) do
      Tilework::Graph.build do |g|
        s0 = g.state(f32([2, 4]), name: :out)
        g.write_into(s0, at: s0.axis(1, 0...2), value: g.input(:l, [2, 2]))
        g.write_into(s0, at: s0.axis(1, 2...4), value: g.input(:r, [2, 2]))
      end.compile
    end
    assert_includes unmerged.message, "unmerged versions"

    plan = Tilework::Graph.build do |g|
      s0 = g.state(f32([2, 4]), name: :out)
      a = g.write_into(s0, at: s0.axis(1, 0...2), value: g.input(:l, [2, 2]))
      b = g.write_into(s0, at: s0.axis(1, 2...4), value: g.input(:r, [2, 2]))
      g.output(g.read(g.merge(a, b)))
    end.compile
    assert_equal 2, plan.order.count { |node| node.op == :state_write }
  end

  def test_rejects_a_value_whose_size_does_not_match_the_region
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build do |g|
        s0 = g.state(f32([4, 2]), name: :kv)
        g.write_into(s0, at: 0, value: g.input(:k, [2, 2]))
      end
    end
    assert_includes error.message, "covers"
  end

  # --- Random numbers and seeds (#19 §6) ---

  # The same expression must draw different values each time, so draws are exempt from CSE.
  def test_each_draw_is_its_own_node
    Tilework::Graph.build do |g|
      a = g.random([4])
      b = g.random([4])
      refute_equal a.node.id, b.node.id
      assert_equal [0, 1], [a.node.attrs[:index], b.node.attrs[:index]]
      assert_equal [4], g.randn([4], stddev: 2.0).shape
      assert_equal 3, g.random_draws
    end
  end

  def test_a_plan_that_draws_requires_a_seed
    plan = Tilework::Graph.build { |g| g.output(g.random([4])) }.compile
    error = assert_raises(Tilework::Error) { plan.run }

    assert_includes error.message, "needs a seed"
    assert_equal 1, plan.random_draws
  end

  def test_seed_is_reserved_as_an_input_name
    Tilework::Graph.build do |g|
      error = assert_raises(Tilework::Error) { g.input(:seed, [4]) }
      assert_includes error.message, "reserved"
    end
  end

  # A delegated RNG runs outside our queue, so it runs synchronously.
  # (MockBridge has no random_available?, so portable mode, which takes the
  # dialect path, is needed for a kernel to appear.)
  def test_random_nodes_run_synchronously
    previous = Tilework::Random.mode
    Tilework::Random.portable!
    plan = Tilework::Graph.build { |g| g.output(g.random([64]) * 2.0) }.compile
    plan.run(seed: 1)

    waits = @bridge.dispatches.map { |d| d[:wait] }
    assert_equal [true, false], waits, "RNG は待つ。後続の掛け算は積んだまま流す"
  ensure
    Tilework::Random.mode = previous
  end

  # --- Output placement (#19 §3) ---

  # Making the write target the producer's output removes write_into's copy.
  def test_a_write_places_its_value_directly_into_the_state
    state = f32([4, 2])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :kv)
      g.write_into(s0, at: 0, value: g.input(:x, [1, 2]) * 2.0)
    end.compile
    plan.run(x: f32([1, 2]))

    assert_equal 1, @bridge.dispatches.length, "掛け算 1 本だけ。コピーは消える"
  end

  # Nodes that cannot choose their output (allocate several inside / opaque) keep the copy.
  def test_opaque_producers_still_copy
    state = f32([4, 2])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :kv)
      value = g.opaque(g.input(:x, [1, 2]), shape: [1, 2]) { |b| b * 2.0 }
      g.write_into(s0, at: 0, value: value)
    end.compile
    plan.run(x: f32([1, 2]))

    assert_equal 2, @bridge.dispatches.length, "ブロックの掛け算 + コピー"
  end

  # Placing a value also read elsewhere on the state would let a later write corrupt that reader.
  def test_a_value_read_elsewhere_is_not_placed
    state = f32([4, 2])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :kv)
      value = g.input(:x, [1, 2]) * 2.0
      g.write_into(s0, at: 0, value: value)
      g.output(value + 1.0)
    end.compile
    plan.run(x: f32([1, 2]))

    assert_equal 3, @bridge.dispatches.length, "掛け算 + 足し算 + コピー"
  end

  # A value derived from the same state cannot be placed (it would read and write one region at once).
  def test_a_value_derived_from_the_same_state_is_not_placed
    state = f32([4, 2])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :kv)
      g.write_into(s0, at: 0, value: g.read(s0, 1) * 2.0)
    end.compile
    plan.run

    assert_equal 2, @bridge.dispatches.length, "掛け算 + コピー"
  end

  # --- Ordering (WAR) ---

  # Version edges give only RAW. Check by execution order that version N+1's
  # write does not run while version N's readers are still reading (the
  # declaration order alone would break).
  def test_a_write_is_ordered_after_the_readers_of_the_version_it_overwrites
    plan = Tilework::Graph.build do |g|
      s0 = g.state(f32([2, 2]), name: :s)
      r = g.read(s0, 0)
      g.write_into(s0, at: 0, value: g.param(f32([1, 2])))
      g.output(r * 1.0)
    end.compile

    ops = plan.order.map(&:op)
    assert_operator ops.index(:binary), :<, ops.index(:state_write),
                    "読みの消費者より前に書き込みが走ってはいけない:\n#{plan.explain}"
    # Confirms the premise: in declaration order (= node id order) the write comes first.
    write = plan.order.find { |node| node.op == :state_write }
    consumer = plan.order.find { |node| node.op == :binary }
    assert_operator write.id, :<, consumer.id
  end

  def test_reading_a_version_that_is_later_overwritten_and_using_it_after_is_rejected
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build do |g|
        s0 = g.state(f32([2, 2]), name: :s)
        stale = g.read(s0, 0)
        s1 = g.write_into(s0, at: 0, value: g.param(f32([1, 2])))
        g.output(stale + g.read(s1, 0))
      end.compile
    end
    assert_includes error.message, "view of the same memory"
  end

  # Even in a plan that only writes and never reads, the write runs (the final
  # version is a root of the plan).
  def test_writes_run_even_when_nothing_reads_them
    plan = Tilework::Graph.build do |g|
      s0 = g.state(f32([4, 2]), name: :kv)
      g.write_into(s0, at: 0, value: g.input(:k, [1, 2]))
    end.compile
    assert_equal 1, plan.order.count { |node| node.op == :state_write }
  end

  # --- Region overlap ---

  def test_region_disjointness_is_exact_for_static_regions
    shape = [2, 4]
    left = Tilework::Graph::Region.axis(shape, 1, 0...2)
    right = Tilework::Graph::Region.axis(shape, 1, 2...4)
    overlap = Tilework::Graph::Region.axis(shape, 1, 1...3)

    assert left.disjoint?(right), "同じ stride で位相が重ならなければ素"
    refute left.disjoint?(overlap)
    assert Tilework::Graph::Region.rows(shape, 0).disjoint?(Tilework::Graph::Region.rows(shape, 1))
    refute Tilework::Graph::Region.rows(shape, 0..1).disjoint?(Tilework::Graph::Region.rows(shape, 1))
  end

  def test_runtime_offsets_fall_back_to_the_conservative_answer
    shape = [4, 2]
    dynamic = Tilework::Graph::Region.rows(shape, Tilework::Graph::Index.new(:pos, 1))
    static = Tilework::Graph::Region.rows(shape, 0)

    refute dynamic.static?
    refute dynamic.disjoint?(static), "実行時に決まる位置は「重なるかもしれない」と答える"
    refute static.disjoint?(dynamic)
    assert_equal 2, dynamic.resolve(pos: 1).offset
  end

  def test_regions_are_bounds_checked
    error = assert_raises(ArgumentError) { Tilework::Graph::Region.rows([4, 2], 4) }
    assert_includes error.message, "out of bounds"

    dynamic = Tilework::Graph::Region.rows([4, 2], Tilework::Graph::Index.new(:pos, 1))
    assert_raises(Tilework::Error) { dynamic.resolve(pos: 4) }
    assert_raises(Tilework::Error) { dynamic.resolve({}) }
  end

  # --- Async launch and buffer reuse (#17 Phase 1) ---

  def chain_plan(steps = 5)
    Tilework::Graph.build do |g|
      value = g.input(:x, [64])
      steps.times { value = (value * 1.5) + 0.5 }
      g.output(value)
    end.compile
  end

  def test_nodes_are_launched_without_waiting
    chain_plan.run(x: f32([64]))

    assert_equal [false], @bridge.dispatches.map { |d| d[:wait] }.uniq
    assert_equal 0, @bridge.synchronize_count, "プランの末尾でも待たない"
  end

  def test_async_can_be_turned_off_for_comparison
    Tilework::Graph.async = false
    plan = chain_plan
    refute plan.async
    plan.run(x: f32([64]))
    assert_equal [true], @bridge.dispatches.map { |d| d[:wait] }.uniq
  ensure
    Tilework::Graph.async = true
  end

  # Intermediates return to the pool rather than being freed, so from the
  # second run on only the outputs leaving the plan are newly allocated.
  # (Measured with fusion off — fusion removes the intermediates themselves,
  # hiding the pool's own effect.)
  def test_intermediates_are_reused_between_runs
    Tilework::Graph.fusion = false
    plan = chain_plan
    input = f32([64])

    plan.run(x: input)
    first = plan.pool_stats
    plan.run(x: input)
    second = plan.pool_stats

    assert_equal 1, second[:allocated] - first[:allocated], "2 回目に増える確保は出力の 1 本だけ"
    assert_operator second[:reused], :>, first[:reused]
  ensure
    Tilework::Graph.fusion = true
  end

  # --- Elementwise fusion (#17 Phase 2) ---

  def test_a_chain_of_elementwise_ops_becomes_one_kernel
    plan = chain_plan
    plan.run(x: f32([64]))

    assert_equal 1, @bridge.dispatches.length, "5 段の連鎖はカーネル 1 本になる"
    assert_equal({ clusters: 1, absorbed: 9, nodes: 2 }, plan.fusion_stats)
    assert_includes plan.explain, "(fused)"
  end

  def test_fusion_can_be_turned_off_for_comparison
    Tilework::Graph.fusion = false
    plan = chain_plan
    plan.run(x: f32([64]))

    assert_equal 10, @bridge.dispatches.length
    assert_equal({ clusters: 0, absorbed: 0, nodes: 11 }, plan.fusion_stats)
  ensure
    Tilework::Graph.fusion = true
  end

  # A value read from two places is not absorbed: the expression would be
  # duplicated and computed twice, and it stays materialized anyway.
  def test_a_value_read_twice_is_not_absorbed
    plan = Tilework::Graph.build do |g|
      shared = g.input(:x, [64]) * 2.0
      g.output(shared + shared.tanh)
    end.compile
    plan.run(x: f32([64]))

    assert_equal 2, @bridge.dispatches.length, "共有された値は実体化される"
  end

  def test_plan_outputs_are_never_absorbed
    plan = Tilework::Graph.build do |g|
      half = g.input(:x, [64]) * 2.0
      g.output(half)
      g.output(half.tanh + 1.0)
    end.compile
    plan.run(x: f32([64]))

    assert_equal 2, @bridge.dispatches.length
  end

  # Nodes named by ordering (WAR) edges are not absorbed: that would move "the
  # moment it finished" to the cluster root and reorder it against state writes.
  def test_nodes_named_by_order_edges_are_not_absorbed
    state = f32([2, 2])
    plan = Tilework::Graph.build do |g|
      s0 = g.state(state, name: :s)
      stale = g.read(s0, 0)
      g.write_into(s0, at: 0, value: g.param(f32([1, 2])))
      g.output((stale * 1.0).tanh)
    end.compile

    assert_equal 0, plan.fusion_stats[:absorbed], "読みの直接の消費者は融合の対象外"
  end

  # With half-precision intermediates, values change between fused and unfused
  # unless rounding is reproduced. (No float16-capable backend is at hand, so
  # the decision is checked directly.)
  def test_float16_intermediates_are_not_absorbed
    node = ->(dtype) { Tilework::Graph::Node.new(1, :unary, [], { fn: :exp }, dtype, [4]) }
    consumers = Hash.new { |h, k| h[k] = [nil] }
    args = [node.call(:float32), {}, consumers, {}]

    assert Tilework::Graph::Fusion.absorbable?(node.call(:float32), *args)
    refute Tilework::Graph::Fusion.absorbable?(node.call(:float16), *args)
  end

  def test_the_pool_can_be_turned_off
    Tilework::Graph.buffer_pool = false
    plan = chain_plan
    plan.run(x: f32([64]))
    assert_nil plan.pool_stats
  ensure
    Tilework::Graph.buffer_pool = true
  end

  # Wait on both sides of a node that leaves our execution path (a user block).
  def test_opaque_nodes_are_fenced
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [64])
      g.output(g.opaque(x * 2.0, shape: [64]) { |buffer| buffer })
    end.compile
    plan.run(x: f32([64]))

    assert_equal 2, @bridge.synchronize_count, "前後で 1 回ずつ"
  end

  # fenced: false is the caller's declaration that the block stays inside our
  # queue, so the fence is removed and it joins its neighbors' submission (#86).
  def test_opaque_can_decline_the_fence
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [64])
      g.output(g.opaque(x * 2.0, shape: [64], fenced: false) { |buffer| buffer + 1.0 })
    end.compile
    plan.run(x: f32([64]))

    assert_equal 0, @bridge.synchronize_count
    assert_equal [false], @bridge.dispatches.map { |d| d[:wait] }.uniq
    assert_equal 0, plan.sync_stats[:synchronous]
  end

  # A value a fenced: false block allocated outside may still be read by running
  # launches, so it is adopted into the pool instead of freed — without growing it.
  def test_unfenced_opaque_buffers_are_adopted_instead_of_freed
    plan = Tilework::Graph.build do |g|
      value = g.opaque(g.input(:x, [64]), shape: [64], fenced: false) do |_buffer|
        Tilework::Buffer.new(:float32, 64)
      end
      g.output(value + 1.0)
    end.compile
    input = f32([64])

    3.times { plan.run(x: input) }
    steady = plan.pool_stats[:pooled]
    3.times { plan.run(x: input) }

    assert_equal steady, plan.pool_stats[:pooled]
  end

  def test_opaque_buffers_allocated_outside_the_pool_are_freed_not_adopted
    plan = Tilework::Graph.build do |g|
      value = g.opaque(g.input(:x, [64]), shape: [64]) do |_buffer|
        Tilework::Buffer.new(:float32, 64)
      end
      g.output(value + 1.0)
    end.compile
    input = f32([64])

    3.times { plan.run(x: input) }
    steady = plan.pool_stats[:pooled]
    3.times { plan.run(x: input) }

    assert_equal steady, plan.pool_stats[:pooled],
                 "opaque が外で確保した値をプールへ混ぜると再利用されず増え続ける"
  end

  # A reduction frees its own partials internally, but the context takes over
  # that free (#86), so it stays queued. Partials are borrowed from the pool too,
  # so no slot is left that is returned but never borrowed.
  def test_reductions_run_asynchronously_because_their_partials_return_to_the_pool
    plan = Tilework::Graph.build { |g| g.output(g.input(:x, [64]).sum) }.compile
    plan.run(x: f32([64]))

    assert_equal [false], @bridge.dispatches.map { |d| d[:wait] }.uniq
    assert_equal 0, @bridge.synchronize_count
  end

  # The final buffer of a whole-buffer reduction is borrowed from the pool too.
  # Partials are Ops' own temporaries, but the result is an ordinary node output
  # the Plan returns to the pool; a Buffer.new across that boundary would add a
  # returned-but-never-borrowed slot every iteration.
  def test_whole_reduction_outputs_come_from_the_pool
    %i[sum min max mean argmin argmax].each do |kind|
      plan = Tilework::Graph.build do |g|
        reduced = g.input(:x, [64]).public_send(kind)
        g.output(reduced + 1)
      end.compile
      input = f32([64])

      3.times { plan.run(x: input) }
      steady = plan.pool_stats[:pooled]
      3.times { plan.run(x: input) }

      assert_equal steady, plan.pool_stats[:pooled],
                   "#{kind} の最終結果がプールを迂回すると実行のたびにプールが増える"
    end
  end

  def test_internal_nodes_cannot_return_an_unpooled_intermediate
    plan = Tilework::Graph.build do |g|
      g.output(g.random([64]) + 1.0)
    end.compile
    # Directly reproduces a future implementation mistakenly returning Buffer.new.
    plan.define_singleton_method(:run_random) do |node, _inputs, _bindings|
      Tilework::Buffer.new(node.dtype, node.length, shape: node.shape)
    end

    error = assert_raises(Tilework::Error) { plan.run(seed: 1) }
    assert_includes error.message, ":random"
    assert_includes error.message, "Ops.allocate_output"
  end

  # Delegated GEMMs take their outputs from the pool too.
  #
  # The plan **returns intermediates to the pool instead of freeing them**, so
  # a node allocating around the pool adds one slot per run. It only happens on
  # backends with gemm_library, so :cpu cannot show it.
  class GemmBridge < MockBridge
    def gemm_available? = true
    def gemm_library_name = :fake
    def gemm_supported_dtypes = %i[float32 bfloat16]

    # Values are not checked (only whether they come from the pool).
    def gemm(_a, _b, _out, _m, _k, _n, _tf32, _code) = nil
  end

  def test_delegated_gemm_output_comes_from_the_pool
    Tilework.bridge = GemmBridge.new
    plan = Tilework::Graph.build do |g|
      w = g.param(f32([2, 2]))
      # Two chained GEMMs: with one, the growth cancels against the single exit and is invisible.
      g.output((g.input(:a, [2, 2]).matmul(w).matmul(w)) * 2.0)
    end.compile

    3.times { plan.run(a: f32([2, 2])) }
    steady = plan.pool_stats[:pooled]
    3.times { plan.run(a: f32([2, 2])) }

    assert_equal steady, plan.pool_stats[:pooled],
                 "プールを迂回して確保するノードがあると、実行のたびにプールが増える"
  end

  # A delegated GEMM leaves our queue, so it waits on both sides — unless the
  # bridge declares that its GEMM is enqueued in order with its kernels.
  class OrderedGemmBridge < GemmBridge
    def ordered_gemm? = true
  end

  def test_delegated_gemm_is_fenced_unless_the_bridge_declares_it_ordered
    stats = lambda do |bridge|
      Tilework.bridge = bridge
      # A :custom backend reports no delegated libraries at all.
      Tilework.instance_variable_set(:@backend, :metal)
      plan = Tilework::Graph.build do |g|
        g.output(g.input(:a, [2, 2]).matmul(g.param(f32([2, 2]))) * 2.0)
      end.compile
      plan.run(a: f32([2, 2]))
      [plan.sync_stats, bridge.synchronize_count]
    end

    fenced, fenced_waits = stats.call(GemmBridge.new)
    assert_equal 2, fenced[:fences]
    assert_equal 1, fenced[:synchronous]
    assert_operator fenced_waits, :>=, 2

    ordered, ordered_waits = stats.call(OrderedGemmBridge.new)
    assert_equal 0, ordered[:fences]
    assert_equal 0, ordered[:synchronous]
    assert_equal 0, ordered_waits
  end

  def test_low_precision_matmul_cast_output_comes_from_the_pool
    Tilework.bridge = GemmBridge.new
    Tilework.instance_variable_set(:@backend, :metal)
    matrix = Tilework::Buffer.new(:bfloat16, 4, shape: [2, 2])
    plan = Tilework::Graph.build do |g|
      product = g.input(:a, [2, 2], dtype: :bfloat16)
                 .matmul(g.param(matrix), out_dtype: :bfloat16)
      g.output(product + 1.0)
    end.compile

    3.times { plan.run(a: matrix) }
    steady = plan.pool_stats[:pooled]
    3.times { plan.run(a: matrix) }

    assert_equal steady, plan.pool_stats[:pooled],
                 "matmul の dtype 変換がプールを迂回すると実行のたびにプールが増える"
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  # --- Plan presentation ---

  def test_plan_reports_its_inputs_and_reads_are_free
    plan = Tilework::Graph.build do |g|
      s0 = g.state(f32([4, 2]), name: :kv)
      x = g.input(:x, [2, 2])
      g.output(g.read(s0, 0..1) + x)
    end.compile

    assert_equal [:x], plan.input_names
    assert_equal 1, plan.states.length
    assert_includes plan.explain, "read kv v0"
  end

  def test_strict_state_mode_is_fixed_at_compile_time
    Tilework::Graph.strict_state = :copy
    plan = Tilework::Graph.build { |g| g.output(g.read(g.state(f32([2, 2])))) }.compile
    assert_equal :copy, plan.strict_state
  ensure
    Tilework::Graph.strict_state = :view
  end

  def test_strict_state_rejects_unknown_modes
    assert_raises(ArgumentError) { Tilework::Graph.strict_state = :nope }
  end

  # --- Broadcasting and axis reductions (docs/graph-ir.md Phase B) ---

  # Binary operands of different shapes are stretched right-aligned before recording.
  def test_binary_operands_are_broadcast_right_aligned
    Tilework::Graph.build do |g|
      x = g.input(:x, [4, 3])
      bias = g.input(:bias, [3])          # a row vector across all rows
      col = g.input(:col, [4, 1])         # a column vector across all columns
      assert_equal [4, 3], (x + bias).shape
      assert_equal [4, 3], (x * col).shape
      assert_equal [4, 3], (bias + col).shape   # both stretch
      # Only the stretched side gets a node in between (matching shapes pass through).
      assert_equal :broadcast, (x + bias).node.inputs[1].op
      assert_equal :input, (x + bias).node.inputs[0].op
    end
  end

  def test_rejects_shapes_that_do_not_broadcast
    Tilework::Graph.build do |g|
      a = g.input(:a, [2, 3])
      b = g.input(:b, [3, 2])
      error = assert_raises(ArgumentError) { a + b }
      assert_includes error.message, "tensor shape mismatch"

      error = assert_raises(ArgumentError) { g.broadcast(a, [3, 3]) }
      assert_includes error.message, "cannot broadcast [2, 3] to [3, 3]"
      # It cannot shrink.
      assert_raises(ArgumentError) { g.broadcast(a, [3]) }
    end
  end

  def test_axis_reductions_drop_the_axis_unless_keepdims
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3, 4])
      assert_equal [2, 3], x.sum(axis: -1).shape
      assert_equal [2, 3, 1], x.sum(axis: 2, keepdims: true).shape
      assert_equal [3, 4], x.mean(axis: 0).shape
      assert_equal [2, 4], x.max(axis: 1).shape
      assert_equal [1], x.sum.shape                    # no axis means everything
      assert_equal [1], g.input(:v, [8]).sum(axis: 0).shape  # [1] once no dims remain

      # Type rules match whole-buffer reductions (mean is float32, indices int32).
      i = g.input(:i, [2, 3], dtype: :int32)
      assert_equal :int32, i.sum(axis: 1).dtype
      assert_equal :float32, i.mean(axis: 1).dtype
      assert_equal :int32, x.argmax(axis: 1).dtype
    end
  end

  def test_rejects_axes_outside_the_shape
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3])
      error = assert_raises(ArgumentError) { x.sum(axis: 2) }
      assert_includes error.message, "axis 2 is out of range for [2, 3]"
      assert_raises(ArgumentError) { x.sum(axis: -3) }
      assert_raises(TypeError) { x.sum(axis: "1") }
    end
  end

  # Different axes are different values; without axis and keepdims in the CSE key they would merge.
  def test_cse_distinguishes_reductions_by_axis
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3])
      assert_equal x.sum(axis: 1).node.id, x.sum(axis: 1).node.id
      refute_equal x.sum(axis: 0).node.id, x.sum(axis: 1).node.id
      refute_equal x.sum(axis: 1).node.id, x.sum(axis: 1, keepdims: true).node.id
      refute_equal x.sum(axis: 1).node.id, x.sum.node.id
    end
  end

  # Both stay queued (since #86). The difference is placement: a whole-buffer
  # reduction allocates inside its tree, so its output cannot be placed.
  def test_axis_reductions_are_placeable_unlike_whole_buffer_ones
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3])
      whole = x.sum.node
      partial = x.sum(axis: 1).node
      refute whole.spec.synchronous?(whole)
      refute partial.spec.synchronous?(partial)
      refute whole.spec.placeable?(whole)
      assert partial.spec.placeable?(partial)
    end
  end

  # --- cast / where / iota / reshape ---

  def test_cast_changes_the_dtype_and_is_a_no_op_when_it_already_matches
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3])
      assert_equal :int32, x.cast(:int32).dtype
      assert_equal [2, 3], x.cast(:int32).shape
      # A cast to the same type records no node.
      assert_same x.node, x.cast(:float32).node
      assert_raises(Tilework::Error) { x.cast(:float64) }
    end
  end

  def test_where_broadcasts_all_three_operands_and_takes_scalars
    Tilework::Graph.build do |g|
      x = g.input(:x, [4, 3])
      row = g.input(:row, [3])
      # Any of the condition or values may stretch.
      assert_equal [4, 3], g.where(x > 0, x, row).shape
      assert_equal [4, 3], g.where(row.eq(0), x, 0.0).shape
      assert_equal :float32, g.where(x > 0, x, 0.0).dtype
      # Both values may be scalars (building a mask); the shape comes from the condition.
      assert_equal :int32, g.where(x > 0, 1, 0).dtype
      assert_equal [4, 3], g.where(x > 0, 1, 0).shape
      assert_equal :float32, g.where(x > 0, 1.0, 0).dtype

      # The condition must be a Tensor.
      assert_raises(TypeError) { g.where(true, x, 0.0) }
    end
  end

  def test_iota_declares_its_own_shape_and_axis
    Tilework::Graph.build do |g|
      assert_equal [2, 3], g.iota([2, 3], axis: 0).shape
      assert_equal :int32, g.iota([2, 3], axis: 0).dtype
      assert_equal :float32, g.iota([4], axis: -1, dtype: :float32).dtype
      assert_raises(ArgumentError) { g.iota([2, 3], axis: 2) }
      assert_raises(Tilework::Error) { g.iota([2, 3], dtype: :float16) }
    end
  end

  def test_reshape_keeps_the_element_count_and_resolves_one_hole
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 6])
      assert_equal [12], x.reshape(12).shape
      assert_equal [3, 4], x.reshape(3, -1).shape
      assert_equal [2, 3, 2], x.reshape([2, 3, 2]).shape
      assert_same x.node, x.reshape(2, 6).node   # the same shape records no node

      error = assert_raises(ArgumentError) { x.reshape(5, 5) }
      assert_includes error.message, "cannot reshape [2, 6]"
      assert_raises(ArgumentError) { x.reshape(-1, -1) }
      assert_raises(ArgumentError) { x.reshape(5, -1) }
    end
  end

  # reshape is a view, so the original node still owns the memory and lifetime
  # must be counted there (miscounting returns memory still being read to the pool).
  def test_reshape_does_not_own_a_buffer
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 6])
      refute Tilework::Graph::Op.fetch(:reshape).owned?
      refute Tilework::Graph::Op.fetch(:reshape).placeable?(x.reshape(12).node)
    end
  end

  # --- Decomposing high-level ops (docs/graph-ir.md Tier 1) ---

  def with_lowering
    Tilework::Graph.lower_high_level = true
    yield
  ensure
    Tilework::Graph.lower_high_level = false
  end

  def test_lowering_replaces_nn_nodes_with_primitives
    plan = Tilework::Graph.build do |g|
      g.output(Tilework::NN.softmax(g.input(:x, [2, 4])))
    end.compile
    assert_equal 1, plan.nodes.count { |n| n.op == :nn }

    lowered = with_lowering do
      Tilework::Graph.build { |g| g.output(Tilework::NN.softmax(g.input(:x, [2, 4]))) }.compile
    end
    assert_equal 0, lowered.nodes.count { |n| n.op == :nn }
    assert_equal [2, 4], lowered.outputs.first.shape
    # The decomposition is made of reductions and broadcasts.
    assert_equal 1, lowered.nodes.count { |n| n.op == :reduce && n.attrs[:kind] == :max }
    assert lowered.nodes.any? { |n| n.op == :broadcast }
  end

  # The switch takes effect at record time, so one plan never mixes the two paths.
  def test_lowering_is_decided_while_recording
    builder = Tilework::Graph.build { |g| g.output(Tilework::NN.rmsnorm(g.input(:x, [2, 4]))) }
    with_lowering do
      # Switching after recording leaves the node :nn.
      assert_equal 1, builder.compile.nodes.count { |n| n.op == :nn }
    end
  end

  def test_lowering_keeps_the_declared_shape_and_dtype
    %i[relu sigmoid silu gelu].each do |fn|
      lowered = with_lowering do
        Tilework::Graph.build { |g| g.output(Tilework::NN.public_send(fn, g.input(:x, [2, 3]))) }.compile
      end
      assert_equal [2, 3], lowered.outputs.first.shape, fn.to_s
      assert_equal :float32, lowered.outputs.first.dtype, fn.to_s
    end
  end

  # Without a decomposition it fails rather than silently substituting another expression.
  def test_ops_without_a_decomposition_say_so
    error = with_lowering do
      assert_raises(Tilework::Error) do
        Tilework::Graph.build { |g| g.output(Tilework::NN.gelu(g.input(:x, [2, 3]), exact: true)) }
      end
    end
    assert_includes error.message, "no decomposition"

    # groupnorm is not in the table, so it passes through as :nn (folding
    # across channels is not a row-wise reduction).
    plan = with_lowering do
      Tilework::Graph.build do |g|
        g.output(Tilework::NN.groupnorm(g.input(:x, [1, 4, 2, 2]), groups: 2))
      end.compile
    end
    assert_equal 1, plan.nodes.count { |n| n.op == :nn }
  end

  def test_the_registry_rejects_a_duplicate_definition
    error = assert_raises(Tilework::Error) { Tilework::Graph::HighLevel.define(:softmax) { nil } }
    assert_includes error.message, "already defined"
  end

  # --- Folding broadcasts (docs/graph-ir.md Phase C) ---

  def materialized(plan, op)
    plan.explain.lines.count { |line| line.include?(op) && !line.include?("(fused)") }
  end

  # No kernel writing out the stretched value is needed when the readers fuse.
  def test_a_broadcast_is_folded_into_the_consumers_index
    plan = Tilework::Graph.build do |g|
      g.output(g.input(:x, [3, 4]) + g.param(f32([4])))
    end.compile

    assert_equal 1, plan.nodes.count { |n| n.op == :broadcast }
    assert_equal 0, materialized(plan, "broadcast")
    assert_equal 1, plan.fusion_stats[:clusters]
  end

  # It computes nothing, so it folds into any number of readers (no duplicated expression).
  def test_a_broadcast_with_many_readers_is_still_folded
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [3, 4])
      bias = g.param(f32([4]))
      3.times { |i| g.output((x * (i + 1.0)) + bias) }
    end.compile

    assert_equal 1, plan.nodes.count { |n| n.op == :broadcast }   # one, thanks to CSE
    assert_equal 0, materialized(plan, "broadcast")
  end

  # If a reader needs the real buffer (a reduction), it is materialized for that reader.
  def test_a_broadcast_read_by_a_reduction_is_materialized
    plan = Tilework::Graph.build do |g|
      wide = g.param(f32([4])).broadcast_to([3, 4])
      g.output(wide.sum(axis: 1))
    end.compile

    assert_equal 1, materialized(plan, "broadcast")
  end

  # The index expression itself. Getting a cross-dimension remap wrong swaps
  # values, so pin the generated expression's shape directly.
  def test_the_folded_index_expression_drops_the_stretched_dimensions
    fusion = Tilework::Graph::Fusion
    kernels = Tilework::Graph::Kernels
    assert_equal "i", fusion.index_expr(nil)
    # [4] -> [3, 4] rereads within a row
    assert_equal "(i % 4)", fusion.index_expr(kernels.broadcast_dims([4], [3, 4]))
    # [3, 1] -> [3, 4] reads the same row's value 4 times
    assert_equal "(i / 4)", fusion.index_expr(kernels.broadcast_dims([3, 1], [3, 4]))
    # [2, 1, 3] -> [2, 2, 3]
    assert_equal "(i % 3) + (i / 6) * 3", fusion.index_expr(kernels.broadcast_dims([2, 1, 3], [2, 2, 3]))
  end

  # --- Row-wise fusion (docs/graph-ir.md Phase C) ---

  def row_wise_plan(rows, cols)
    Tilework::Graph.build do |g|
      x = g.input(:x, [rows, cols])
      g.output(x / x.sum(axis: -1, keepdims: true))
    end.compile
  end

  # "Fold a row and spread it back over that row" becomes one kernel.
  def test_a_row_reduction_and_its_readers_become_one_kernel
    plan = row_wise_plan(256, 8)
    assert_equal 1, plan.fusion_stats[:clusters]
    assert_equal 0, materialized(plan, "sum axis")
    assert_equal 0, materialized(plan, "broadcast")
  end

  # One thread owns one row, so few rows means low parallelism; below the
  # threshold it stays on the elementwise path.
  def test_row_fusion_stays_off_when_there_are_too_few_rows
    plan = row_wise_plan(4, 8)
    assert_equal 1, materialized(plan, "sum axis")
  end

  def test_row_fusion_can_be_turned_off
    Tilework::Graph.row_fusion = false
    plan = row_wise_plan(256, 8)
    assert_equal 1, materialized(plan, "sum axis")
  ensure
    Tilework::Graph.row_fusion = true
  end

  # Elementwise ops before the reduction join the same cluster (rmsnorm's x * x).
  def test_the_expression_feeding_a_row_reduction_is_absorbed_too
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [256, 8])
      g.output(x / ((x * x).mean(axis: -1, keepdims: true) + 1e-6).sqrt)
    end.compile

    assert_equal 1, plan.fusion_stats[:clusters]
    assert_equal 0, materialized(plan, "mean axis")
    assert_equal 0, materialized(plan, "*")
  end

  # --- permute / slice / gather / pad / concat(axis:) (Phase D) ---

  def test_permute_reorders_the_shape_and_rejects_bad_axes
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3, 4])
      assert_equal [3, 2, 4], x.permute(1, 0, 2).shape
      assert_equal [4, 3, 2], x.permute([2, 1, 0]).shape
      assert_same x.node, x.permute(0, 1, 2).node          # identity records nothing
      assert_raises(ArgumentError) { x.permute(0, 1) }     # too few axes
      assert_raises(ArgumentError) { x.permute(0, 1, 1) }  # duplicate axis
    end
  end

  # permute is an index remap like broadcast, so fusion folds it into the
  # consumer's index.
  def test_permute_is_folded_into_the_consumer_like_a_broadcast
    plan = Tilework::Graph.build do |g|
      g.output(g.input(:x, [3, 4]).permute(1, 0) * 2.0)
    end.compile

    assert_equal 0, materialized(plan, "permute")
    assert Tilework::Graph::Op.fetch(:permute).index_map?
  end

  def test_slice_takes_the_outermost_dimension_only
    Tilework::Graph.build do |g|
      x = g.input(:x, [4, 3])
      assert_equal [2, 3], x.slice(1..2).shape
      assert_equal [1, 3], x.slice(0).shape
      assert_equal [2, 3], x.slice(-2..).shape
      assert_same x.node, x.slice(0..3).node               # the whole range records nothing
      assert_raises(ArgumentError) { x.slice(2..1) }
      assert_raises(ArgumentError) { x.slice(9) }
      assert_raises(Tilework::Error) { g.input(:v, [4]).slice(0..1) }
      refute Tilework::Graph::Op.fetch(:slice).owned?      # a view owns nothing
    end
  end

  def test_gather_shapes_follow_the_indices
    Tilework::Graph.build do |g|
      table = g.input(:t, [10, 4])
      ids = g.input(:i, [3], dtype: :int32)
      assert_equal [3, 4], g.gather(table, ids).shape
      assert_equal [10, 3], g.gather(table, ids, axis: 1).shape
      # Indices must be int32.
      assert_raises(Tilework::Error) { g.gather(table, g.input(:f, [3])) }
    end
  end

  def test_pad_grows_each_dimension_by_the_given_amount
    Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3])
      assert_equal [3, 6], g.pad(x, before: [1, 1], after: [0, 2]).shape
      assert_equal [4, 5], g.pad(x, before: 1).shape          # a number pads every dim
      assert_same x.node, g.pad(x, before: 0).node            # 0 records nothing
      assert_raises(ArgumentError) { g.pad(x, before: [1]) }
      assert_raises(ArgumentError) { g.pad(x, before: [-1, 0]) }
    end
  end

  def test_concat_takes_an_axis
    Tilework::Graph.build do |g|
      a = g.input(:a, [2, 3])
      b = g.input(:b, [2, 5])
      assert_equal [2, 8], g.concat(a, b, axis: 1).shape
      assert_equal [2, 8], g.concat(a, b, axis: -1).shape
      # Every axis but the concat axis must match.
      assert_raises(ArgumentError) { g.concat(a, b) }
    end
  end

  # --- conv2d and plan.requires (docs/graph-ir.md Phase E) ---

  # Pretend to be a backend with convolution. capabilities asks the bridge via
  # Conv.library, so adding those two methods selects the delegated path.
  class ConvBridge < MockBridge
    def conv2d_available? = true
    def conv_library_name = :mock_dnn
    def conv2d(*) = nil
  end

  # capabilities merges delegated_library_capabilities only while a backend is
  # set, so the name is swapped too (the same trick as module_test.rb).
  def with_conv_backend
    previous = Tilework.backend
    Tilework.bridge = ConvBridge.new
    Tilework.instance_variable_set(:@backend, :metal)
    yield
  ensure
    Tilework.instance_variable_set(:@backend, previous)
  end

  def conv_graph
    Tilework::Graph.build do |g|
      x = g.input(:x, [1, 2, 4, 4])
      g.output(Tilework::Conv.forward(x, g.param(f32([3, 2, 3, 3])), pad: 1))
    end
  end

  def test_conv2d_is_recorded_as_one_node_when_the_backend_has_convolution
    with_conv_backend do
      plan = conv_graph.compile

      assert_equal 1, plan.nodes.count { |n| n.op == :conv2d }
      assert_equal [1, 3, 4, 4], plan.outputs.first.shape
      # requires answers against the current capabilities (not a record from compile time).
      assert_equal [:conv], plan.requires.map(&:capability)
      assert plan.requires.first.satisfied?
      assert_includes plan.explain, "conv2d pad [1, 1] stride [1, 1]"
    end
  end

  # On a backend without convolution it is decomposed into primitives at record
  # time — the first concrete case of "a graph makes it portable".
  def test_conv2d_is_lowered_when_the_backend_has_no_convolution
    plan = conv_graph.compile   # MockBridge has no convolution

    assert_equal 0, plan.nodes.count { |n| n.op == :conv2d }
    assert_empty plan.requires
    assert_equal [1, 3, 4, 4], plan.outputs.first.shape
    # The decomposition is im2col + matmul.
    assert plan.nodes.any? { |n| n.op == :gather }
    assert plan.nodes.any? { |n| n.op == :matmul }
  end

  # If the backend changes after recording, the requirement may become
  # unsatisfiable; it fails at compile, not after starting to run.
  def test_compile_refuses_a_graph_the_backend_cannot_run
    builder = with_conv_backend { conv_graph }   # record on a backend with convolution
    Tilework.bridge = MockBridge.new             # compile on one without

    error = assert_raises(Tilework::Error) { builder.compile }
    assert_includes error.message, "conv"
    assert_includes error.message, "does not provide it"
  end

  def test_requires_is_empty_for_a_graph_that_needs_nothing_special
    plan = Tilework::Graph.build { |g| g.output(g.input(:x, [4]) * 2.0) }.compile
    assert_empty plan.requires
  end
end
