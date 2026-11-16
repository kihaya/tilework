# frozen_string_literal: true

require_relative "../test_helper"

# The **policy** of automatic offload (#17 Phase 4 / docs/graph-offload.md) on :cpu.
#
# That paging does not change values is checked on every backend by
# Tilework::Conformance's graph_offload_*; this file holds the scheduler's
# decisions themselves — how the budget is split, how many bytes cyclic Belady
# moves in steady state, failing before running. These tests look at
# **counters**, not values.
class GraphResidencyCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
    @plans = []
  end

  def teardown
    @plans.each(&:free_offload!)
    Tilework::Graph.offload = true
    Tilework::Graph.offload_fence_reuse = true
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  # An n x n weight that is not a view (reshape returns a view, which cannot be paged).
  def square(n) = Tilework[Array.new(n) { Array.new(n) { rand } }]

  # A plan that just multiplies by count weights in turn.
  def chain(weights, vram: nil)
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [4, 4])
      weights.each { |w| x = x.matmul(g.param(w)) }
      g.output(x)
    end.compile(vram: vram)
    @plans << plan
    plan
  end

  # A plan with a synchronous node (:opaque) in it, giving eviction fences
  # something to ride along with (#17 Phase 4c). A chain of only matmul or :nn
  # never synchronizes on :cpu (no gemm_library, so matmul is not foreign, and
  # :nn became asynchronous in #86). A user block may leave our queue, so
  # opaque still waits on both sides.
  def relu_chain(weights, vram: nil)
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [4, 4])
      weights.each do |w|
        product = x.matmul(g.param(w))
        x = g.opaque(product, shape: [4, 4], dtype: :float32) { |buffer| Tilework::NN.relu(buffer) }
      end
      g.output(x)
    end.compile(vram: vram)
    @plans << plan
    plan
  end

  def slot_bytes = 4 * 4 * 4 # a 4x4 float32

  # A dry run to read the budget's reserve from real values; 1 GB always fits.
  def probe(weights) = chain(weights).then { chain(weights, vram: 1 << 30).residency }

  def test_no_budget_means_no_residency_pass
    plan = chain(Array.new(3) { square(4) })

    assert_nil plan.residency
    assert_nil plan.offload_stats
  end

  # A loose budget is free: a report is produced but the machinery does not
  # exist (passing a budget alone must not change behavior).
  def test_a_budget_that_fits_pages_nothing
    weights = Array.new(3) { square(4) }
    plan = chain(weights, vram: 1 << 30)

    refute_nil plan.residency
    assert_equal 0, plan.residency.floor
    assert_equal 3, plan.residency.paged
    assert_nil plan.offload_stats, "nothing should be paged when the budget fits"
    plan.run(x: square(4))
  end

  # Steady state moves "the shortfall plus at most one slot". A sequential scan
  # needs one slot to load new weights into, so what carries over is budget
  # minus one slot rather than the full budget. The hand-written
  # Offload.pipeline streams **everything** every run; that is the gap
  # automation closes (5 slots vs 2–3 for 5 weights).
  def test_the_steady_state_moves_about_the_shortfall
    weights = Array.new(5) { square(4) }
    reserved = probe(weights).reserved
    plan = chain(weights, vram: reserved + (3 * slot_bytes))
    x = square(4)
    floor = plan.residency.floor

    plan.run(x: x) # first run is cold (everything resident, so it starts by evicting)
    moved = 4.times.map do
      before = plan.offload_stats[:bytes_in]
      plan.run(x: x)
      plan.offload_stats[:bytes_in] - before
    end

    assert_equal 2 * slot_bytes, floor
    assert_operator moved.min, :>=, floor
    assert_operator moved.max, :<=, floor + slot_bytes
    assert_operator moved.sum / moved.length, :<, weights.length * slot_bytes
  end

  # Stays within budget. A plan binding resident Buffers **starts over budget**,
  # so check that it sheds without waiting for a miss.
  def test_the_peak_never_exceeds_the_budget
    weights = Array.new(5) { square(4) }
    reserved = probe(weights).reserved
    budget = 2 * slot_bytes
    plan = chain(weights, vram: reserved + budget)

    2.times { plan.run(x: square(4)) }

    assert_operator plan.offload_stats[:peak_bytes], :<=, budget
  end

  # Eviction needs a fence (never buffer_free VRAM a pending kernel reads), but
  # only one per eviction **round**.
  def test_evictions_are_batched_behind_one_fence
    weights = Array.new(4) { square(4) }
    reserved = probe(weights).reserved
    plan = chain(weights, vram: reserved + (2 * slot_bytes))

    plan.run(x: square(4))
    stats = plan.offload_stats

    assert_operator stats[:added_fences] + stats[:free_fences], :<=, stats[:evictions]
    assert_operator stats[:added_fences] + stats[:free_fences], :>, 0
  end

  # Ride along with waits that nodes already do (#17 Phase 4c). On a chain with
  # :nn in it the extra waits drop to almost 0 (the one left is the run's entry
  # — the plan does not synchronize at the end, so the queue is assumed dirty
  # at first).
  def test_eviction_fences_ride_along_with_synchronous_nodes
    weights = Array.new(4) { square(4) }
    reserved = probe(weights).reserved
    budget = reserved + (2 * slot_bytes)
    x = square(4)

    Tilework::Graph.offload_fence_reuse = false
    without = relu_chain(weights, vram: budget)
    3.times { without.run(x: x) }
    without.free_offload!

    Tilework::Graph.offload_fence_reuse = true
    with = relu_chain(weights, vram: budget)
    3.times { with.run(x: x) }

    assert_operator with.offload_stats[:added_fences], :<, without.offload_stats[:added_fences]
    assert_operator with.offload_stats[:free_fences], :>, 0
    # Riding along **only removes waits**; it does not change what is dropped when.
    assert_equal without.offload_stats[:bytes_in], with.offload_stats[:bytes_in]
    assert_equal without.offload_stats[:evictions], with.offload_stats[:evictions]
  end

  # A weight with no file (the user bound a resident Buffer) is written out
  # once on its first eviction and only dropped after that. Writing back the
  # same contents again would be waste.
  def test_a_resident_weight_is_spilled_once
    weights = Array.new(4) { square(4) }
    reserved = probe(weights).reserved
    plan = chain(weights, vram: reserved + (2 * slot_bytes))
    x = square(4)

    4.times { plan.run(x: x) }

    assert_operator plan.offload_stats[:spills], :<=, weights.length
    assert_operator plan.offload_stats[:evictions], :>, plan.offload_stats[:spills]
  end

  # Fail before running when the budget is smaller than one weight.
  def test_a_budget_that_cannot_hold_one_weight_fails_at_compile
    weights = Array.new(3) { square(4) }
    reserved = probe(weights).reserved

    error = assert_raises(Tilework::Error) { chain(weights, vram: reserved + (slot_bytes / 2)) }
    assert_match(/largest single weight/, error.message)
  end

  def test_the_budget_must_be_a_positive_integer
    assert_raises(ArgumentError) { chain([square(4)], vram: -1) }
  end

  # Can be turned off (like the other switches, for measuring and bisecting).
  def test_the_kill_switch_disables_the_pass
    weights = Array.new(4) { square(4) }
    reserved = probe(weights).reserved
    Tilework::Graph.offload = false
    plan = chain(weights, vram: reserved + slot_bytes)

    assert_nil plan.residency
    assert_nil plan.offload_stats
  end

  # A view cannot move (restore! allocates a new handle, leaving views of the
  # old one dangling). It must be counted as pinned without breaking.
  def test_a_view_weight_is_pinned_rather_than_paged
    flat = Tilework[Array.new(16) { rand }]
    plan = chain([flat.reshape(4, 4)], vram: 1 << 30)

    assert_equal 0, plan.residency.paged
    assert_equal 1, plan.residency.resident_always
    assert_equal slot_bytes, plan.residency.resident_always_bytes
  end
end
