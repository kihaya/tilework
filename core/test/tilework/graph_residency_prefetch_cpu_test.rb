# frozen_string_literal: true

require_relative "../test_helper"

# The **policy** of prefetch (#17 Phase 4b / docs/graph-offload.md) on :cpu.
#
# The device async-transfer ABI (buffer_save / transfer_begin/poll/end) exists
# only on :cuda / :hip, so this wraps :cpu's CPUBridge with a minimal async
# ABI. Real concurrency (the bandwidth measured at depth 2–4 on Jetson) cannot
# be measured here; what can is scheduling correctness — several transfers in
# flight at once, the budget kept, values intact. Performance is for hardware
# (bench/offload_bench.rb).
class GraphResidencyPrefetchCPUTest < Minitest::Test
  # A thin wrapper adding transfer_begin/poll/end and offload_available? /
  # transfer_available? to CPUBridge. Transfers finish a little after they are
  # issued, so issuing several makes them overlap, and max_in_flight counts
  # how many were in flight at once.
  class AsyncMockBridge
    Token = Struct.new(:handle, :path, :bytesize, :offset, :ready_at)

    def initialize(inner)
      @inner = inner
      inner.methods(false).each { |name| define_singleton_method(name) { |*a, &b| inner.public_send(name, *a, &b) } }
      @in_flight = 0
      @max_in_flight = 0
      @started = 0
    end

    attr_reader :max_in_flight, :started

    def offload_available? = true
    def transfer_available? = true

    # Offload.write_file_native writes the header first, then calls this.
    # Recreating the file here would erase the header, so append the body only.
    def buffer_save(handle, path, bytesize, _offset)
      body = @inner.buffer_read(handle, bytesize, 0)
      File.open(path, "r+b") { |f| f.seek(4096); f.write(body) }
    end

    def buffer_load(handle, path, bytesize, offset)
      body = File.open(path, "rb") { |f| f.seek(4096); f.read(bytesize) }
      @inner.buffer_write(handle, body, offset)
    end

    def transfer_begin(handle, path, bytesize, offset, _direction)
      @started += 1
      @in_flight += 1
      @max_in_flight = [@max_in_flight, @in_flight].max
      Token.new(handle, path, bytesize, offset, Process.clock_gettime(Process::CLOCK_MONOTONIC) + 0.001)
    end

    def transfer_poll(token) = Process.clock_gettime(Process::CLOCK_MONOTONIC) >= token.ready_at ? 1 : 0

    def transfer_end(token)
      remaining = token.ready_at - Process.clock_gettime(Process::CLOCK_MONOTONIC)
      sleep(remaining) if remaining.positive?
      @in_flight -= 1
      buffer_load(token.handle, token.path, token.bytesize, token.offset)
    end
  end

  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
    @inner = Tilework.bridge
    @async = AsyncMockBridge.new(@inner)
    Tilework.bridge = @async
    Tilework.instance_variable_set(:@backend, :cpu)
    @plans = []
  end

  def teardown
    @plans.each(&:free_offload!)
    Tilework.bridge = @inner
    Tilework::Graph.offload = true
    Tilework::Graph.offload_fence_reuse = true
    Tilework::Graph.offload_prefetch = true
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  def square(n) = Tilework[Array.new(n) { Array.new(n) { rand } }]

  def relu_chain(weights, vram: nil)
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [8, 8])
      weights.each { |w| x = Tilework::NN.relu(x.matmul(g.param(w))) }
      g.output(x)
    end.compile(vram: vram)
    @plans << plan
    plan
  end

  def slot_bytes = 8 * 8 * 4 # an 8x8 float32

  def probe(weights) = relu_chain(weights).then { relu_chain(weights, vram: 1 << 30).residency }

  def test_async_is_available_through_the_mock
    assert Tilework::Offload.async?, "the mock bridge should make Offload.async? true"
  end

  def test_prefetch_issues_more_than_one_transfer_at_once
    weights = Array.new(10) { square(8) }
    reserved = probe(weights).reserved
    plan = relu_chain(weights, vram: reserved + (4 * slot_bytes))
    x = square(8)

    5.times { plan.run(x: x) }

    assert_operator @async.max_in_flight, :>, 1,
                     "prefetch should overlap transfers, not issue them one at a time"
  end

  def test_prefetch_off_never_calls_the_async_abi
    weights = Array.new(10) { square(8) }
    reserved = probe(weights).reserved
    Tilework::Graph.offload_prefetch = false
    plan = relu_chain(weights, vram: reserved + (4 * slot_bytes))
    x = square(8)

    5.times { plan.run(x: x) }

    assert_equal 0, @async.started, "with prefetch off, everything should go through the synchronous restore! path"
  end

  # Prefetch pays a different kind of trade-off from 4c's eviction ride-along.
  # That one spent bytes to save waits and was measured and **rejected**
  # (2304 -> 2816 bytes, fewer waits, nothing faster). This one spends bytes to
  # move them concurrently, and on Jetson (docs/graph-offload.md) concurrency
  # buys 3–4x the bandwidth, so the trade pays.
  #
  # The extra is **capped at PREFETCH_DEPTH slots** and does not grow with the
  # weight count (a flat +2 slots per run at 8/20/40 weights). This pins that
  # it never falls back to "everything every run" like Offload.pipeline.
  def test_prefetch_overshoot_is_bounded_by_the_prefetch_depth
    weights = Array.new(8) { square(8) }
    reserved = probe(weights).reserved
    budget = reserved + (3 * slot_bytes)
    x = square(8)
    runs = 6

    Tilework::Graph.offload_prefetch = false
    without = relu_chain(weights, vram: budget)
    runs.times { without.run(x: x) }
    baseline = without.offload_stats[:bytes_in]
    without.free_offload!

    Tilework::Graph.offload_prefetch = true
    with = relu_chain(weights, vram: budget)
    runs.times { with.run(x: x) }
    prefetched = with.offload_stats[:bytes_in]
    with.free_offload!

    overshoot_per_run = (prefetched - baseline) / runs.to_f
    assert_operator overshoot_per_run, :<=, Tilework::Graph::Residency::PREFETCH_DEPTH * slot_bytes
    # And it never goes back to "everything every run" (the price pipeline pays unknowingly).
    assert_operator prefetched / runs, :<, weights.length * slot_bytes
  end

  # Values stay intact. This is exactly where eviction and prefetch race — a
  # different decision about a weight while its transfer is still in flight.
  def test_prefetched_runs_stay_bit_identical
    weights = Array.new(10) { square(8) }
    x = square(8)
    want = relu_chain(weights).run(x: x).to_a

    reserved = probe(weights).reserved
    plan = relu_chain(weights, vram: reserved + (4 * slot_bytes))
    20.times { assert_equal want, plan.run(x: x).to_a }
  end

  # The budget holds including VRAM held by prefetches — restore_async!
  # allocates at issue time, so missing that would exceed the budget unnoticed
  # (docs/graph-offload.md, "VRAM is charged at issue time").
  def test_the_peak_never_exceeds_the_budget_with_prefetch
    weights = Array.new(10) { square(8) }
    reserved = probe(weights).reserved
    budget = 4 * slot_bytes
    plan = relu_chain(weights, vram: reserved + budget)
    x = square(8)

    8.times { plan.run(x: x) }

    assert_operator plan.offload_stats[:peak_bytes], :<=, budget
  end

  # Even at an extreme budget (exactly one weight) the cold path always
  # succeeds. If prefetches held on to VRAM, a later cold fetch could not make
  # room and would fail; check that guarantee holds.
  def test_the_tightest_possible_budget_still_runs
    weights = Array.new(6) { square(8) }
    reserved = probe(weights).reserved
    plan = relu_chain(weights, vram: reserved + slot_bytes)
    x = square(8)

    5.times { plan.run(x: x) }
    assert_operator plan.offload_stats[:peak_bytes], :<=, slot_bytes
  end

  # free_offload! collects outstanding transfers before deleting; otherwise a
  # Buffer holding VRAM and the bridge's worker would be left behind.
  def test_free_offload_drains_outstanding_prefetches
    weights = Array.new(10) { square(8) }
    reserved = probe(weights).reserved
    plan = relu_chain(weights, vram: reserved + (4 * slot_bytes))
    x = square(8)

    plan.run(x: x) # prefetches are issued here
    plan.free_offload!

    weights.each do |w|
      assert w.resident? || !w.freed?, "every weight should end up in a well-defined state, not stuck mid-transfer"
    end
  end
end
