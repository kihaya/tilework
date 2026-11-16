# frozen_string_literal: true

module Tilework
  module Graph
    # Pages weights (:const node buffers) in and out to fit a VRAM budget
    # (#17 Phase 4 / docs/graph-offload.md).
    #
    # Manual Offload (buffer.rb / runtime/offload.rb) is mechanism without
    # policy — which buffer, when, and how far ahead are all the caller's call,
    # and Offload.pipeline's depth: is chosen by hand. A plan knows the
    # execution order and every shape before touching the device, so it takes
    # over those decisions here.
    #
    # The budget is split statically at compile time:
    #
    #   pageable budget = vram - (inputs/outputs + state + peak intermediates + pinned consts)
    #
    # Eviction is **cyclic Belady** — drop whatever is used farthest in the
    # future. The execution order is static, so the optimal policy is
    # available; making it cyclic is the key, otherwise a decode loop drops
    # everything at the end of each run and reloads it at the start (a graph
    # that should fit thrashes).
    #
    # **The victim is chosen at run time.** The plan holds statically only
    # "what is needed at which index" and decides what to drop as it runs. The
    # 4-weights-in-3-slots example (docs/graph-offload.md) shows why: in steady
    # state the victim rotates from run to run, which a static index -> eviction
    # table cannot express. The next-use table is static, so the run-time
    # decision is a single O(pageable entries) scan.
    class Residency
      # The unit of paging: one :const node and its buffer.
      #
      #   uses      execution positions reading this weight (ascending); next_use wraps around
      #   has_file  whether an offload file exists already. If not, the first
      #             eviction writes it (offload!); otherwise VRAM is just dropped (evict!)
      Entry = Struct.new(:node_id, :buffer, :bytes, :uses, :has_file, :spilled) do
        def resident? = !buffer.freed? && buffer.resident?
      end

      # What is decided statically: the interface for saying, before running,
      # whether it fits and how much moves per run (#17's value 5).
      # bytes_per_run is **an upper-side estimate**; offload_stats[:bytes_in] is
      # the measurement. A sequential scan lands exactly on floor, but a tight
      # budget keeps the eviction victim from rotating fully and exceeds floor
      # (4 weights in 2 slots is an example).
      Report = Struct.new(:vram, :reserved, :budget, :weights, :paged, :resident_always,
                          :resident_always_bytes, :bytes_per_run, :floor, :steps,
                          :prefetch_headroom, keyword_init: true) do
        # How many prefetches can overlap is **headroom / one weight's size**.
        # "Greater than 0" is not enough — Jetson's tight budget had exactly one
        # weight of headroom, but a single transfer in flight gave no bandwidth
        # (0.66 GB/s), over 3x below the half budget's two (2.16 GB/s). 4b's gain
        # needs **at least two weights** of headroom.
        def prefetch_slots(weight_bytes)
          return 0 unless weight_bytes.to_i.positive?

          prefetch_headroom.to_i / weight_bytes
        end
        # The lower bound on bytes moved per run. A run touches every weight, so
        # at most budget carries over; the difference moves under any policy.
        def optimal? = bytes_per_run <= floor + 1

        def to_s
          pinned = resident_always.positive? ? format(", %d pinned (%s)", resident_always, Residency.human(resident_always_bytes)) : ""
          format("vram %s, reserved %s, weights %s of %s, paged %d of %d%s, bytes/run <= %s (floor %s), prefetch headroom %s",
                 Residency.human(vram), Residency.human(reserved), Residency.human(budget),
                 Residency.human(weights), paged, paged + resident_always, pinned,
                 Residency.human(bytes_per_run), Residency.human(floor),
                 Residency.human(prefetch_headroom.to_i))
        end
      end

      def self.human(bytes)
        units = %w[B KB MB GB TB]
        value = bytes.to_f
        unit = 0
        while value >= 1024.0 && unit < units.length - 1
          value /= 1024.0
          unit += 1
        end
        format("%.2f %s", value, units[unit])
      end

      def self.bytes_of(dtype, length) = length * Buffer::DTYPES.fetch(dtype)[:size]

      # The VRAM prefetches may hold: **the budget minus the largest single
      # step's need**, which directly decides how many can overlap.
      #
      # The subtraction is for correctness: a prefetching buffer cannot be
      # evicted (it is not even resident yet), so prefetching too much would
      # leave a cold fetch that must succeed unable to make room. compile only
      # guarantees that every single step's need fits in budget, so one step's
      # worth is always kept free.
      #
      # As a side effect it **also caps performance**: Jetson's tight budget
      # (2 slots, one step needing 1) leaves 1 slot of headroom, so one transfer
      # in flight = no concurrency, and 4b's bandwidth does not show (0.66 GB/s
      # measured vs 2.16 GB/s at the half budget). A budget buys **the ability
      # to overlap**, not only bytes.
      def self.prefetch_ceiling(needs, budget)
        budget - needs.each_value.filter_map { |list| list&.sum(&:bytes) }.max.to_i
      end

      # How many transfers prefetch keeps in flight. On Jetson
      # (docs/graph-offload.md, "Corrected by measurement") depth=2 gave
      # 1.85 GB/s and depth=4 2.24 GB/s — about 40% more, but most of it already
      # at 2, so 2 is the default as the balance against extra resident VRAM.
      PREFETCH_DEPTH = 2

      # How many steps ahead to look for the next demand. The scan only runs
      # until it finds an unfilled demand and returns at once on a miss, so a
      # wide window is cheap — up to one full cycle.
      LOOKAHEAD_HOPS = 64

      attr_reader :entries, :report

      # entries: the pageable Entry list
      # needs:   execution position -> Entries that must be resident there
      # steps:   executed nodes per run (next_use's wraparound width)
      # budget:  bytes available for paging
      def initialize(entries:, needs:, steps:, budget:, report:, spill_dir:, backing: :file)
        @entries = entries
        @by_id = entries.to_h { |entry| [entry.node_id, entry] }
        @needs = needs
        # Only positions with demand. Prefetch just walks this forward, which
        # misses less than stepping over positions without demand one by one.
        # needs is a Hash with a default (Plan#residency_needs), so build from
        # keys to avoid [] inserting empty arrays.
        @demand_positions = needs.keys.select { |i| !needs[i].to_a.empty? }.sort
        @steps = steps
        @budget = budget
        # The ceiling prefetches may occupy (derivation in Residency.prefetch_ceiling).
        @prefetch_ceiling = Residency.prefetch_ceiling(needs, budget)
        @report = report
        @spill_dir = spill_dir
        @backing = backing
        @spilled_paths = []
        # node_id -> Offload::Transfer: prefetches issued but not yet resident.
        # Their VRAM is allocated at issue time (see buffer.rb's restore_async!),
        # so they must count toward occupancy.
        @pending = {}
        # The plan does not synchronize at its end, so the queue is assumed dirty on entry.
        @queue_dirty = true
        reset_stats!
      end

      def reset_stats!
        @fetches = 0
        @evictions = 0
        @bytes_in = 0
        @spills = 0
        @spill_bytes = 0
        @waited = 0.0
        @added_fences = 0
        @free_fences = 0
        @peak = 0
      end

      def stats
        { fetches: @fetches, bytes_in: @bytes_in, waited_seconds: @waited.round(6),
          evictions: @evictions, spills: @spills, spill_bytes: @spill_bytes,
          added_fences: @added_fences, free_fences: @free_fences, peak_bytes: @peak,
          prefetched: @pending.length }
      end

      # Called before running the node at execution position index: loads the
      # weights it needs, making room with Belady when short.
      #
      # 4a fetched synchronously here (as did the early docs/graph-offload.md).
      # On Jetson the same transfers measured 0.56 GB/s one at a time
      # synchronously vs 1.85–2.24 GB/s asynchronously at depth 2–4, a 3–4x gap
      # (Offload.pipeline prefetched asynchronously while 4a called `restore!`
      # one by one). So "fetch synchronously here if it has not arrived" is
      # demoted to **the fallback on a miss**, and the main path harvests
      # transfers `prefetch!` set up asynchronously beforehand (#17 Phase 4b).
      def ensure!(index)
        needed = @needs[index] || []
        cold = []
        needed.each do |entry|
          next if entry.resident?

          if (transfer = @pending.delete(entry.node_id))
            wait_transfer(entry, transfer)
          else
            cold << entry
          end
        end
        # Besides loading what is missing, shrink when over budget (a plan
        # binding resident Buffers starts over budget).
        #
        # make_room! is called even when cold is empty: with want=0 it still
        # shrinks if occupancy exceeds budget. Calling it only when cold is
        # non-empty would never detect being over budget whenever the demand
        # happens to be satisfied already.
        make_room!(cold.sum(&:bytes), keep: needed, from: index)
        cold.each { |entry| fetch_sync(entry) }
        prefetch!(index, needed) if Graph.offload_prefetch && Offload.async?
        # Staying within budget can only be shown by measurement, so record the
        # moment right after everything needed is loaded (= peak const
        # occupancy). VRAM held by prefetches is charged here too, hence
        # occupied_bytes.
        @peak = [@peak, occupied_bytes].max
        nil
      end

      # Sets up near-future demand asynchronously before it is due. **Issues
      # only, never waits** — waiting happens in ensure! (wait_transfer) when
      # the index that needs the weight arrives. Several in flight overlap, so
      # the depth effect measured on Jetson carries over.
      #
      # Failure never breaks the run: if no room can be made it quietly gives
      # up and that weight is fetched synchronously at its demand point as
      # before (the cold fallback always works, so with zero prefetches the
      # result matches 4a).
      def prefetch!(index, needed)
        return if @pending.length >= PREFETCH_DEPTH
        return if @prefetch_ceiling <= 0 # the largest single step's need uses the whole budget

        candidates = upcoming_targets(index, needed)
        candidates.each do |entry, target_index, keep|
          break if @pending.length >= PREFETCH_DEPTH

          next if entry.resident? || @pending.key?(entry.node_id)
          next unless pending_bytes + entry.bytes <= @prefetch_ceiling

          begin
            make_room!(entry.bytes, keep: keep, from: target_index) unless occupied_bytes + entry.bytes <= @budget
          rescue Error
            next # no room; this weight waits for its demand point
          end
          fetch_async(entry)
        end
        nil
      end

      # Searches from after index up to LOOKAHEAD_HOPS ahead for weights not
      # yet resident and not already prefetching. Returns each as (Entry,
      # demand position, set of Entries to protect until then).
      #
      # **keep starts with the current step's (index) demand.** ensure! has
      # made all of index's demand resident before getting here, but the node
      # has not run yet — evicting one in this call would make the very next
      # node read freed VRAM. Belady could misjudge it as distant (it looks
      # unused until the next cycle), so it must be protected explicitly rather
      # than by proximity.
      def upcoming_targets(index, needed)
        return [] if @demand_positions.empty?

        start = @demand_positions.bsearch_index { |pos| pos > index } || 0
        seen = {}
        keep = needed.dup
        found = []
        LOOKAHEAD_HOPS.times do |hop|
          pos = @demand_positions[(start + hop) % @demand_positions.length]
          break if hop.positive? && pos == @demand_positions[start] # went all the way around

          (@needs[pos] || []).each do |entry|
            keep << entry
            next if seen[entry.node_id]
            next if entry.resident? || @pending.key?(entry.node_id)

            seen[entry.node_id] = true
            found << [entry, pos, keep.dup]
          end
          break if found.length >= PREFETCH_DEPTH
        end
        found
      end

      # The Plan reports the queue state. **Eviction needs a fence**, but if the
      # previous node ran synchronously (:nn / a reduction / a delegated GEMM)
      # its wait already left the queue empty — riding on it adds no
      # Tilework.synchronize at all (#17 Phase 4c).
      #
      # Riding along **only removes waits**; it changes neither the victim nor
      # the timing. Dropping early (freeing future room at the fence) was
      # measured and rejected: dropping early has fewer candidates (weights not
      # yet done are ineligible), so it picks a worse victim than waiting would.
      # Measured, waits went 10 -> 4 but transfers grew 2304 -> 2816 bytes; on
      # a disk-bound workload trading bytes for waits does not pay.
      def note_queue(dirty:)
        @queue_dirty = dirty
        nil
      end

      # Deletes only the offload files the plan wrote. **Files of weights the
      # user offloaded themselves are left alone** — they are not ours.
      #
      # Before deleting, **return what was borrowed**: if a weight we wrote out
      # is still offloaded, bring it back to VRAM first. Deleting without
      # returning would leave the user's Buffer with neither file nor VRAM,
      # unusable forever.
      #
      # Transfers set up by prefetch that nobody waited on are collected here
      # too; otherwise they keep holding VRAM and the bridge's worker lingers
      # (the same reason as Offload.pipeline's ensure block — see offload.rb).
      def free_spills!
        require "fileutils"
        @pending.each_value(&:wait)
        @pending.clear
        @entries.each do |entry|
          next unless entry.spilled
          next if entry.buffer.freed?

          entry.buffer.restore! unless entry.resident?
          entry.buffer.discard_host_backing! if @backing == :host
          entry.has_file = false
        end
        @spilled_paths.each { |path| FileUtils.rm_f(path) }
        @spilled_paths.clear
        @entries.each { |entry| entry.spilled = false }
        nil
      end

      # What is known without running.
      def needs_at(index) = @needs[index] || []

      private

      # Evictions are batched under one fence. **evict! is buffer_free**, and a
      # kernel not yet executed may still read that VRAM (the plan does not
      # synchronize between nodes) — the same hazard Pool avoids by returning
      # rather than freeing — so always wait before dropping. Batching means a
      # single wait instead of one per eviction.
      def make_room!(want, keep:, from:)
        free = @budget - occupied_bytes
        return if free >= want

        victims = choose_victims(want - free, keep: keep, from: from)
        return if victims.empty?

        # Wait here only if the previous node did not. Rides are counted in
        # free_fences so they can be told apart from added waits.
        if @queue_dirty
          Tilework.synchronize
          @added_fences += 1
          @queue_dirty = false
        else
          @free_fences += 1
        end
        victims.each { |entry| evict(entry) }
      end

      # Drop whatever is used farthest in the future (Belady). Being cyclic,
      # something unused for the rest of this run is judged by its position in
      # the next run.
      def choose_victims(want, keep:, from:)
        keep_ids = keep.to_h { |entry| [entry.node_id, true] }
        candidates = @entries.select { |entry| entry.resident? && !keep_ids[entry.node_id] }
                             .sort_by { |entry| -next_use(entry, from) }
        victims = []
        freed = 0
        candidates.each do |entry|
          break if freed >= want

          victims << entry
          freed += entry.bytes
        end
        return victims if freed >= want

        # Unreachable if compile-time checks passed (they verify that the
        # weights one node needs at once fit in the budget).
        raise Error, "cannot free #{Residency.human(want)} of VRAM at step #{from}: " \
                     "#{Residency.human(freed)} is all the paged weights hold. " \
                     "Raise the vram: budget passed to compile."
      end

      def next_use(entry, from)
        found = entry.uses.bsearch { |use| use >= from }
        found || (entry.uses.first + @steps)
      end

      def resident_bytes = @entries.sum { |entry| entry.resident? ? entry.bytes : 0 }

      # Prefetching weights occupy VRAM too — restore_async! has called
      # buffer_alloc at issue time and resident? stays false until the transfer
      # ends (see buffer.rb's __finish_restore__). Missing them here would let
      # actual occupancy exceed budget without make_room! noticing.
      def occupied_bytes = resident_bytes + pending_bytes

      def pending_bytes = @pending.each_key.sum { |id| @by_id.fetch(id).bytes }

      # The fallback when something has not arrived at its demand point. One
      # synchronous transfer at a time is slow (0.56 GB/s measured on Jetson),
      # but it alone guarantees correctness — with zero effective prefetches the
      # run gives the same result as 4a.
      def fetch_sync(entry)
        started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        entry.buffer.restore!
        @waited += Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
        entry.has_file = true # restore! succeeding means the file exists
        @fetches += 1
        @bytes_in += entry.bytes
        entry
      end

      # Issue only. Stats (fetches / bytes_in / waited) are counted in
      # wait_transfer — nothing has finished at issue time, so counting here
      # would break bytes_in's meaning of "bytes transferred".
      def fetch_async(entry)
        @pending[entry.node_id] = entry.buffer.restore_async!
        entry.has_file = true
        entry
      end

      def wait_transfer(entry, transfer)
        started = Process.clock_gettime(Process::CLOCK_MONOTONIC)
        transfer.wait
        @waited += Process.clock_gettime(Process::CLOCK_MONOTONIC) - started
        @fetches += 1
        @bytes_in += entry.bytes
        entry
      end

      # If an offload file already exists, just drop the VRAM. Weights are
      # read-only, so writing back would rewrite the same contents with
      # synchronous I/O (the same reason Offload.pipeline's write_back: defaults
      # to false). A missing file means the user bound a resident Buffer; only
      # then is it written out, once.
      def evict(entry)
        if entry.has_file
          entry.buffer.evict!
        elsif @backing == :host
          entry.buffer.offload_to_host!
          entry.has_file = true
          entry.spilled = true
          @spills += 1
          @spill_bytes += entry.bytes
        else
          path = spill_path(entry)
          entry.buffer.offload!(path)
          entry.has_file = true
          entry.spilled = true
          @spilled_paths << path
          @spills += 1
          @spill_bytes += entry.bytes
        end
        @evictions += 1
        entry
      end

      def spill_path(entry)
        require "fileutils"
        FileUtils.mkdir_p(@spill_dir)
        File.join(@spill_dir, format("const-%08d.rmgpu", entry.node_id))
      end
    end
  end
end
