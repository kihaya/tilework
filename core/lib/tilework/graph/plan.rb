# frozen_string_literal: true

module Tilework
  module Graph
    # An executable graph with a fixed operation order and buffer lifetime plan.
    #
    # Nodes run through the existing eager layer (Ops), fused where possible.
    # What compile fixes is **order** and **lifetime**:
    #
    #   1. topologically sort after adding mutable state's WAR edges to the value (RAW) edges
    #   2. release intermediates at their last read (state buffers excluded)
    class Plan
      # Owning memory, running synchronously, leaving our execution path and
      # accepting a placed output are all op properties, held by the Graph::Op
      # table. This class consults it and concentrates on order and lifetime.
      attr_reader :order, :nodes, :strict_state, :async

      def initialize(builder, vram: nil, spill_dir: nil, backing: :file, host_budget: nil)
        @builder = builder
        # The mode is fixed at compile time (so its meaning cannot change between runs).
        @strict_state = Graph.strict_state
        @async = Graph.async
        @pool = Graph.buffer_pool ? Pool.new : nil
        @context = Context.new(pool: @pool, async: @async)
        check_single_leaf!
        @roots = builder.outputs + builder.states.map { |st| st.leaves.first.node }
        @nodes = reachable(@roots)
        @nodes_by_id = @nodes.to_h { |node| [node.id, node] }
        check_dtypes_supported!
        check_requirements!
        @order_deps = order_dependencies
        @order = toposort
        @clusters, @absorbed = plan_fusion
        @placements = plan_placement
        @release_at = plan_releases
        @residency = plan_residency(vram, spill_dir, backing, host_budget)
      end

      def input_names = @builder.input_nodes.keys
      def states = @builder.states
      def outputs = @builder.outputs

      # How many random draws this plan makes; 0 means no seed is needed.
      def random_draws = @builder.random_draws

      # The seed to pass to the next step. RNG inside a plan is a pure function
      # of keys derived from the seed, so **it only needs advancing between
      # steps** (#19 §6). run does not return the seed, which would make the
      # output layout variable; advancing is offered here as a pure function.
      def next_seed(seed) = seed + 1

      # Per-node key. Mixing in the index scaled by the golden ratio keeps the
      # sequence apart from both neighboring seeds and neighboring nodes (the
      # same trick as Random.next_seed).
      def self.derive_key(seed, index) = (seed ^ ((index + 1) * 0x9E37_79B9_7F4A_7C15)) & 0xFFFF_FFFF_FFFF_FFFF

      # Runs the plan with named input and index bindings.
      #
      # @param bindings [Hash{Symbol => Buffer, Integer}] values declared with {Builder#input} or {Builder#index}
      # @return [Buffer, Array<Buffer>] graph outputs in {Builder#output} order
      # @note The method does not synchronize; reading an output waits for queued work.
      #
      #   plan.run(x: buf, pos: 7)
      #
      # Outputs come in g.output order; a single one is returned as a bare
      # Buffer. State buffers are updated in place in the bound Buffers (not
      # returned).
      #
      # **The plan does not synchronize at its end.** It returns with launches
      # still queued, so rerunning the same plan, as a decode loop does, adds no
      # per-step wait. Reading a result (`#to_a`) waits for the host transfer,
      # so there is one synchronization when the value is actually wanted.
      def run(**bindings)
        values = {}
        output_ids = outputs.map(&:id)
        # Nodes whose placement took effect; for them the write needs no copy.
        @placed = {}
        @sync_stats = { nodes: 0, synchronous: 0, fences: 0 }
        # If there is an enclosing submit boundary, hand ownership to it (#86
        # stage 2). The plan installs its own context, so without this, deferred
        # frees of buffers made in the submit block would land outside the boundary.
        @context.outer = Submission.context
        Ops.with_context(@context) do
          index = 0
          @order.each do |node|
            next if @absorbed[node.id] # computed inside its cluster, not run alone

            @residency&.ensure!(index)
            values[node.id] = execute_node(node, values, bindings)
            release(index, values, output_ids)
            note_queue_state(node)
            index += 1
          end
        end
        @context.drain_deferred!
        results = outputs.map { |node| values[node.id] }
        results.length == 1 ? results.first : results
      end

      # The capabilities this plan requires: the interface for saying "this
      # graph does not run on :metal" **before running** (#17's value 5).
      #
      #   plan.requires
      #   # => [#<Requirement capability=:conv ops=[:conv2d] satisfied=true>]
      #
      # compile fails on anything unsatisfiable, so satisfied=false appears
      # here only if capabilities changed afterwards. Ops with a decomposition
      # are expanded into primitives at record time and never appear here.
      Requirement = Struct.new(:capability, :ops, :satisfied) do
        def satisfied? = satisfied
        def to_s = "#{capability} (#{ops.join(', ')})#{satisfied ? '' : ' — missing'}"
      end

      def requires
        @nodes.filter_map { |node| [node.spec.requires, node.op] if node.spec.requires }
              .group_by(&:first)
              .map { |capability, pairs| Requirement.new(capability, pairs.map(&:last).uniq, Op.satisfied?(capability)) }
      end

      # How weights were assigned against the VRAM budget (only when
      # compile(vram:) was given). Tells before running whether it fits and
      # how many bytes move per run — see docs/graph-offload.md.
      #
      #   plan.residency
      #   # => vram 8.00 GB, reserved 1.07 GB, weights 6.93 GB of 13.47 GB,
      #   #    paged 18 of 34, bytes/run 7.00 GB (floor 6.55 GB)
      def residency = @residency_report

      # Bytes actually moved and time waited. Compare bytes_in with
      # residency.floor first.
      def offload_stats = @residency&.stats

      # Deletes the offload files the plan itself wrote (the companion of
      # free_pool!). Files of weights the user offloaded are left alone.
      def free_offload! = @residency&.free_spills!

      # How many buffers the pool holds and how many allocations / reuses
      # happened — for measuring whether reuse works (allocated stops growing
      # from the second run on).
      def pool_stats = @pool&.stats

      # Releases the pool's device memory.
      def free_pool! = @pool&.free_all!

      # In the latest run, how many nodes ran and how many made the host wait
      # (#86's acceptance criterion: observing fewer host synchronizations per
      # dispatch). `synchronous` counts nodes launched with wait: true, `fences`
      # the `Tilework.synchronize` calls added around them.
      #
      #   plan.run(x: buf); plan.sync_stats
      #   # => {nodes: 42, synchronous: 0, fences: 0}
      def sync_stats = @sync_stats&.dup

      # A readable listing of the execution order (for debugging). Nodes fused
      # away are never launched, so they are marked as such.
      def explain
        index = -1
        @order.map do |node|
          next "  (fused) #{node}" if @absorbed[node.id]

          index += 1
          suffix = @clusters[node.id] ? " <- fused #{@clusters[node.id].members.length + 1} nodes" : ""
          "#{index}: #{node}#{suffix}"
        end.join("\n")
      end

      # Fusion results. nodes counts nodes actually executed (including ones
      # like inputs and constants that launch no kernel), so it is not the
      # launch count itself.
      def fusion_stats
        { clusters: @clusters.length, absorbed: @absorbed.length,
          nodes: @order.count { |node| !@absorbed[node.id] } }
      end

      def inspect
        "#<Tilework::Graph::Plan #{@order.length} nodes, #{states.length} states, " \
          "#{outputs.length} outputs, strict_state: #{@strict_state.inspect}>"
      end

      private

      # --- Compile ---

      # Each state must end in a single final version; forgetting to rejoin
      # forked writes to disjoint regions with g.merge fails here.
      def check_single_leaf!
        @builder.states.each do |st|
          next if st.leaves.length == 1

          versions = st.leaves.map { |leaf| "v#{leaf.version}" }.join(", ")
          raise Error, "#{st.name} has #{st.leaves.length} unmerged versions (#{versions}); " \
                       "disjoint writes must be joined with g.merge before the plan ends"
        end
      end

      # Type checking (Builder) and whether those types can be allocated on this
      # backend (here) are separate questions. A float16 graph built on :cpu is
      # reported **at compile time**, all at once, instead of failing at the
      # first Buffer.new mid-run — the smallest form of #17's "know portability
      # before running".
      def check_dtypes_supported!
        supported = Tilework.capabilities.dtypes
        missing = @nodes.map(&:dtype).uniq.reject { |dtype| supported.include?(dtype) }
        return if missing.empty?

        raise Error, "the graph uses #{missing.map(&:inspect).join(', ')} but the " \
                     "#{Tilework.backend.inspect} backend can only allocate " \
                     "#{supported.map(&:inspect).join(', ')}"
      end

      # As with types, whatever can be said before running is said before
      # running. Only "the op is missing and has no decomposition" gets here —
      # ops with a decomposition were expanded into primitives at record time.
      def check_requirements!
        missing = requires.reject(&:satisfied?)
        return if missing.empty?

        raise Error, "the graph needs #{missing.map(&:to_s).join(', ')} but the " \
                     "#{Tilework.backend.inspect} backend does not provide it"
      end

      def reachable(roots)
        seen = {}
        stack = roots.dup
        while (node = stack.pop)
          next if seen.key?(node.id)

          seen[node.id] = node
          node.inputs.each { |input| stack.push(input) }
        end
        seen.values.sort_by(&:id)
      end

      # Version edges give only RAW (write, then read), not WAR — a version is
      # **the same memory**, not a new value, so version N+1's write running
      # while version N's readers are still reading breaks things. There is no
      # value edge for that, so it is added here as an ordering edge:
      #
      #   a node overwriting version N runs after the readers of version N and their direct consumers.
      #
      # Direct consumers suffice because a read returns a view (zero copy) and
      # each consumer owns its output buffer; once it has run, nothing aliases
      # the state any more.
      def order_dependencies
        deps = Hash.new { |h, k| h[k] = [] }
        consumers = consumer_index
        @nodes.each do |writer|
          next unless writer.op == :state_write

          version_node = writer.inputs.first
          consumers[version_node.id].each do |reader|
            next unless reader.op == :state_read
            next if reader.equal?(writer)

            deps[writer.id] << reader.id
            consumers[reader.id].each do |consumer|
              deps[writer.id] << consumer.id unless consumer.equal?(writer)
            end
          end
        end
        deps.each_value(&:uniq!)
        deps
      end

      def consumer_index
        index = Hash.new { |h, k| h[k] = [] }
        in_plan = @nodes.each_with_object({}) { |n, h| h[n.id] = true }
        @nodes.each do |node|
          node.inputs.each { |input| index[input.id] << node if in_plan[input.id] }
        end
        index
      end

      # Topological sort over value edges + ordering edges. Ready nodes come out
      # in id order, so the same graph always yields the same order.
      def toposort
        adjacency = Hash.new { |h, k| h[k] = [] }
        indegree = @nodes.to_h { |node| [node.id, 0] }
        by_id = @nodes.to_h { |node| [node.id, node] }

        @nodes.each do |node|
          (node.inputs.map(&:id) + @order_deps[node.id]).uniq.each do |from|
            adjacency[from] << node.id
            indegree[node.id] += 1
          end
        end

        ready = indegree.select { |_, d| d.zero? }.keys.sort
        sorted = []
        until ready.empty?
          id = ready.shift
          sorted << by_id[id]
          adjacency[id].each do |to|
            indegree[to] -= 1
            ready = ready.push(to).sort if indegree[to].zero?
          end
        end
        return sorted if sorted.length == @nodes.length

        raise Error, cycle_message(@nodes.reject { |n| sorted.include?(n) })
      end

      # In practice a cycle has only one shape: a read of version N used by a
      # node **after** the one overwriting version N. The read is a view, so by
      # then its value has been overwritten; no order resolves it, so fail.
      def cycle_message(stuck)
        states = stuck.select(&:writes_state?).map { |n| n.state.name }.uniq
        detail = states.empty? ? "" : " (#{states.join(', ')})"
        "cannot order the graph#{detail}: a value read from a state version is used after that " \
          "version was overwritten. A read is a view of the same memory, so the value would be " \
          "clobbered — read the newer version instead, or set " \
          "Tilework::Graph.strict_state = :copy to materialize reads.\n" \
          "  stuck: #{stuck.map(&:to_s).join("\n         ")}"
      end

      # The fusion plan. Nodes absorbed into a cluster are no longer
      # materialized, so liveness and execution from here on work in terms of
      # **effective_inputs** (inputs coming from outside the cluster).
      def plan_fusion
        return [{}, {}] unless Graph.fusion

        # Nodes named by ordering edges are not absorbed: that would move "the
        # moment it finished" to the cluster root and reorder it against state
        # writes. @order_deps is a Hash with a default, and toposort leaves empty
        # arrays behind for every node it looked up — consider **only non-empty
        # edges**.
        pinned = @order_deps.flat_map { |writer, readers| readers.empty? ? [] : [writer, *readers] }
                            .to_h { |id| [id, true] }
        Fusion.plan(order: @order, output_ids: outputs.map(&:id).to_h { |id| [id, true] },
                    consumers: consumer_index, pinned: pinned)
      end

      # Output placement (#19 §3). `write_into` needs no copy in principle —
      # making the write target the producer's **output** costs nothing. This
      # picks only the nodes where that is safe.
      #
      # Four conditions, each of which breaks things if dropped:
      #
      #   1. the op allocates its output once and fills it. NN, opaque and
      #      reductions allocate several inside, so which one is the result is
      #      unknowable
      #   2. the write is the only consumer. Placing a value read elsewhere on
      #      the state lets a later write corrupt that reader
      #   3. the write carries no ordering (WAR) edge. **The actual memory
      #      mutation moves to the value's node**, so the edge would have to move
      #      with it, and if it points at that node's descendants it becomes a
      #      cycle; only cases needing no move pass
      #   4. the value's node does not read the same state (it would read and
      #      write one region at once)
      #
      # 1 is an op property held by the table (Graph::Op); the other three are
      # about the graph's shape, so they are checked here.
      def plan_placement
        return {} unless Graph.placement

        consumers = consumer_index
        output_ids = outputs.map(&:id).to_h { |id| [id, true] }
        @nodes.select { |node| node.op == :state_write }.filter_map do |writer|
          value = writer.inputs[1]
          next unless placeable?(value, writer, consumers, output_ids)

          [value.id, writer]
        end.to_h
      end

      def placeable?(value, writer, consumers, output_ids)
        return false unless value.spec.placeable?(value)
        return false if output_ids.key?(value.id)
        return false unless consumers[value.id].length == 1
        return false unless @order_deps[writer.id].empty?

        !reads_state?(value, writer.state)
      end

      # Whether any node touching the same state precedes producing value.
      def reads_state?(value, state)
        seen = {}
        stack = [value]
        while (node = stack.pop)
          next if seen[node.id]

          seen[node.id] = true
          return true if node.attrs[:state].equal?(state)

          node.inputs.each { |input| stack.push(input) }
        end
        false
      end

      # What is known only at run time (dynamic positions, contiguity) is checked here.
      def placement_for(node, bindings)
        writer = @placements[node.id]
        return nil unless writer

        region = writer.region.resolve(bindings)
        return nil unless region.contiguous?

        writer.state.buffer.view(region.offset, region.length, shape: node.shape)
      end

      # reshape and slice own no memory and just return a view of their input.
      # Reading a view reads the original buffer, so lifetime must be counted
      # on **the owning side**, not the view — otherwise memory still being read
      # goes back to the pool and to the next writer.
      ALIAS_OPS = %i[reshape slice].freeze

      def alias_root(node)
        node = node.inputs.first while ALIAS_OPS.include?(node.op)
        node
      end

      # Inputs closed inside a cluster disappear, so liveness uses inputs from outside it.
      def effective_inputs(node)
        cluster = @clusters[node.id]
        cluster ? cluster.inputs : node.inputs
      end

      # Liveness: release intermediates where they are last read. State buffers
      # (and inputs and constants) are not owned, so they are left alone.
      def plan_releases
        releases = Hash.new { |h, k| h[k] = [] }
        executed = @order.reject { |node| @absorbed[node.id] }
        position = executed.each_with_index.to_h { |node, i| [node.id, i] }
        # reshape is a view, so the output is held by the owning side.
        output_ids = outputs.map { |node| alias_root(node).id }
        last = {}
        executed.each do |node|
          effective_inputs(node).each { |input| last[alias_root(input).id] = position[node.id] }
        end
        @nodes.each do |node|
          next unless node.spec.owned?
          next if output_ids.include?(node.id)
          next if @absorbed[node.id] # never materialized, so nothing to release
          next unless last.key?(node.id)

          releases[last[node.id]] << node.id
        end
        releases
      end

      # Assigning weights against the budget (#17 Phase 4 / docs/graph-offload.md).
      # Nothing happens without vram: — paging on its own without being given a
      # budget would hide the performance cliff (the same judgment as #17's
      # "no implicit laziness").
      def plan_residency(vram, spill_dir, backing, host_budget)
        return nil if vram.nil? || !Graph.offload

        unless %i[file host].include?(backing)
          raise ArgumentError, "backing: must be :file or :host"
        end
        if backing == :host
          raise Error, "pageable host backing is not available on the active backend" unless Tilework.capabilities[:host_backing]
          raise ArgumentError, "host_budget: must be a positive Integer of bytes" unless host_budget.is_a?(Integer) && host_budget.positive?
        end

        raise ArgumentError, "vram: must be a positive Integer of bytes" unless vram.is_a?(Integer) && vram.positive?

        executed = @order.reject { |node| @absorbed[node.id] }
        positions = executed.each_with_index.to_h { |node, i| [node.id, i] }
        candidates, pinned = split_consts
        reserved = reserved_bytes(executed) + pinned.sum { |node| bytes_of(node) }
        budget = vram - reserved
        weights = candidates.sum { |node| bytes_of(node) }
        if backing == :host && weights > host_budget
          raise Error, "host backing budget #{host_budget} bytes cannot hold pageable candidates totaling #{weights} bytes"
        end

        entries = residency_entries(candidates, executed, positions)
        needs = residency_needs(entries, executed, positions)
        check_budget!(entries, needs, budget, vram, reserved, weights)
        # Report even when it fits (being able to say "it fits" is worth something).
        @residency_report = residency_report(vram, reserved, budget, weights, entries, pinned, executed, needs)
        # A loose budget is free — if it fits, the machinery does not exist.
        return nil if budget >= weights

        Residency.new(entries: entries, needs: needs, steps: executed.length, budget: budget,
                      spill_dir: spill_dir || default_spill_dir, backing: backing, report: @residency_report)
      end

      # Splits consts into pageable and pinned. Two kinds cannot move: views
      # (restore! allocates a new handle, leaving views of the old one dangling;
      # buffer.rb's reject_view! refuses them too) and plan outputs (they must
      # stay resident after run).
      def split_consts
        output_roots = outputs.map { |node| alias_root(node).id }.to_h { |id| [id, true] }
        consts = @nodes.select { |node| node.op == :const && !@absorbed[node.id] }
                       .reject { |node| node.attrs[:buffer].freed? }
        consts.partition do |node|
          !node.attrs[:buffer].view? && !output_roots[node.id]
        end
      end

      def residency_entries(candidates, executed, positions)
        uses = Hash.new { |h, k| h[k] = [] }
        executed.each do |node|
          const_inputs(node).each { |const| uses[const.id] << positions[node.id] }
        end
        candidates.filter_map do |node|
          used = uses[node.id].uniq.sort
          next if used.empty?

          buffer = node.attrs[:buffer]
          Residency::Entry.new(node.id, buffer, bytes_of(node), used, !buffer.resident?, false)
        end
      end

      def residency_needs(entries, executed, positions)
        by_node = entries.to_h { |entry| [entry.node_id, entry] }
        needs = Hash.new { |h, k| h[k] = [] }
        executed.each do |node|
          const_inputs(node).each do |const|
            entry = by_node[const.id]
            needs[positions[node.id]] << entry if entry
          end
        end
        needs.each_value(&:uniq!)
        needs
      end

      # Consts that must be resident when this node runs. Inputs fused away have
      # become the cluster's inputs, so effective_inputs is consulted. reshape /
      # slice create views (Buffer#view requires residency), so they appear here
      # as direct inputs and count as a requirement at that position too.
      def const_inputs(node)
        effective_inputs(node).flat_map { |input| [input, alias_root(input)] }
                              .select { |input| input.op == :const }.uniq
      end

      # The part of the budget weights cannot use. Inputs/outputs and state
      # cannot move, and the pool keeps intermediates, so those count at peak.
      def reserved_bytes(executed)
        io = @builder.input_nodes.each_value.sum { |node| bytes_of(node) } +
             outputs.sum { |node| bytes_of(alias_root(node)) }
        state = states.sum { |st| Residency.bytes_of(st.dtype, st.buffer.length) }
        io + state + peak_intermediate_bytes(executed)
      end

      # Peak intermediates. The pool keeps slots across runs, so this is
      # steady-state occupancy, not the first run's. Liveness (@release_at)
      # already knows the release points, so this walks the same table in bytes.
      def peak_intermediate_bytes(executed)
        output_ids = outputs.map { |node| alias_root(node).id }.to_h { |id| [id, true] }
        live = 0
        peak = 0
        executed.each_with_index do |node, index|
          live += bytes_of(node) if node.spec.owned? && !output_ids[node.id]
          peak = live if live > peak
          @release_at[index].each do |id|
            released = @nodes.find { |n| n.id == id }
            live -= bytes_of(released) if released && !output_ids[id]
          end
        end
        peak
      end

      # Fail before running whatever can be (like dtype / requires). Two
      # checks: no room at all, or one node's simultaneous weights do not fit.
      # The latter is "enough overall, but this one line does not pass", so the
      # node is named.
      def check_budget!(entries, needs, budget, vram, reserved, weights)
        largest = entries.map(&:bytes).max || 0
        if budget < largest
          raise Error, "the graph reserves #{Residency.human(reserved)} for inputs, state and " \
                       "intermediates out of a #{Residency.human(vram)} budget, leaving " \
                       "#{Residency.human(budget)} for #{Residency.human(weights)} of weights; " \
                       "the largest single weight is #{Residency.human(largest)}"
        end

        needs.each do |index, required|
          want = required.sum(&:bytes)
          next if want <= budget

          node = @order.reject { |n| @absorbed[n.id] }[index]
          raise Error, "step #{index} (#{node}) needs #{Residency.human(want)} of weights at once " \
                       "but only #{Residency.human(budget)} is left after reserving " \
                       "#{Residency.human(reserved)} of a #{Residency.human(vram)} budget"
        end
      end

      def residency_report(vram, reserved, budget, weights, entries, pinned, executed, needs)
        # Bytes moved per run. At most budget carries over, so the difference
        # moves under any policy (= floor). Cyclic Belady lands within one slot
        # of it.
        floor = [weights - budget, 0].max
        # In practice one slot more moves — a sequential scan needs one slot for
        # the eviction victim to rotate through. offload_stats[:bytes_in] is the
        # measurement.
        rotating = floor.positive? ? (entries.map(&:bytes).max || 0) : 0
        Residency::Report.new(
          vram: vram, reserved: reserved, budget: budget, weights: weights,
          paged: entries.length, resident_always: pinned.length,
          resident_always_bytes: pinned.sum { |node| bytes_of(node) },
          prefetch_headroom: Residency.prefetch_ceiling(needs, budget),
          bytes_per_run: floor + rotating, floor: floor, steps: executed.length,
        )
      end

      def bytes_of(node) = Residency.bytes_of(node.dtype, node.length)

      def default_spill_dir
        require "tmpdir"
        File.join(Dir.tmpdir, "tilework-offload-#{Process.pid}-#{object_id}")
      end

      # --- Execution ---

      # Returns last-read intermediates to the pool (frees them without one).
      # Views own nothing, so they never enter the pool.
      def release(index, values, output_ids)
        @release_at[index].each do |id|
          buffer = values[id]
          next unless buffer.is_a?(Buffer) && !output_ids.include?(id)

          values[id] = nil
          if @pool && !buffer.view?
            if @pool.owns?(buffer)
              @pool.release(buffer)
            elsif @nodes_by_id.fetch(id).op == :opaque
              # opaque may return a value the user made with Buffer.new. If
              # fenced, the node ran synchronously and it can be freed on the
              # spot; with fenced: false launches may still be running, so it
              # is adopted into the pool instead of freed.
              node = @nodes_by_id.fetch(id)
              if node.attrs[:fenced]
                buffer.free
              else
                @pool.release(@pool.adopt(buffer))
              end
            else
              # An ownership violation that should be unreachable. In case the op
              # was asynchronous, wait before freeing so the diagnostic exception
              # does not leave a new leak behind.
              Tilework.synchronize
              buffer.free
              raise Error, "graph node #{@nodes_by_id.fetch(id).op.inspect} returned an intermediate " \
                           "buffer that did not come from Ops.allocate_output"
            end
          else
            buffer.free
          end
        end
      end

      # After a synchronous node the queue is empty; tell residency, so the next
      # eviction can ride on that wait (#17 Phase 4c).
      def note_queue_state(node)
        return unless @residency
        # Nodes that launch nothing (:const / :input / views) leave the queue as
        # it was. Counting them as dirtying it would mean the :const right
        # before every weight-reading matmul prevents any ride-along.
        return unless node.spec.launches?

        @residency.note_queue(dirty: !Graph.offload_fence_reuse || !drains_queue?(node))
      end

      # Whether the queue is empty once this node finishes. Synchronous nodes
      # (:nn / whole-buffer reductions / delegated GEMM / opaque) launch with
      # wait: true, so earlier work on the same queue has completed too.
      # With asynchronous execution disabled, every node drains the queue.
      def drains_queue?(node) = !@async || node.spec.synchronous?(node)

      # Nodes that can stay asynchronous do; nodes that (may) leave our
      # execution path wait on both sides and run synchronously.
      def execute_node(node, values, bindings)
        target = placement_for(node, bindings)
        return dispatch_node(node, values, bindings) unless target

        result = @context.place(target) { dispatch_node(node, values, bindings) }
        @placed[node.id] = @context.placement_consumed?
        result
      end

      def dispatch_node(node, values, bindings)
        @sync_stats[:nodes] += 1
        return execute(node, values, bindings) unless synchronous?(node)

        @sync_stats[:synchronous] += 1
        foreign = foreign?(node)
        if foreign
          @sync_stats[:fences] += 2
          Tilework.synchronize
        end
        result = @context.synchronous { execute(node, values, bindings) }
        Tilework.synchronize if foreign
        result
      end

      # Whether to run synchronously or wait is an op property (including
      # batching and delegation). Making everything synchronous only when
      # @async is off is this method's job.
      def synchronous?(node) = @async && node.spec.synchronous?(node)

      def foreign?(node) = node.spec.foreign?(node)

      # The table holds each op's execution method. All take the same
      # (node, inputs, bindings), so no branching is needed here.
      def execute(node, values, bindings)
        cluster = @clusters[node.id]
        return run_fused(cluster, cluster.inputs.map { |input| values[input.id] }) if cluster

        inputs = node.inputs.map { |input| values[input.id] }
        send(node.spec.run, node, inputs, bindings)
      end

      def run_const(node, _inputs, _bindings) = node.attrs[:buffer]
      def run_state_init(node, _inputs, _bindings) = node.state.buffer
      def run_state_merge(node, _inputs, _bindings) = node.state.buffer

      def run_input(node, _inputs, bindings)
        name = node.attrs[:name]
        buffer = bindings.fetch(name) do
          raise Error, "plan.run is missing input #{name.inspect} (expects #{input_names.inspect})"
        end
        raise TypeError, "input #{name.inspect} must be a Tilework::Buffer, got #{buffer.class}" unless buffer.is_a?(Buffer)
        unless buffer.dtype == node.dtype && buffer.shape == node.shape
          raise Error, "input #{name.inspect} was declared as #{node.shape.inspect} #{node.dtype} " \
                       "but got #{buffer.shape.inspect} #{buffer.dtype}"
        end

        buffer
      end

      # A read is a view by default = no kernel. It is materialized when
      #   - it is a plan output (so later writes do not corrupt it)
      #   - strict_state == :copy (for comparison with the pure semantics)
      #   - the region is not contiguous (a view cannot express it)
      def run_state_read(node, _inputs, bindings)
        region = node.region.resolve(bindings)
        buffer = node.state.buffer
        materialize = @strict_state == :copy || node.attrs[:materialize] || !region.contiguous?
        unless materialize
          return buffer.view(region.offset, region.length, shape: region.shape)
        end

        out = Ops.allocate_output(node.dtype, region.length, region.shape)
        region.each_block do |src_offset, dst_offset, len|
          Graph.copy(buffer.view(src_offset, len), out.view(dst_offset, len))
        end
        out
      end

      def run_state_write(node, inputs, bindings)
        value = inputs[1]
        # If the producer already wrote straight into the target (placement
        # took effect), there is nothing left to copy.
        return node.state.buffer if @placed[node.inputs[1].id]

        region = node.region.resolve(bindings)
        buffer = node.state.buffer
        region.each_block do |dst_offset, src_offset, len|
          source = value.length == len ? value : value.view(src_offset, len)
          Graph.copy(source, buffer.view(dst_offset, len))
        end
        buffer
      end

      def run_binary(node, inputs, _bindings)
        op = node.attrs[:op]
        operand = node.attrs.key?(:scalar) ? node.attrs[:scalar] : inputs[1]
        case node.attrs[:kind]
        when :compare then Ops.compare(op, inputs[0], operand)
        else Ops.arithmetic(op, inputs[0], operand, reversed: node.attrs[:reversed] || false)
        end
      end

      def run_unary(node, inputs, _bindings)
        fn = node.attrs[:fn]
        fn == :abs ? Ops.unary_abs(inputs[0]) : Ops.unary_math(fn, inputs[0])
      end

      # A whole-buffer reduction takes Ops' device: true path (tree folding). It
      # never reaches the host, so the result passes on as a one-element Buffer.
      def run_reduce(node, inputs, _bindings)
        kind = node.attrs[:kind]
        return launch_axis_reduce(node, inputs[0]) unless node.attrs[:axis].nil?
        return Ops.mean_op(inputs[0], device: true) if kind == :mean

        Ops.reduce(kind, inputs[0], device: true)
      end

      def run_arg_reduce(node, inputs, _bindings)
        return launch_axis_reduce(node, inputs[0]) unless node.attrs[:axis].nil?

        Ops.arg_reduce(node.attrs[:kind], inputs[0], device: true)
      end

      # An axis reduction is one kernel. The input is viewed as
      # [outer, axis_len, inner], one thread per output element folding axis_len
      # values. outer and inner are the products of the dims before and after
      # the axis, so any rank works the same.
      def launch_axis_reduce(node, input)
        shape = node.inputs.first.shape
        axis = node.attrs[:axis]
        axis_len = shape[axis]
        inner = shape.drop(axis + 1).inject(1, :*)
        kernel =
          if node.op == :arg_reduce
            Kernels.axis_arg_reduce(kind: node.attrs[:kind], in_dtype: input.dtype)
          else
            Kernels.axis_reduce(kind: node.attrs[:kind], in_dtype: input.dtype, out_dtype: node.dtype)
          end

        out = Ops.allocate_output(node.dtype, node.length, node.shape)
        Ops.launch_kernel(kernel, grid: (node.length / Ops::BLOCK.to_f).ceil, block: Ops::BLOCK,
                                  args: [flat(input), out, axis_len, inner, node.length])
        out
      end

      def run_cast(node, inputs, _bindings)
        kernel = Kernels.cast(from: node.inputs.first.dtype, to: node.dtype)
        launch_elementwise(node, kernel, [flat(inputs[0])])
      end

      # Arguments are always [cond, a, b, out, n]; the side that is a scalar is
      # passed as a value (which one is fixed at compile time).
      def run_where(node, inputs, _bindings)
        a_scalar = node.attrs[:a_scalar]
        b_scalar = node.attrs[:b_scalar]
        kernel = Kernels.select(dtype: node.dtype, a_scalar: !a_scalar.nil?, b_scalar: !b_scalar.nil?)
        rest = inputs.drop(1).map { |buffer| flat(buffer) }
        a = a_scalar || rest.shift
        b = b_scalar || rest.shift
        launch_elementwise(node, kernel, [flat(inputs[0]), a, b])
      end

      def run_iota(node, _inputs, _bindings)
        kernel = Kernels.iota(shape: node.shape, axis: node.attrs[:axis], dtype: node.dtype)
        launch_elementwise(node, kernel, [])
      end

      # Only changes the shape. **No kernel runs**; a view of the input buffer is
      # returned. The view points at the original memory, so the original must
      # stay alive until this node's consumers finish (alias_root in plan_releases).
      def run_reshape(node, inputs, _bindings)
        inputs[0].reshape(node.shape)
      end

      # One thread per output element; shared by the three above and broadcast.
      def launch_elementwise(node, kernel, args)
        out = Ops.allocate_output(node.dtype, node.length, node.shape)
        Ops.launch_kernel(kernel, grid: (node.length / Ops::BLOCK.to_f).ceil, block: Ops::BLOCK,
                                  args: [*args, out, node.length])
        out
      end

      # Materializes an index remap (broadcast / permute). Only reached when
      # fusion did not fold it.
      def run_remap(node, inputs, _bindings)
        kernel = Kernels.remap(dims: Kernels.remap_dims(node), dtype: node.dtype, name: node.op.to_s)
        launch_elementwise(node, kernel, [flat(inputs[0])])
      end

      # An outermost-dimension slice. A view like reshape, so no kernel runs.
      def run_slice(node, inputs, _bindings)
        inputs[0].view(node.attrs[:offset], node.length, shape: node.shape)
      end

      def run_gather(node, inputs, _bindings)
        source = node.inputs.first
        axis = node.attrs[:axis]
        inner = source.shape.drop(axis + 1).inject(1, :*)
        kernel = Kernels.gather(dtype: node.dtype, axis_len: source.shape[axis],
                                idx_len: node.inputs[1].length, inner: inner)
        launch_elementwise(node, kernel, [flat(inputs[0]), flat(inputs[1])])
      end

      # The inverse of gather: one thread per output element scans the index
      # list (see the note on Kernels.scatter_add).
      def run_scatter_add(node, inputs, _bindings)
        axis = node.attrs[:axis]
        inner = node.shape.drop(axis + 1).inject(1, :*)
        kernel = Kernels.scatter_add(dtype: node.dtype, axis_len: node.shape[axis],
                                     idx_len: node.inputs[0].length, inner: inner)
        launch_elementwise(node, kernel, [flat(inputs[0]), flat(inputs[1])])
      end

      def run_pad(node, inputs, _bindings)
        source = node.inputs.first
        strides = Kernels.row_major_strides(source.shape)
        dims = node.shape.each_with_index.map do |dim, d|
          [dim, source.shape[d], node.attrs[:before][d], strides[d]]
        end
        kernel = Kernels.pad(dims: dims, dtype: node.dtype, value: node.attrs[:value])
        out = Ops.allocate_output(node.dtype, node.length, node.shape)
        Ops.launch_kernel(kernel, grid: (node.length / Ops::BLOCK.to_f).ceil, block: Ops::BLOCK,
                                  args: [flat(inputs[0]), out, node.attrs[:value], node.length])
        out
      end

      # Kernels see indices as 1D, so multidimensional buffers are passed
      # flattened (as in Graph.copy).
      def flat(buffer) = buffer.shape.length > 1 ? buffer.reshape(buffer.length) : buffer

      # Runs a fused cluster as one kernel. The output takes the same two forms
      # as everything else (dialect source / :cpu Ruby source), so it rides
      # Kernel's SHA cache as is — clusters with the same expression compile
      # once even across layers.
      def run_fused(cluster, inputs)
        return run_row_fused(cluster, inputs) if cluster.row_wise?

        kernel = Ops.compiled(
          dialect_name: "fused", dialect_src: Fusion.dialect_source(cluster),
          cpu_name: "fused",
          cpu_src: lambda {
            Ops.codegen.fused(cluster.expr,
                              input_dtypes: cluster.inputs.map(&:dtype),
                              scalar_dtypes: cluster.scalars.map { |s| s[:dtype] },
                              out_dtype: cluster.dtype)
          },
        )
        out = Ops.allocate_output(cluster.dtype, cluster.length, cluster.shape)
        args = inputs + cluster.scalars.map { |s| s[:value] } + [out, cluster.length]
        Ops.launch_kernel(kernel, grid: (cluster.length / Ops::BLOCK.to_f).ceil, block: Ops::BLOCK, args: args)
        out
      end

      # A row-wise cluster. **The launch domain is rows, not elements** (one
      # thread per row); only the launch differs from elementwise clusters.
      def run_row_fused(cluster, inputs)
        kernel = Ops.compiled(
          dialect_name: "row_fused", dialect_src: Fusion.dialect_source(cluster),
          cpu_name: "row_fused",
          cpu_src: lambda {
            Ops.codegen.row_fused(cluster.expr, reductions: cluster.reductions,
                                                input_dtypes: cluster.inputs.map(&:dtype),
                                                scalar_dtypes: cluster.scalars.map { |s| s[:dtype] },
                                                out_dtype: cluster.dtype)
          },
        )
        out = Ops.allocate_output(cluster.dtype, cluster.length, cluster.shape)
        args = inputs + cluster.scalars.map { |s| s[:value] } + [out, cluster.rows, cluster.cols]
        Ops.launch_kernel(kernel, grid: (cluster.rows / Ops::BLOCK.to_f).ceil, block: Ops::BLOCK, args: args)
        out
      end

      def run_matmul(node, inputs, _bindings)
        unless node.attrs[:mode] == :batched
          return Ops.matmul_op(inputs[0], inputs[1], out_dtype: node.attrs.fetch(:out_dtype, node.dtype))
        end

        run_batched_matmul(node, inputs[0], inputs[1])
      end

      # [b, m, k] x [b, k, n]; a strided batched GEMM does it in one call.
      #
      # The bridge's GEMM is **column-major** (see tensor_core.rb), so row-major
      # OUT = A·B is reinterpreted transposed: OUT^T = B^T · A^T, i.e. swap A
      # and B and swap m and n as column-major arguments. Each ld is its
      # row-major width, which is exactly the column-major leading dimension.
      def run_batched_matmul(node, a, b)
        batch, m, k = a.shape
        n = b.shape[2]
        compute_dtype = node.dtype == :int32 ? :int32 : :float32
        compute_out = Ops.allocate_output(compute_dtype, node.length, node.shape)
        if batched_gemm_usable?(node, a, b)
          TensorCore.gemm_batched(b, a, compute_out,
                                  m: n, k: k, n: m,
                                  lda: n, ldb: k, ldc: n,
                                  stride_a: k * n, stride_b: m * k, stride_c: m * n,
                                  batch: batch)
          return convert_batched_output(compute_out, node.dtype)
        end

        # Without a delegate, fall back to per-batch matmuls. The output is
        # allocated here, so one extra copy per batch moves each result into place.
        batch.times do |i|
          part = Ops.matmul_op(a.slice(i).reshape(m, k), b.slice(i).reshape(k, n))
          Graph.copy(part, compute_out.view(i * m * n, m * n))
          part.free
        end
        convert_batched_output(compute_out, node.dtype)
      end

      def convert_batched_output(out, dtype)
        return out if out.dtype == dtype

        converted = out.cast(dtype)
        out.free
        converted
      end

      # gemm_batched applies only when both sides share a float type and the output is float32.
      def batched_gemm_usable?(_node, a, b)
        a.dtype == b.dtype && a.dtype != :int32 && TensorCore.gemm_batched? &&
          Tilework.capabilities.gemm_dtypes.include?(a.dtype)
      end

      # concat accepts only the outermost dim, so each piece is contiguous in
      # the output; one copy per piece suffices, with no scattered writes.
      def run_concat(node, inputs, _bindings)
        out = Ops.allocate_output(node.dtype, node.length, node.shape)
        axis = node.attrs[:axis] || 0
        return concat_contiguous(out, inputs) if axis.zero?

        inner = node.shape.drop(axis + 1).inject(1, :*)
        offset = 0
        node.inputs.each_with_index do |piece, i|
          kernel = Kernels.concat_piece(dtype: node.dtype, piece_axis: piece.shape[axis],
                                        total_axis: node.shape[axis], inner: inner, offset: offset)
          Ops.launch_kernel(kernel, grid: (piece.length / Ops::BLOCK.to_f).ceil, block: Ops::BLOCK,
                                    args: [flat(inputs[i]), out, piece.length])
          offset += piece.shape[axis]
        end
        out
      end

      # Along the outermost dim pieces are contiguous in the output, so one copy each suffices.
      def concat_contiguous(out, inputs)
        offset = 0
        inputs.each do |buffer|
          Graph.copy(buffer, out.view(offset, buffer.length))
          offset += buffer.length
        end
        out
      end

      # Derives each node's key from (plan seed, static index). Philox is
      # counter-based and takes its key as an argument, so this alone makes RNG
      # **pure** — independent of execution order, same seed same sequence.
      def run_random(node, _inputs, bindings)
        seed = bindings.fetch(:seed) do
          raise Error, "plan.run needs a seed: this graph draws #{@builder.random_draws} random value(s)"
        end
        raise TypeError, "seed must be an Integer, got #{seed.class}" unless seed.is_a?(Integer)

        out = Ops.allocate_output(node.dtype, node.length, node.shape)
        key = Plan.derive_key(seed, node.attrs[:index])
        if node.attrs[:kind] == :normal
          Random.normal!(out, mean: node.attrs[:mean], stddev: node.attrs[:stddev], seed: key)
        else
          Random.uniform!(out, seed: key)
        end
        out
      end

      # Calls the same NN function that was recorded. There is no graph-specific
      # implementation, so values cannot drift from eager (two diverging copies
      # of the math is the biggest risk #17 named).
      def run_nn(node, inputs, _bindings)
        arity = node.attrs[:arity]
        kwargs = node.attrs[:opts].merge(node.attrs[:keys].zip(inputs.drop(arity)).to_h)
        result = NN.public_send(node.attrs[:fn], *inputs.first(arity), *node.attrs[:args], **kwargs)
        check_result!(result, node, "#{node.attrs[:fn]} node")
      end

      # Calls the delegate (cuDNN / MPSGraph) as is. No graph-specific
      # implementation, so values cannot drift from eager (same idea as :nn).
      def run_conv2d(node, inputs, _bindings)
        result = Conv.forward(inputs[0], inputs[1], pad: node.attrs[:pad], stride: node.attrs[:stride],
                              dilation: node.attrs[:dilation], groups: node.attrs[:groups])
        check_result!(result, node, "conv2d node")
      end

      # Backward calls the delegate as is too (same idea as run_conv2d — no
      # graph-specific implementation, so no drift from eager). inputs[2] is
      # there only for its shape.
      def run_conv2d_backward_data(node, inputs, _bindings)
        result = Conv.backward_data(inputs[0], inputs[1], inputs[2],
                                    pad: node.attrs[:pad], stride: node.attrs[:stride],
                                    dilation: node.attrs[:dilation], groups: node.attrs[:groups])
        check_result!(result, node, "conv2d backward data node")
      end

      def run_conv2d_backward_filter(node, inputs, _bindings)
        result = Conv.backward_filter(inputs[0], inputs[1], inputs[2],
                                      pad: node.attrs[:pad], stride: node.attrs[:stride],
                                      dilation: node.attrs[:dilation], groups: node.attrs[:groups])
        check_result!(result, node, "conv2d backward filter node")
      end

      def run_opaque(node, inputs, _bindings)
        result = node.attrs[:body].call(*inputs)
        check_result!(result, node, "opaque node #{node.attrs[:name].inspect}")
      end

      def check_result!(result, node, what)
        unless result.is_a?(Buffer) && result.dtype == node.dtype && result.shape == node.shape
          got = result.is_a?(Buffer) ? "#{result.shape.inspect} #{result.dtype}" : result.class.to_s
          raise Error, "#{what} declared #{node.shape.inspect} #{node.dtype} but returned #{got}"
        end

        result
      end
    end
  end
end
