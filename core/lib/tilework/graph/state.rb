# frozen_string_literal: true

module Tilework
  module Graph
    # Mutable state brought into the graph: buffers that **outlive a plan
    # run**, such as KV caches, optimizer moments and gradient accumulators.
    #
    # It binds an existing Buffer as is (no reallocation). Only two things are
    # special from the graph's point of view:
    #
    #   - it is live-in and live-out of the plan, so it is never reused as an
    #     intermediate
    #   - a write advances its version, and a read depends on a version
    class State
      attr_reader :id, :name, :buffer, :dtype, :shape
      attr_accessor :leaves, :version_count

      def initialize(id, buffer, name)
        @id = id
        @name = name || :"state_#{id}"
        @buffer = buffer
        @dtype = buffer.dtype
        @shape = buffer.shape
        @version_count = 0
        @leaves = []
      end

      def next_version = @version_count += 1
      def to_s = name.to_s
    end

    # A reference to **one version** of a state. `write_into` does not mutate
    # in place; it returns a new StateRef (the next version). Reads depend on
    # the version, so a real data edge connects a write and its readers and the
    # scheduler cannot reorder them.
    #
    # Unlike SSA, a version points at the same memory rather than a new value,
    # which is why WAR ordering is needed as well as RAW (added by Plan#compile).
    class StateRef
      attr_reader :graph, :state, :version, :node

      def initialize(graph, state, version, node)
        @graph = graph
        @state = state
        @version = version
        @node = node
      end

      def dtype = state.dtype
      def shape = state.shape

      # Region constructors. They are relative to the state's shape, so they live here.
      def rows(range) = Region.rows(state.shape, range)
      def axis(axis, range) = Region.axis(state.shape, axis, range)
      def whole = Region.whole(state.shape)

      # Shorthand delegations (the same as g.write_into / g.read).
      def write(value, at: nil) = graph.write_into(self, value: value, at: at)
      def read(at = nil) = graph.read(self, at)

      def inspect = "#<Tilework::Graph::StateRef #{state.name} v#{version}>"
    end
  end
end
