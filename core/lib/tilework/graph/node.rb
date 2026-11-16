# frozen_string_literal: true

module Tilework
  module Graph
    # One graph node. It holds **no value** — only dtype and shape (the
    # metadata part of a Buffer) and how it is produced. Plan#run assigns the
    # actual buffer.
    #
    # A node only knows which op it is. How that op behaves (may it be freed,
    # does it run synchronously, can it fuse, how it prints) lives in the
    # Graph::Op table.
    #
    # order_deps are ordering edges that carry no value, added by
    # Plan#compile. They express the WAR rule for mutable state ("a write runs
    # after the readers").
    Node = Struct.new(:id, :op, :inputs, :attrs, :dtype, :shape) do
      def length = shape.inject(1, :*)
      def state = attrs[:state]
      def region = attrs[:region]
      def spec = Op.fetch(op)
      def writes_state? = spec.writes_state?
      def label = spec.label_for(self)

      def to_s = "##{id} #{label} #{shape.inspect} #{dtype}"
    end
  end
end
