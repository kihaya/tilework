# frozen_string_literal: true

module Tilework
  module Graph
    # A symbolic graph value.
    #
    # Arithmetic methods record nodes and return another tensor; they never execute a kernel.
    #
    # The operator surface matches Ops — the same expression runs immediately
    # on a Buffer and records into the graph on a Tensor. Type and shape rules
    # follow Ops too.
    class Tensor
      # @return [Builder] graph that owns this tensor
      # @return [Node] underlying graph node
      attr_reader :graph, :node

      def initialize(graph, node)
        @graph = graph
        @node = node
      end

      # @return [Symbol] tensor storage dtype
      def dtype = node.dtype
      # @return [Array<Integer>] logical shape
      def shape = node.shape
      # @return [Integer] element count
      def length = node.length
      # @return [Boolean] whether this tensor is two-dimensional
      def matrix? = shape.length == 2
      # @return [Integer] first dimension
      def rows = shape[0]
      # @return [Integer] second dimension
      def cols = shape[1]
      def symbolic? = true

      Ops::OPERATORS.each do |op|
        define_method(op) { |other| graph.arithmetic(op, self, other) }
      end

      Ops::COMPARISONS.each do |op|
        define_method(op) { |other| graph.compare(op, self, other) }
      end

      Ops::UNARY_MATH.each do |fn|
        define_method(fn) { graph.unary_math(fn, self) }
      end

      def eq(other) = graph.compare(:==, self, other)
      def ne(other) = graph.compare(:!=, self, other)
      def abs = graph.unary_abs(self)
      def matmul(other, out_dtype: nil) = graph.matmul(self, other, out_dtype: out_dtype)

      # Reductions. Unlike eager, they return a **Tensor, not a host number**
      # (dropping to the host mid-graph would end the plan). Without axis: the
      # whole tensor folds to shape [1] and runs on Ops' device: true path.
      def sum(axis: nil, keepdims: false) = graph.reduce(:sum, self, axis: axis, keepdims: keepdims)
      def min(axis: nil, keepdims: false) = graph.reduce(:min, self, axis: axis, keepdims: keepdims)
      def max(axis: nil, keepdims: false) = graph.reduce(:max, self, axis: axis, keepdims: keepdims)
      def mean(axis: nil, keepdims: false) = graph.reduce(:mean, self, axis: axis, keepdims: keepdims)
      def argmax(axis: nil, keepdims: false) = graph.arg_reduce(:max, self, axis: axis, keepdims: keepdims)
      def argmin(axis: nil, keepdims: false) = graph.arg_reduce(:min, self, axis: axis, keepdims: keepdims)

      # Broadcasts the tensor to a compatible shape.
      #
      # @param shape [Array<Integer>]
      # @return [Tensor]
      def broadcast_to(shape) = graph.broadcast(self, shape)

      # Changes only the logical shape, with at most one inferred `-1` dimension.
      #
      # @param dims [Array<Integer>]
      # @return [Tensor]
      def reshape(*dims) = graph.reshape(self, *dims)

      # Casts to a different storage dtype.
      #
      # @param dtype [:float32, :float16, :bfloat16, :int32]
      # @return [Tensor]
      def cast(dtype) = graph.cast(self, dtype)

      # Permutes tensor axes.
      #
      # @param axes [Array<Integer>] permutation of every axis
      # @return [Tensor]
      def permute(*axes) = graph.permute(self, *axes)

      # Slices the outermost dimension without a kernel launch.
      #
      # @param range [Integer, Range]
      # @return [Tensor]
      def slice(range) = graph.slice(self, range)

      # Gathers entries from this tensor using int32 indices.
      #
      # @param indices [Tensor]
      # @param axis [Integer]
      # @return [Tensor]
      def gather(indices, axis: 0) = graph.gather(self, indices, axis: axis)

      # Makes scalar-on-the-left expressions like 2.0 * tensor work (as Ops::Scalar does).
      def coerce(num) = [Scalar.new(self, num), self]

      def inspect = "#<Tilework::Graph::Tensor #{shape.inspect} #{dtype} (#{node.label})>"

      # The scalar-side proxy returned by coerce. Non-commutative ops record a
      # node with swapped operands; comparisons flip the operator.
      class Scalar
        def initialize(tensor, value)
          @tensor = tensor
          @value = value
        end

        def +(other) = other.graph.arithmetic(:+, other, @value)
        def *(other) = other.graph.arithmetic(:*, other, @value)
        def -(other) = other.graph.arithmetic(:-, other, @value, reversed: true)
        def /(other) = other.graph.arithmetic(:/, other, @value, reversed: true)
        def >(other) = other.graph.compare(:<, other, @value)
        def <(other) = other.graph.compare(:>, other, @value)
        def >=(other) = other.graph.compare(:<=, other, @value)
        def <=(other) = other.graph.compare(:>=, other, @value)
      end
    end
  end
end
