# frozen_string_literal: true

module Tilework
  module Graph
    # The target region of a write to / read from mutable state.
    #
    # Represented as **a strided block over the flat element space**
    # `(offset, block, stride, count)`: block elements from offset, repeated
    # count times every stride. With count == 1 it is just the contiguous
    # range `[offset, offset + block)`.
    #
    #   channels c0...c1 of [n, c, h, w] → offset = c0*h*w, block = (c1-c0)*h*w,
    #                                      stride = c*h*w,  count = n
    #
    # The reason for this form is overlap testing. Deciding that "writes to
    # disjoint regions need not be serialized" needs an exact test, which is
    # undecidable for arbitrary index expressions. Strided blocks express every
    # shape actually needed (outermost-dimension slices and channel ranges) and
    # the test is O(1).
    #
    # **Shapes it cannot decide answer "may overlap"** (#disjoint?). Erring on
    # the safe side loses only parallelism, never correctness.
    #
    # offset can be a Param as well as an Integer. An offset **decided at run
    # time**, like a KV cache's write position, is resolved from the plan's
    # bindings (`plan.run(pos: 3)`). static? is then false and the overlap test
    # always takes the safe side.
    class Region
      # An offset decided at run time; its value is `bias + bindings[name] * scale`.
      Param = Struct.new(:name, :scale, :bias) do
        def resolve(bindings)
          value = bindings.fetch(name) do
            raise Error, "plan.run is missing the runtime index #{name.inspect} " \
                         "(used as a write/read position)"
          end
          unless value.is_a?(Integer) && !value.negative?
            raise ArgumentError, "runtime index #{name.inspect} must be a non-negative Integer, got #{value.inspect}"
          end

          bias + (value * scale)
        end

        def key = [:param, name, scale, bias]
      end

      attr_reader :offset, :block, :stride, :count, :total, :shape

      # total is the target buffer's element count (for range checks). shape is
      # the logical shape of the tensor read from this region.
      def initialize(offset:, block:, count: 1, stride: nil, total:, shape: nil)
        @offset = offset
        @block = block
        @count = count
        @stride = stride || block
        @total = total
        @shape = shape || [block * count]
        validate!
      end

      # The whole buffer.
      def self.whole(shape)
        total = shape.inject(1, :*)
        new(offset: 0, block: total, total: total, shape: shape)
      end

      # A slice along the outermost dimension (Buffer#slice semantics); always
      # contiguous. range takes an Integer / Range or a run-time position (Index).
      def self.rows(shape, range)
        raise Error, "rows requires a shaped region (shape: #{shape.inspect})" if shape.length < 2

        row = shape.drop(1).inject(1, :*)
        total = shape.inject(1, :*)
        first, count = normalize(range, shape[0])
        offset = first.is_a?(Param) ? Param.new(first.name, first.scale * row, first.bias * row) : first * row
        new(offset: offset, block: count * row, total: total, shape: [count, *shape.drop(1)])
      end

      # A slice along any axis: the same contiguous region as rows for
      # axis == 0, a strided block for axis > 0.
      def self.axis(shape, axis, range)
        unless axis.is_a?(Integer) && (0...shape.length).cover?(axis)
          raise ArgumentError, "axis #{axis.inspect} is out of range for shape #{shape.inspect}"
        end
        return rows(shape, range) if axis.zero?

        inner = shape.drop(axis + 1).inject(1, :*)
        outer = shape.take(axis).inject(1, :*)
        first, count = normalize(range, shape[axis])
        if first.is_a?(Param)
          raise Error, "a runtime index is only supported on the outermost axis (got axis #{axis})"
        end

        out_shape = shape.dup
        out_shape[axis] = count
        new(offset: first * inner, block: count * inner, count: outer,
            stride: shape[axis] * inner, total: shape.inject(1, :*), shape: out_shape)
      end

      # Equivalent to Buffer#normalize_range (private there, so the rule is repeated here).
      def self.normalize(range, limit)
        case range
        when Index
          [Param.new(range.name, 1, 0), range.count]
        when Param
          [range, 1]
        when Integer
          idx = range.negative? ? limit + range : range
          raise ArgumentError, "index #{range} is out of bounds (#{limit})" unless (0...limit).cover?(idx)

          [idx, 1]
        when Range
          first = range.begin || 0
          first += limit if first.negative?
          last = range.end || limit
          last += limit if last.negative?
          last -= 1 if range.exclude_end? && range.end
          count = last - first + 1
          raise ArgumentError, "range #{range} is out of bounds (#{limit})" unless first >= 0 && count.positive? && first + count <= limit

          [first, count]
        else
          raise ArgumentError, "expected an Integer, a Range, or a runtime index, got #{range.class}"
        end
      end
      private_class_method :normalize

      def length = block * count
      def contiguous? = count == 1
      def static? = offset.is_a?(Integer)

      # Applies run-time bindings and returns a Region whose offset is a fixed Integer.
      def resolve(bindings)
        return self if static?

        concrete = Region.new(offset: offset.resolve(bindings), block: block, count: count,
                              stride: stride, total: total, shape: shape)
        concrete.bounds_check!
        concrete
      end

      def bounds_check!
        last = offset + ((count - 1) * stride) + block
        return if last <= total

        raise Error, "region #{inspect} runs past the end of a #{total}-element buffer"
      end

      # Yields [offset in the target buffer, offset in the value, length] per
      # block. The value (written from / read into) is treated as the selected
      # region packed contiguously.
      def each_block
        return enum_for(:each_block) unless block_given?

        count.times { |i| yield(offset + (i * stride), i * block, self.block) }
      end

      # **true only when the regions certainly do not overlap**; shapes it
      # cannot decide answer false.
      #
      #   - either has a run-time offset → false (conservative)
      #   - both contiguous → exact interval test
      #   - equal strides → exact test on the phase within the stride
      #   - different strides → true only when the overall spans are apart
      def disjoint?(other)
        return false unless static? && other.static?
        return spans_disjoint?(other) if contiguous? && other.contiguous?
        return spans_disjoint?(other) unless stride == other.stride
        return true if spans_disjoint?(other)

        # On the same lattice, disjoint if the phases within the stride do not overlap.
        phase = (other.offset - offset) % stride
        !(phase < self.block || stride - phase < other.block)
      end

      def spans_disjoint?(other)
        span_end <= other.offset || other.span_end <= offset
      end

      def span_end = offset + ((count - 1) * stride) + block

      # For the CSE key. A run-time offset compares by Param identity, not value.
      def key
        [static? ? offset : offset.key, block, stride, count, total, shape]
      end

      def inspect
        head = static? ? offset.to_s : "#{offset.name}*#{offset.scale}+#{offset.bias}"
        contiguous? ? "#<Region #{head}...+#{block}>" : "#<Region #{head}...+#{block} x#{count} stride #{stride}>"
      end

      private

      def validate!
        raise ArgumentError, "block must be a positive Integer" unless @block.is_a?(Integer) && @block.positive?
        raise ArgumentError, "count must be a positive Integer" unless @count.is_a?(Integer) && @count.positive?
        raise ArgumentError, "stride must be at least block" if @stride < @block
        unless @shape.inject(1, :*) == length
          raise ArgumentError, "region shape #{@shape.inspect} does not match #{length} elements"
        end

        bounds_check! if static?
      end
    end

    # A position decided at run time, resolved from `plan.run(pos: 3)` bindings.
    #
    #   g.write_into(kv0, at: g.index(:pos), value: k_new)
    Index = Struct.new(:name, :count)
  end
end
