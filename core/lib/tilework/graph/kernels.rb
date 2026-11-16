# frozen_string_literal: true

module Tilework
  module Graph
    # Kernels the graph layer owns: ops with no eager (Ops) counterpart, such
    # as axis reductions and index remaps.
    #
    # Their output takes the same two forms as everything else (dialect source /
    # :cpu Ruby source), so it rides Kernel's SHA cache as is. No branching on
    # backend names; Ops.compiled routes by capabilities[:dialect].
    #
    # ## Numerics
    #
    # Axis reductions **fold in order from the first element in one thread**.
    # Unlike Ops.reduce's tree, the order is fully fixed, so the same input gives
    # the same value every run. The accumulator is float (ACC) on the dialect
    # side and Ruby double on :cpu, so the two differ by what accumulates before
    # the final rounding — which is why conformance uses a tolerance for axis
    # reductions (bit identity as with fusion is not available).
    module Kernels
      module_function

      # Axis reduction. Viewed as [outer, axis_len, inner], one thread per
      # output element folds axis_len values. outer / inner / axis_len are
      # launch-time scalars, so the source does not change with the shape (no
      # recompilation).
      #
      # args: [a, out, axis_len, inner, n]
      def axis_reduce(kind:, in_dtype:, out_dtype:)
        acc = Ops::CTYPE.fetch(out_dtype)
        name = "axis_#{kind}_#{in_dtype}_#{out_dtype}"
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const #{Ops::CTYPE.fetch(in_dtype)}* a,
                                  device #{acc}* c,
                                  constant int& axis_len,
                                  constant int& inner,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) {
              int base = (i / inner) * axis_len * inner + (i % inner);
              #{acc} acc = (#{acc})a[base];
              for (int t = 1; t < axis_len; ++t) {
                #{acc} v = (#{acc})a[base + t * inner];
                #{combine_expr(kind)}
              }
              c[i] = #{finish_expr(kind, acc)};
            }
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: -> { Ops.codegen.axis_reduce(kind: kind, in_dtype: in_dtype, out_dtype: out_dtype) })
      end

      # Index reduction along an axis: returns positions, not values. Ties keep
      # the smaller index (left-biased like Ops.arg_reduce).
      #
      # args: [a, out, axis_len, inner, n]
      def axis_arg_reduce(kind:, in_dtype:)
        acc = Ops::CTYPE.fetch(Ops::LOW_PRECISION_DTYPES.include?(in_dtype) ? :float32 : in_dtype)
        cmp = kind == :max ? ">" : "<"
        name = "axis_arg#{kind}_#{in_dtype}"
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const #{Ops::CTYPE.fetch(in_dtype)}* a,
                                  device int* c,
                                  constant int& axis_len,
                                  constant int& inner,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) {
              int base = (i / inner) * axis_len * inner + (i % inner);
              #{acc} best = (#{acc})a[base];
              int at = 0;
              for (int t = 1; t < axis_len; ++t) {
                #{acc} v = (#{acc})a[base + t * inner];
                if (v #{cmp} best) { best = v; at = t; }
              }
              c[i] = at;
            }
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: -> { Ops.codegen.axis_arg_reduce(kind: kind, in_dtype: in_dtype) })
      end


      # Cast. It **only changes the storage type**, with C cast semantics (float
      # -> int truncates toward zero). :cpu's to_i rounds the same way, so both
      # paths give the same values.
      #
      # args: [a, out, n]
      def cast(from:, to:)
        name = "cast_#{from}_#{to}"
        read = Ops.read_dtype(from, "a[i]")
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const #{Ops::CTYPE.fetch(from)}* a,
                                  device #{Ops::CTYPE.fetch(to)}* c,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) c[i] = (#{Ops::CTYPE.fetch(to)})#{read};
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: -> { Ops.codegen.cast(from: from, to: to) })
      end

      # Select by condition. a / b may each be a buffer or a scalar, which the
      # host already knows (graph shapes are static), so the argument layout is
      # baked into the source. The positions are always [cond, a, b, out, n];
      # only the side that became a scalar turns into a constant reference.
      def select(dtype:, a_scalar:, b_scalar:)
        ct = Ops::CTYPE.fetch(dtype)
        name = "select_#{dtype}#{a_scalar ? '_sa' : ''}#{b_scalar ? '_sb' : ''}"
        operand = ->(sym, scalar) { scalar ? sym.to_s : "#{sym}[i]" }
        param = ->(sym, scalar) { scalar ? "constant #{ct}& #{sym}" : "device const #{ct}* #{sym}" }
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const int* cond,
                                  #{param.call(:a, a_scalar)},
                                  #{param.call(:b, b_scalar)},
                                  device #{ct}* c,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) c[i] = cond[i] ? (#{ct})#{operand.call(:a, a_scalar)} : (#{ct})#{operand.call(:b, b_scalar)};
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: -> { Ops.codegen.select(dtype: dtype, a_scalar: a_scalar, b_scalar: b_scalar) })
      end

      # Index along an axis: builds masks and positions without bringing
      # constants in from the host.
      #
      # args: [out, n]
      def iota(shape:, axis:, dtype:)
        size = shape[axis]
        inner = shape.drop(axis + 1).inject(1, :*)
        ct = Ops::CTYPE.fetch(dtype)
        name = "iota_#{dtype}"
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device #{ct}* c,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) c[i] = (#{ct})((i / #{inner}) % #{size});
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: -> { Ops.codegen.iota(dtype: dtype, size: size, inner: inner) })
      end

      # Table lookup. Each element of indices points at a position along
      # source's axis — exactly what an embedding does (pulling rows from a
      # vocabulary table).
      #
      # **Indices are clamped into range.** Reading out of range on a GPU is
      # undefined, so clamping to the ends beats breaking silently (:cpu clamps
      # the same way, so the answer does not depend on the path).
      #
      # args: [a, idx, out, n]
      def gather(dtype:, axis_len:, idx_len:, inner:)
        ct = Ops::CTYPE.fetch(dtype)
        name = "gather_#{dtype}"
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const #{ct}* a,
                                  device const int* idx,
                                  device #{ct}* c,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) {
              int block = #{idx_len * inner};
              int j = (i % block) / #{inner};
              int at = idx[j];
              at = at < 0 ? 0 : (at > #{axis_len - 1} ? #{axis_len - 1} : at);
              c[i] = a[(i / block) * #{axis_len * inner} + at * #{inner} + (i % #{inner})];
            }
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: lambda {
                       Ops.codegen.gather(dtype: dtype, axis_len: axis_len, idx_len: idx_len, inner: inner)
                     })
      end

      # The inverse of table lookup (scatter-add): adds cotangents back to where
      # gather read.
      #
      #   dsource[o, j, k] = Σ_i [ids[i] == j] · dy[o, i, k]
      #
      # **No atomics.** One thread per output element scans the index list and
      # adds only the entries addressed to it. Each thread owns its destination,
      # so nothing races, and the fold order is fixed, so **every run gives the
      # same value** (an atomic scatter-add moves the last bits as its order
      # varies). The same reason keeps GPU_ATOMIC_ADD out of the dialect — a
      # macro whose meaning differs across backends violates the compat header
      # policy.
      #
      # The work is index count × output elements, the same as a matmul with a
      # match matrix. The difference is not materializing the [table rows,
      # index count] mask, which matters when the table is large, as with
      # embeddings.
      #
      # args: [idx, dy, out, n]
      def scatter_add(dtype:, axis_len:, idx_len:, inner:)
        ct = Ops::CTYPE.fetch(dtype)
        acc = Ops::CTYPE.fetch(Ops::LOW_PRECISION_DTYPES.include?(dtype) ? :float32 : dtype)
        name = "scatter_add_#{dtype}"
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const int* idx,
                                  device const #{ct}* dy,
                                  device #{ct}* c,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) {
              int block = #{axis_len * inner};
              int outer = i / block;
              int j = (i % block) / #{inner};
              int lane = i % #{inner};
              #{acc} acc = (#{acc})0;
              for (int t = 0; t < #{idx_len}; ++t) {
                int at = idx[t];
                at = at < 0 ? 0 : (at > #{axis_len - 1} ? #{axis_len - 1} : at);
                if (at == j) acc += (#{acc})dy[outer * #{idx_len * inner} + t * #{inner} + lane];
              }
              c[i] = (#{ct})acc;
            }
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: lambda {
                       Ops.codegen.scatter_add(dtype: dtype, axis_len: axis_len,
                                               idx_len: idx_len, inner: inner)
                     })
      end

      # Fills a border. dims holds, per output dim,
      # [output size, input size, amount added in front, input-side stride].
      #
      # args: [a, out, n]
      def pad(dims:, dtype:, value:)
        ct = Ops::CTYPE.fetch(dtype)
        name = "pad_#{dtype}"
        lines = pad_lines(dims)
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const #{ct}* a,
                                  device #{ct}* c,
                                  constant #{ct}& fill,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) {
              int rem = i;
              int src = 0;
              int inside = 1;
              int p = 0;
          #{lines.map { |line| "    #{line}" }.join("\n")}
              c[i] = inside ? a[src] : fill;
            }
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: -> { Ops.codegen.pad(dtype: dtype, dims: dims) })
      end

      # Builds the index per dim, checking whether the coordinate falls inside the input.
      def pad_lines(dims)
        dims.reverse.map do |out_dim, in_dim, before, stride|
          coord = "p = rem % #{out_dim}; rem /= #{out_dim};"
          next "#{coord} src += p * #{stride};" if before.zero? && in_dim == out_dim

          "#{coord} p -= #{before}; " \
            "inside = inside && p >= 0 && p < #{in_dim}; " \
            "src += (p < 0 ? 0 : (p >= #{in_dim} ? 0 : p)) * #{stride};"
        end
      end

      # Writes one concat piece into its place in the output. Along the
      # outermost dim pieces are contiguous and one copy suffices (Plan picks
      # that); along an inner dim they are scattered, so each element's
      # destination is computed.
      #
      # args: [piece, out, n]
      def concat_piece(dtype:, piece_axis:, total_axis:, inner:, offset:)
        ct = Ops::CTYPE.fetch(dtype)
        name = "concat_piece_#{dtype}"
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const #{ct}* a,
                                  device #{ct}* c,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) {
              int block = #{piece_axis * inner};
              c[(i / block) * #{total_axis * inner} +
                (#{offset} + (i % block) / #{inner}) * #{inner} + (i % #{inner})] = a[i];
            }
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: lambda {
                       Ops.codegen.concat_piece(dtype: dtype, piece_axis: piece_axis,
                                                total_axis: total_axis, inner: inner, offset: offset)
                     })
      end

      # Index remap: decompose the output index per dim and rebuild the input
      # index. **Broadcast and axis permutation are the same kernel**; only the
      # table of "how far the input advances per output dim (stride)" differs:
      #
      #   broadcast  a stretched dim has stride 0 (always the same element)
      #   permute    output dim d takes the stride of input dim axes[d]
      #
      # The table is known on the host (graph shapes are static), so it is
      # **baked into the kernel as constants**; each shape combination compiles once.
      #
      # args: [a, out, n]
      def remap(dims:, dtype:, name: "remap")
        ctype = Ops::CTYPE.fetch(dtype)
        name = "#{name}_#{dtype}"
        src = <<~KERNEL
          GPU_KERNEL void #{name}(GPU_ARGS,
                                  device const #{ctype}* a,
                                  device #{ctype}* c,
                                  constant int& n) {
            int i = GLOBAL_ID_X;
            if (i < n) {
              int rem = i;
              int src = 0;
          #{dims.reverse.map { |dim| "    #{index_line(dim)}" }.join("\n")}
              c[i] = a[src];
            }
          }
        KERNEL

        Ops.compiled(dialect_name: name, dialect_src: src, cpu_name: name,
                     cpu_src: -> { Ops.codegen.remap(dtype: dtype, dims: dims) })
      end

      # The stride table for an index-remapping node (:broadcast / :permute).
      def remap_dims(node)
        source = node.inputs.first
        case node.op
        when :broadcast then broadcast_dims(source.shape, node.shape)
        when :permute   then permute_dims(source.shape, node.attrs[:axes])
        else raise Error, "#{node.op.inspect} does not remap indices"
        end
      end

      # [output size, input-side stride] per output dim. A stretched dim has
      # stride 0 — advancing along it always points at the same input element.
      def broadcast_dims(in_shape, out_shape)
        padded = Array.new(out_shape.length - in_shape.length, 1) + in_shape
        strides = row_major_strides(padded)
        out_shape.each_with_index.map do |dim, d|
          [dim, padded[d] == 1 && dim != 1 ? 0 : strides[d]]
        end
      end

      # Output dim d corresponds to input dim axes[d], so it takes that stride as is.
      def permute_dims(in_shape, axes)
        strides = row_major_strides(in_shape)
        axes.map { |axis| [in_shape[axis], strides[axis]] }
      end

      def row_major_strides(shape)
        stride = 1
        shape.reverse.map { |dim| stride.tap { stride *= dim } }.reverse
      end

      # Index arithmetic for one dim, peeling dims off from the last one.
      def index_line(dim)
        size, stride = dim
        return "rem /= #{size};" if stride.zero?

        "src += (rem % #{size}) * #{stride}; rem /= #{size};"
      end

      def combine_expr(kind)
        case kind
        when :sum, :mean then "acc = acc + v;"
        when :min        then "acc = v < acc ? v : acc;"
        when :max        then "acc = v > acc ? v : acc;"
        else raise Error, "unknown reduction #{kind.inspect}"
        end
      end

      def finish_expr(kind, acc) = kind == :mean ? "acc / (#{acc})axis_len" : "acc"
    end
  end
end
