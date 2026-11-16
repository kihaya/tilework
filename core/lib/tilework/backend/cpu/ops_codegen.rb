# frozen_string_literal: true

module Tilework
  module CPU
    # The :cpu backend for the array-op layer. Generates Ruby source for the
    # same op set as lib/tilework/ops.rb (arithmetic, comparisons, elementwise
    # composition, math functions, clamp, matmul). ops.rb is left untouched.
    #
    # Whole-buffer reduce/arg_reduce are not here: threadgroup reductions with
    # GPU_SHARED/GPU_BARRIER would need an interpreter that emulates
    # barrier-separated phases to reproduce correctly on the CPU, which is not
    # worth it for two uses internal to Ops (not exposed to user
    # Tilework.kernel {}). Instead ops.rb computes them directly in Ruby when
    # Tilework.host_fallback? is true, bypassing kernel generation.
    #
    # The generated source follows the same convention as
    # lib/tilework/backend/cpu/dsl_codegen.rb:
    # `{ arg_kinds: [...], proc: lambda { |bufs, scalars, gx, gy, gz, lx, ly,
    # lz, bx, by, bz| ... } }`. The argument order must match what
    # lib/tilework/ops.rb passes to `kernel.launch(args: [...])`.
    module OpsCodegen
      # A descriptor for one argument. role: :buffer | :scalar, dtype: :float32 | :int32.
      Arg = Struct.new(:name, :role, :dtype)

      module_function

      def arithmetic_expr(op)
        ->(lhs, rhs) { "(#{lhs} #{op} #{rhs})" }
      end

      def compare_expr(op)
        ->(lhs, rhs) { "((#{lhs} #{op} #{rhs}) ? 1 : 0)" }
      end

      def binary_fn_expr(fn)
        case fn
        when "max" then ->(lhs, rhs) { "[#{lhs}, #{rhs}].max" }
        when "min" then ->(lhs, rhs) { "[#{lhs}, #{rhs}].min" }
        else ->(lhs, rhs) { "#{fn}(#{lhs}, #{rhs})" }
        end
      end

      # --- Elementwise binary (Buffer op Buffer) --- args: [a, b, out, n]
      def buffer_binary(_name, a_dtype:, b_dtype:, out_dtype:, compute_dtype:) # rubocop:disable Lint/UnusedMethodArgument
        cast = out_dtype == :int32 ? "to_i" : "to_f"
        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = (#{yield("a[i]", "b[i]")}).#{cast}
          end
        RUBY
        wrap([Arg.new("a", :buffer, a_dtype), Arg.new("b", :buffer, b_dtype),
              Arg.new("c", :buffer, out_dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Elementwise binary (Buffer op scalar) --- args: [a, s, out, n]
      def scalar_binary(_name, a_dtype:, out_dtype:, compute_dtype:, reversed: false) # rubocop:disable Lint/UnusedMethodArgument
        lhs, rhs = "a[i]", "s"
        lhs, rhs = rhs, lhs if reversed
        cast = out_dtype == :int32 ? "to_i" : "to_f"
        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = (#{yield(lhs, rhs)}).#{cast}
          end
        RUBY
        wrap([Arg.new("a", :buffer, a_dtype), Arg.new("s", :scalar, compute_dtype),
              Arg.new("c", :buffer, out_dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Unary (math functions, abs) --- args: [a, out, n]
      def unary(_name, a_dtype:, out_dtype:)
        cast = out_dtype == :int32 ? "to_i" : "to_f"
        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = (#{yield('a[i]')}).#{cast}
          end
        RUBY
        wrap([Arg.new("a", :buffer, a_dtype), Arg.new("c", :buffer, out_dtype),
              Arg.new("n", :scalar, :int32)], body)
      end

      def unary_math(fn, a_dtype:)
        unary("ew_#{fn}", a_dtype: a_dtype, out_dtype: :float32) { |x| "Math.#{fn}((#{x}).to_f)" }
      end

      def unary_abs(a_dtype:)
        unary("abs_#{a_dtype}", a_dtype: a_dtype, out_dtype: a_dtype) { |x| "(#{x}).abs" }
      end

      # --- clamp --- args: [a, lo, hi, out, n]
      def clamp(a_dtype:, dtype:)
        cast = dtype == :int32 ? "to_i" : "to_f"
        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = (a[i] < lo ? lo : (a[i] > hi ? hi : a[i])).#{cast}
          end
        RUBY
        wrap([Arg.new("a", :buffer, a_dtype), Arg.new("lo", :scalar, dtype), Arg.new("hi", :scalar, dtype),
              Arg.new("c", :buffer, dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Matrix multiply --- args: [a, b, out, m, k, n]
      def matmul(a_dtype:, b_dtype:, dtype:)
        cast = dtype == :int32 ? "to_i" : "to_f"
        body = <<~RUBY
          i = gx
          if (i < m * n)
            row = i / n
            col = i % n
            acc = 0
            (0...k).each do |t|
              acc = acc + a[row * k + t] * b[t * n + col]
            end
            c[i] = acc.#{cast}
          end
        RUBY
        wrap([Arg.new("a", :buffer, a_dtype), Arg.new("b", :buffer, b_dtype), Arg.new("c", :buffer, dtype),
              Arg.new("m", :scalar, :int32), Arg.new("k", :scalar, :int32), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Cast --- args: [a, out, n]
      #
      # to_i truncates toward zero, matching the dialect's C cast (Float#round
      # or floor would not).
      def cast(from:, to:)
        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = (a[i]).#{to == :int32 ? 'to_i' : 'to_f'}
          end
        RUBY
        wrap([Arg.new("a", :buffer, from), Arg.new("c", :buffer, to),
              Arg.new("n", :scalar, :int32)], body)
      end

      # --- Select by condition --- args: [cond, a, b, out, n]
      #
      # The positions are always [cond, a, b, out, n]; only the side that became
      # a scalar changes role (the same order as the dialect side).
      def select(dtype:, a_scalar:, b_scalar:)
        cast = dtype == :int32 ? "to_i" : "to_f"
        left = a_scalar ? "a" : "a[i]"
        right = b_scalar ? "b" : "b[i]"
        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = ((cond[i] != 0) ? #{left} : #{right}).#{cast}
          end
        RUBY
        wrap([Arg.new("cond", :buffer, :int32),
              Arg.new("a", a_scalar ? :scalar : :buffer, dtype),
              Arg.new("b", b_scalar ? :scalar : :buffer, dtype),
              Arg.new("c", :buffer, dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Index along an axis --- args: [out, n]
      def iota(dtype:, size:, inner:)
        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = ((i / #{inner}) % #{size}).#{dtype == :int32 ? 'to_i' : 'to_f'}
          end
        RUBY
        wrap([Arg.new("c", :buffer, dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Axis reduction (Tilework::Graph::Kernels) --- args: [a, out, axis_len, inner, n]
      #
      # As noted at the top, **whole-buffer** reduce/arg_reduce use
      # GPU_SHARED/GPU_BARRIER and never come here. An axis reduction is one
      # thread per output element folding in order, with no shared memory or
      # barrier, so it is generated in the same shape as the dialect kernel.
      def axis_reduce(kind:, in_dtype:, out_dtype:)
        cast = out_dtype == :int32 ? "to_i" : "to_f"
        body = <<~RUBY
          i = gx
          if (i < n)
            base = (i / inner) * axis_len * inner + (i % inner)
            acc = a[base]
            (1...axis_len).each do |t|
              v = a[base + t * inner]
              #{axis_combine(kind)}
            end
            c[i] = (#{kind == :mean ? 'acc / axis_len.to_f' : 'acc'}).#{cast}
          end
        RUBY
        wrap([Arg.new("a", :buffer, in_dtype), Arg.new("c", :buffer, out_dtype),
              Arg.new("axis_len", :scalar, :int32), Arg.new("inner", :scalar, :int32),
              Arg.new("n", :scalar, :int32)], body)
      end

      # --- Index reduction along an axis --- args: [a, out, axis_len, inner, n]
      def axis_arg_reduce(kind:, in_dtype:)
        cmp = kind == :max ? ">" : "<"
        body = <<~RUBY
          i = gx
          if (i < n)
            base = (i / inner) * axis_len * inner + (i % inner)
            best = a[base]
            at = 0
            (1...axis_len).each do |t|
              v = a[base + t * inner]
              if (v #{cmp} best)
                best = v
                at = t
              end
            end
            c[i] = at
          end
        RUBY
        wrap([Arg.new("a", :buffer, in_dtype), Arg.new("c", :buffer, :int32),
              Arg.new("axis_len", :scalar, :int32), Arg.new("inner", :scalar, :int32),
              Arg.new("n", :scalar, :int32)], body)
      end

      # --- Table lookup --- args: [a, idx, out, n]
      def gather(dtype:, axis_len:, idx_len:, inner:)
        body = <<~RUBY
          i = gx
          if (i < n)
            block = #{idx_len * inner}
            at = idx[(i % block) / #{inner}].to_i
            at = at < 0 ? 0 : (at > #{axis_len - 1} ? #{axis_len - 1} : at)
            c[i] = a[(i / block) * #{axis_len * inner} + at * #{inner} + (i % #{inner})]
          end
        RUBY
        wrap([Arg.new("a", :buffer, dtype), Arg.new("idx", :buffer, :int32),
              Arg.new("c", :buffer, dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Inverse of table lookup (scatter-add) --- args: [idx, dy, out, n]
      def scatter_add(dtype:, axis_len:, idx_len:, inner:)
        body = <<~RUBY
          i = gx
          if (i < n)
            block = #{axis_len * inner}
            outer = i / block
            j = (i % block) / #{inner}
            lane = i % #{inner}
            acc = 0.0
            t = 0
            while t < #{idx_len}
              at = idx[t].to_i
              at = at < 0 ? 0 : (at > #{axis_len - 1} ? #{axis_len - 1} : at)
              acc = acc + dy[outer * #{idx_len * inner} + t * #{inner} + lane] if at == j
              t = t + 1
            end
            c[i] = acc
          end
        RUBY
        wrap([Arg.new("idx", :buffer, :int32), Arg.new("dy", :buffer, dtype),
              Arg.new("c", :buffer, dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- Border fill --- args: [a, out, fill, n]
      def pad(dtype:, dims:)
        lines = dims.reverse.map do |out_dim, in_dim, before, stride|
          coord = "p = rem % #{out_dim}; rem /= #{out_dim}"
          if before.zero? && in_dim == out_dim
            "#{coord}; src += p * #{stride}"
          else
            "#{coord}; p -= #{before}; inside = inside && p >= 0 && p < #{in_dim}; " \
              "src += (p < 0 || p >= #{in_dim} ? 0 : p) * #{stride}"
          end
        end
        body = <<~RUBY
          i = gx
          if (i < n)
            rem = i
            src = 0
            inside = true
            p = 0
            #{lines.join("\n    ")}
            c[i] = inside ? a[src] : fill
          end
        RUBY
        wrap([Arg.new("a", :buffer, dtype), Arg.new("c", :buffer, dtype),
              Arg.new("fill", :scalar, dtype), Arg.new("n", :scalar, :int32)], body)
      end

      # --- One concat piece --- args: [piece, out, n]
      def concat_piece(dtype:, piece_axis:, total_axis:, inner:, offset:)
        body = <<~RUBY
          i = gx
          if (i < n)
            block = #{piece_axis * inner}
            c[(i / block) * #{total_axis * inner} + (#{offset} + (i % block) / #{inner}) * #{inner} +
              (i % #{inner})] = a[i]
          end
        RUBY
        wrap([Arg.new("a", :buffer, dtype), Arg.new("c", :buffer, dtype),
              Arg.new("n", :scalar, :int32)], body)
      end

      # --- Index remap (broadcast / axis permutation) --- args: [a, out, n]
      #
      # dims is a list of [output size, input-side stride] from dim 0. A
      # stretched dim has stride 0, so its term drops out of the index.
      def remap(dtype:, dims:)
        lines = dims.reverse.map do |size, stride|
          stride.zero? ? "rem /= #{size}" : "src += (rem % #{size}) * #{stride}; rem /= #{size}"
        end
        body = <<~RUBY
          i = gx
          if (i < n)
            rem = i
            src = 0
            #{lines.join("\n    ")}
            c[i] = a[src]
          end
        RUBY
        wrap([Arg.new("a", :buffer, dtype), Arg.new("c", :buffer, dtype),
              Arg.new("n", :scalar, :int32)], body)
      end

      def axis_combine(kind)
        case kind
        when :sum, :mean then "acc = acc + v"
        when :min        then "acc = v < acc ? v : acc"
        when :max        then "acc = v > acc ? v : acc"
        else raise Error, "unknown reduction #{kind.inspect}"
        end
      end

      # --- Fused elementwise (Tilework::Graph::Fusion) --- args: [a0.., s0.., out, n]
      #
      # core builds the expression tree (graph/fusion.rb); this only renders it
      # as Ruby while **writing it the way the unfused path does**: no casts on
      # reads, and each node's result rounded to its output type. That makes
      # "fused == unfused" hold bit for bit.
      def fused(expr, input_dtypes:, scalar_dtypes:, out_dtype:)
        args = input_dtypes.each_with_index.map { |dtype, i| Arg.new("a#{i}", :buffer, dtype) }
        args += scalar_dtypes.each_with_index.map { |dtype, i| Arg.new("s#{i}", :scalar, dtype) }
        args << Arg.new("c", :buffer, out_dtype)
        args << Arg.new("n", :scalar, :int32)

        body = <<~RUBY
          i = gx
          if (i < n)
            c[i] = #{fused_expr(expr)}
          end
        RUBY
        wrap(args, body)
      end

      # --- Row-wise fusion (Tilework::Graph::Fusion) --- args: [a0.., s0.., out, rows, cols]
      #
      # The key is rounding the reduction's result through store. The unfused
      # path writes the reduction to a buffer, rounding it to single precision;
      # without reproducing that here the low bits would differ between fused
      # and unfused (the same reason each node is rounded in elementwise fusion).
      def row_fused(expr, reductions:, input_dtypes:, scalar_dtypes:, out_dtype:)
        args = input_dtypes.each_with_index.map { |dtype, i| Arg.new("a#{i}", :buffer, dtype) }
        args += scalar_dtypes.each_with_index.map { |dtype, i| Arg.new("s#{i}", :scalar, dtype) }
        args << Arg.new("c", :buffer, out_dtype)
        args << Arg.new("rows", :scalar, :int32)
        args << Arg.new("cols", :scalar, :int32)

        passes = reductions.each_with_index.map do |reduction, k|
          <<~PASS.chomp
            acc#{k} = nil
            (0...cols).each do |ci|
              i = base + ci
              v = #{fused_expr(reduction[:expr])}
              acc#{k} = (ci == 0) ? v : (#{row_combine(reduction[:kind], "acc#{k}", 'v')})
            end
            red#{k} = #{store(row_finish(reduction[:kind], "acc#{k}"), reduction[:dtype])}
          PASS
        end

        body = <<~RUBY
          r = gx
          if (r < rows)
            base = r * cols
            #{passes.join("\n    ")}
            (0...cols).each do |ci|
              i = base + ci
              c[i] = #{fused_expr(expr)}
            end
          end
        RUBY
        wrap(args, body)
      end

      def row_combine(kind, acc, value)
        case kind
        when :sum, :mean then "#{acc} + #{value}"
        when :min        then "(#{value} < #{acc} ? #{value} : #{acc})"
        when :max        then "(#{value} > #{acc} ? #{value} : #{acc})"
        else raise Error, "cannot fuse the reduction #{kind.inspect}"
        end
      end

      def row_finish(kind, acc) = kind == :mean ? "(#{acc} / cols.to_f)" : acc

      def fused_expr(expr)
        case expr.kind
        # Index remaps (folded broadcasts) use the same expression as the
        # dialect side. Integer division truncates the same in Ruby and C.
        when :buffer  then "a#{expr.index}[#{Graph::Fusion.index_expr(expr.map)}]"
        when :reduced then "red#{expr.index}"
        when :scalar  then "s#{expr.index}"
        when :binary  then store("(#{fused_expr(expr.children[0])} #{expr.op} #{fused_expr(expr.children[1])})",
                                 expr.dtype)
        when :compare then store("((#{fused_expr(expr.children[0])} #{expr.op} " \
                                 "#{fused_expr(expr.children[1])}) ? 1 : 0)", expr.dtype)
        when :math    then store("Math.#{expr.op}((#{fused_expr(expr.children.first)}).to_f)", expr.dtype)
        when :abs     then store("(#{fused_expr(expr.children.first)}).abs", expr.dtype)
        when :cast    then store(fused_expr(expr.children.first), expr.dtype)
        when :select  then store("((#{fused_expr(expr.children[0])}) != 0 ? " \
                                 "#{fused_expr(expr.children[1])} : #{fused_expr(expr.children[2])})",
                                 expr.dtype)
        else raise Error, "unknown fused expression #{expr.kind.inspect}"
        end
      end

      # The rounding that corresponds to the unfused path's write to a buffer.
      #
      # The pack/unpack at float32 is not decoration: Ruby computes in double,
      # so fusion would **drop the per-node rounding to single precision from
      # the middle of the expression**. Without reproducing it here, the low
      # bits would differ between fused and unfused (dialect backends keep
      # intermediates in float, so they never have this problem).
      def store(rendered, dtype)
        return "(#{rendered}).to_i" if dtype == :int32

        "[#{rendered}].pack(\"f\").unpack1(\"f\")"
      end

      def wrap(args, body)
        arg_kinds = args.map { |a| :"#{a.dtype}_#{a.role}" }
        preamble = args.each_with_index.map do |a, idx|
          a.role == :buffer ? "#{a.name} = bufs[#{idx}]" : "#{a.name} = scalars[#{idx}]"
        end
        <<~RUBY
          {
            arg_kinds: #{arg_kinds.inspect},
            proc: lambda do |bufs, scalars, gx, gy, gz, lx, ly, lz, bx, by, bz|
              #{preamble.join("\n    ")}
          #{body}
            end,
          }
        RUBY
      end
      private_class_method :wrap
    end
  end
end
