# frozen_string_literal: true

module Tilework
  # A DSL that turns kernels written as Ruby blocks into the CUDA-flavored
  # dialect. Argument types are unknown when the block is defined, so a
  # signature is built from the actual arguments at call time and source is
  # specialized per signature (the output is ordinary dialect, so it rides
  # the Dialect layer and the PSO cache as is).
  #
  # The supported Ruby subset is a whitelist; anything else is a CompileError
  # with a line number. The AST comes from RubyVM::AbstractSyntaxTree (which
  # ties the DSL to MRI).
  module DSL
    BLOCK = 256

    BUILTIN_IDS = {
      global_id_x: "GLOBAL_ID_X",
      global_id_y: "GLOBAL_ID_Y",
      local_id_x: "LOCAL_ID_X",
      group_id_x: "GROUP_ID_X",
    }.freeze
    MATH_FNS = { exp: 1, log: 1, sqrt: 1, sin: 1, cos: 1, tanh: 1, pow: 2 }.freeze
    ARITH_OPS = %i[+ - * / %].freeze
    COMPARE_OPS = %i[== != < <= > >=].freeze

    # One signature element: buffers carry an element type, scalars a value
    # type. shape is [rows, cols] when the argument is a :buffer with a 2D
    # shape (used to expand buf[r, c] and as compile-time constants for
    # buf.rows / buf.cols), nil otherwise. storage is the in-memory spelling;
    # only float16 differs from type (:float). Half precision is **a storage
    # type only** and always widened to float in expressions (in a bare
    # expression __half's operator* and friends match ambiguously).
    Param = Struct.new(:kind, :type, :shape, :storage) # kind: :buffer | :scalar | :var

    # A constant fixed at transpile time (an unroll loop variable or a host
    # numeric constant appearing in an expression). code is the C literal to
    # embed, type is :int / :float.
    Const = Struct.new(:code, :type)

    # An alias to a name inside the kernel (an argument / local variable) when
    # an unroll element is a symbol. name is the identifier to emit; assignable
    # says whether it can be assigned (only :var).
    Alias = Struct.new(:name, :type, :assignable)

    module_function

    def signature_for(args)
      args.map do |arg|
        case arg
        when Buffer
          shape = arg.matrix? ? arg.shape : nil
          # Half precision is :float as a type (expressions use float) but is
          # declared as GPU_HALF; resolve_index inserts the cast on reads.
          if %i[float16 bfloat16].include?(arg.dtype)
            storage = arg.dtype == :float16 ? "GPU_HALF" : "GPU_BFLOAT"
            Param.new(:buffer, :float, shape, storage)
          else
            Param.new(:buffer, arg.dtype == :int32 ? :int : :float, shape)
          end
        when Integer then Param.new(:scalar, :int)
        when Float   then Param.new(:scalar, :float)
        else
          raise LaunchError, "unsupported kernel argument: #{arg.class} (expected Tilework::Buffer, Float, or Integer)"
        end
      end
    end

    # Ruby block + signature → dialect kernel source
    class Transpiler
      def initialize(block, signature, name, domain: nil)
        @block = block
        @signature = signature
        @name = name
        @domain = domain # [x range, y range]; nil means an explicit launch, so no guard
        @binding = block.binding # closure binding used to resolve unroll / host constants
      end

      def transpile
        scope = ast_of(@block)
        locals, args_node, body = scope.children
        param_names = locals.first(args_node.children[0])
        unless param_names.length == @signature.length
          raise CompileError,
                "kernel '#{@name}' takes #{param_names.length} parameters but was called with #{@signature.length} arguments"
        end

        @env = param_names.zip(@signature).to_h
        # Names of buffers written in the body (targets of buf[i] = ...).
        # Unwritten buffers become device const* to benefit from the constant cache.
        @written = {}

        # The const decision needs @written, so translate the body before declaring params.
        lines = body.nil? ? [] : statements(body, 1)

        params = param_names.zip(@signature).map do |pname, param|
          if param.kind == :buffer
            # Unwritten buffers are const (decided by scanning the body, so a
            # mistake shows up as a compile error, not a silent wrong result).
            # No restrict: aliasing the same buffer could break it.
            qualifier = @written[pname] ? "device" : "device const"
            "#{qualifier} #{param.storage || param.type}* #{pname}"
          else
            "constant #{param.type}& #{pname}"
          end
        end
        <<~KERNEL
          GPU_KERNEL void #{@name}(GPU_ARGS,
                                   #{params.join(",\n                         ")}) {
          #{domain_guard}#{lines.join("\n")}
          }
        KERNEL
      end

      private

      # Keep threads outside the launch domain out of the body. A grid launches
      # whole blocks only, so extra edge threads always run. An early return
      # keeps existing lines' indentation unchanged (DSL kernels never use
      # GPU_BARRIER, so some threads returning early is safe).
      def domain_guard
        return "" if @domain.nil?

        x, y = @domain
        cond = ["GLOBAL_ID_X >= #{x}"]
        cond << "GLOBAL_ID_Y >= #{y}" if y > 1
        "  if (#{cond.join(' || ')}) return;\n"
      end

      def ast_of(block)
        RubyVM::AbstractSyntaxTree.of(block)
      rescue StandardError => e
        raise CompileError, "Tilework.kernel requires a block defined in a readable source file: #{e.message}"
      end

      def unsupported!(node, what = nil)
        raise CompileError,
              "unsupported Ruby syntax in Tilework.kernel at line #{node.first_lineno}: #{what || node.type}"
      end

      # --- Statements ---

      def statements(node, depth)
        list = node.type == :BLOCK ? node.children : [node]
        list.flat_map { |n| statement(n, depth) }
      end

      def statement(node, depth)
        pad = "  " * depth
        case node.type
        when :LASGN, :DASGN, :DASGN_CURR
          [pad + assign(node)]
        when :ATTRASGN
          [pad + index_assign(node)]
        when :IF, :UNLESS
          if_statement(node, depth)
        when :WHILE
          cond, = expr(node.children[0])
          [pad + "while (#{cond}) {", *statements(node.children[1], depth + 1), pad + "}"]
        when :ITER
          loop_statement(node, depth)
        else
          unsupported!(node)
        end
      end

      def assign(node)
        name, value_node = node.children
        code, type = expr(value_node)
        if (existing = @env[name])
          case existing
          when Const
            unsupported!(node, "cannot assign to unrolled constant '#{name}'")
          when Alias
            unsupported!(node, "cannot assign to read-only '#{name}' (:#{existing.name})") unless existing.assignable
            unsupported!(node, "reassigning #{type} value to #{existing.type} variable '#{name}'") if
              existing.type == :int && type == :float
            "#{existing.name} = #{code};"
          else
            unsupported!(node, "reassigning #{type} value to #{existing.type} variable '#{name}'") if
              existing.type == :int && type == :float
            "#{name} = #{code};"
          end
        else
          @env[name] = Param.new(:var, type)
          "#{type} #{name} = #{code};"
        end
      end

      def index_assign(node)
        recv, mid, args = node.children
        unsupported!(node, "#{mid} assignment") unless mid == :[]=
        parts = args.children.compact
        # ATTRASGN's arguments are [indices..., value] ([idx, value] for one
        # index, [r, c, value] for two).
        value_node = parts.pop
        target, elem = resolve_index(recv, parts, node)
        # After resolve_index, recv is a buffer parameter's LVAR/DVAR. Record its
        # name as written, removing it from const.
        @written[recv.children[0]] = true
        val_code, val_type = expr(value_node)
        val_code = "(#{elem})(#{val_code})" unless val_type == elem
        "#{target} = #{val_code};"
      end

      def if_statement(node, depth)
        pad = "  " * depth
        # UNLESS also has children [condition, body, else]; only the condition is negated
        cond_node, then_node, else_node = node.children
        cond, = expr(cond_node)
        cond = "!(#{cond})" if node.type == :UNLESS
        lines = [pad + "if (#{cond}) {", *statements(then_node, depth + 1)]
        lines += [pad + "} else {", *statements(else_node, depth + 1)] if else_node
        lines << (pad + "}")
      end

      def loop_statement(node, depth)
        pad = "  " * depth
        call, scope = node.children

        # unroll(host_array) do |a, b| ... end: not a loop at GPU run time —
        # the host array is iterated in Ruby at transpile time and the body duplicated.
        return unroll_statement(call, scope, depth) if call.type == :FCALL && call.children[0] == :unroll

        var = scope.children[0].first
        recv, mid = call.children
        recv = unwrap_block(recv)
        header =
          case mid
          when :times
            count, type = expr(recv)
            unsupported!(node, "non-integer .times receiver") unless type == :int
            "for (int #{var} = 0; #{var} < #{count}; #{var}++) {"
          when :each
            unsupported!(node, "each on #{recv.type}") unless %i[DOT2 DOT3].include?(recv.type)
            lo, lo_t = expr(recv.children[0])
            hi, hi_t = expr(recv.children[1])
            unsupported!(node, "non-integer range bounds") unless lo_t == :int && hi_t == :int
            cmp = recv.type == :DOT3 ? "<" : "<="
            "for (int #{var} = #{lo}; #{var} #{cmp} #{hi}; #{var}++) {"
          else
            unsupported!(node, "block call .#{mid} (only .times and range .each loop)")
          end

        outer = @env.key?(var) ? @env[var] : nil
        @env[var] = Param.new(:var, :int)
        body = scope.children[2] ? statements(scope.children[2], depth + 1) : []
        outer ? @env[var] = outer : @env.delete(var)
        [pad + header, *body, pad + "}"]
      end

      # Expands unroll(collection) do |a, b| ... end. collection is evaluated in
      # the host binding; for each element the block parameters are inserted
      # into @env as constants, the body is translated, and the outputs are
      # concatenated. The loop variable is a compile-time constant, so there is
      # no for/while — the body is simply duplicated.
      def unroll_statement(fcall, scope, depth)
        arg_nodes = fcall.children[1]&.children&.compact || []
        unsupported!(fcall, "unroll takes exactly one argument") unless arg_nodes.length == 1

        collection = host_eval(arg_nodes.first)
        unless collection.is_a?(Array)
          raise CompileError, "unroll expects a host Array, got #{collection.class} (at line #{fcall.first_lineno})"
        end

        locals, args_node, body = scope.children
        param_names = locals.first(args_node.children[0])

        collection.flat_map do |element|
          values = param_names.length == 1 ? [element] : Array(element)
          unless values.length == param_names.length
            raise CompileError,
                  "unroll block expects #{param_names.length} value(s) but element #{element.inspect} yields #{values.length}"
          end

          with_bindings(param_names.zip(values)) do
            body ? statements(body, depth) : []
          end
        end
      end

      # Temporarily inserts unroll loop variables into @env, evaluates the block, then restores.
      def with_bindings(pairs)
        saved = pairs.map { |name, _| [name, @env.fetch(name, :__absent)] }
        pairs.each { |name, value| @env[name] = bind_value(value) }
        yield
      ensure
        saved.each { |name, old| old == :__absent ? @env.delete(name) : (@env[name] = old) }
      end

      # Turns an unroll element's value into a binding for the loop variable.
      #   number  → Const (embedded as a C literal)
      #   symbol  → Alias (an alias to a kernel argument / local variable)
      def bind_value(value)
        case value
        when Integer then Const.new(value.to_s, :int)
        when Float   then Const.new("#{value}f", :float)
        when Symbol
          target = @env[value] or
            raise CompileError, "unroll symbol :#{value} does not name a kernel parameter or variable"
          case target
          when Param then Alias.new(value.to_s, target.type, target.kind == :var)
          else target # already a Const / Alias, so chain it as is
          end
        else
          raise CompileError, "unroll values must be Integer, Float, or Symbol, got #{value.class} (#{value.inspect})"
        end
      end

      # Evaluates an AST node as a host value (for unroll arguments and host
      # constants in expressions). Only constants, local variables and
      # numeric/array literals are allowed; no arbitrary code runs.
      def host_eval(node)
        case node.type
        when :INTEGER, :FLOAT, :LIT then node.children[0]
        when :STR                   then node.children[0]
        when :LVAR, :DVAR           then @binding.local_variable_get(node.children[0])
        when :CONST, :COLON2, :COLON3 then @binding.eval(const_name(node))
        when :LIST, :ARRAY          then node.children.compact.map { |c| host_eval(c) }
        else
          raise CompileError,
                "cannot evaluate host expression (#{node.type}) at line #{node.first_lineno}; " \
                "use a constant, local variable, or literal"
        end
      end

      def const_name(node)
        case node.type
        when :CONST  then node.children[0].to_s
        when :COLON2 then "#{node.children[0] ? const_name(node.children[0]) : ''}::#{node.children[1]}"
        when :COLON3 then "::#{node.children[0]}"
        else unsupported!(node, "constant reference")
        end
      end

      # Resolves a constant appearing in an expression (like WIN_LENGTH) on the
      # host and embeds it as a C literal if numeric. Non-numerics such as arrays
      # cannot be embedded in an expression.
      def host_const(node)
        value = host_eval(node)
        case value
        when Integer then [value.to_s, :int]
        when Float   then ["#{value}f", :float]
        else unsupported!(node, "constant #{const_name(node)} is #{value.class}, not a number")
        end
      end

      # --- Expressions → [C code, type (:int | :float)] ---

      def expr(node)
        case node.type
        when :LIT
          literal(node, node.children[0])
        when :INTEGER then [node.children[0].to_s, :int]
        when :FLOAT   then literal(node, node.children[0])
        when :LVAR, :DVAR
          variable(node)
        when :OPCALL
          opcall(node)
        when :AND, :OR
          # a && b && c puts every operand in one AND node rather than nesting,
          # so fold all children, not just children[0..1].
          op = node.type == :AND ? "&&" : "||"
          parts = node.children.compact.map { |child| expr(child).first }
          ["(#{parts.join(" #{op} ")})", :int]
        when :CALL
          call(node)
        when :VCALL
          builtin(node)
        when :CONST, :COLON2, :COLON3
          host_const(node)
        when :IF
          ternary(node)
        when :BLOCK
          expr(unwrap_block(node))
        else
          unsupported!(node)
        end
      end

      # Some Ruby versions parse grouping parentheses inside an expression
      # (e.g. 1.0 + (2.0 + 3.0), (lo...hi).each) as a single-element :BLOCK node
      # (normally for begin/end or multiple statements). In expression or
      # receiver context a :BLOCK is always this one-element parenthesis
      # pattern, so unwrap it. Anything else (not exactly one element, or not a
      # :BLOCK) returns node as is.
      def unwrap_block(node)
        return node unless node.type == :BLOCK

        children = node.children.compact
        unsupported!(node, "expected a single parenthesized expression") unless children.length == 1
        children.first
      end

      def literal(node, value)
        case value
        when Integer then [value.to_s, :int]
        when Float   then ["#{value}f", :float]
        else unsupported!(node, "#{value.class} literal")
        end
      end

      def variable(node)
        name = node.children[0]
        entry = @env[name]
        case entry
        when Const then [entry.code, entry.type] # a constant inserted by unroll
        when Alias then [entry.name, entry.type] # an alias bound by unroll to a kernel variable
        when Param
          unsupported!(node, "buffer '#{name}' must be indexed") if entry.kind == :buffer
          [name.to_s, entry.type]
        else
          unsupported!(node, "unknown variable '#{name}'")
        end
      end

      def opcall(node)
        recv, op, args = node.children
        if args.nil? # unary operator
          code, type = expr(recv)
          case op
          when :-@ then ["(-#{code})", type]
          when :!  then ["(!#{code})", :int]
          else unsupported!(node, "unary #{op}")
          end
        else
          l, lt = expr(recv)
          r, rt = expr(args.children[0])
          if COMPARE_OPS.include?(op)
            ["(#{l} #{op} #{r})", :int]
          elsif ARITH_OPS.include?(op)
            type = lt == :float || rt == :float ? :float : :int
            unsupported!(node, "% on float operands") if op == :% && type == :float
            ["(#{l} #{op} #{r})", type]
          else
            unsupported!(node, "operator #{op}")
          end
        end
      end

      def call(node)
        recv, mid, args = node.children
        case
        when mid == :[]
          code, elem, storage = resolve_index(recv, args.children.compact, node)
          # Half cannot sit in a bare expression (operators match ambiguously); widen to float.
          read = storage == "GPU_BFLOAT" ? "GPU_BFLOAT_TO_FLOAT(#{code})" : "((float)#{code})"
          [storage ? read : code, elem]
        when recv.type == :CONST && recv.children[0] == :Math
          math_call(node, mid, args)
        when mid == :abs && args.nil?
          code, type = expr(recv)
          [type == :float ? "fabs(#{code})" : "abs(#{code})", type]
        when (mid == :rows || mid == :cols) && args.nil? && %i[LVAR DVAR].include?(recv.type)
          shape = buffer_shape(recv, node, "call .#{mid}")
          [(mid == :rows ? shape[0] : shape[1]).to_s, :int]
        else
          unsupported!(node, "method call .#{mid}")
        end
      end

      def math_call(node, mid, args)
        arity = MATH_FNS[mid] or unsupported!(node, "Math.#{mid}")
        actual = args ? args.children.compact : []
        unsupported!(node, "Math.#{mid} takes #{arity} arguments") unless actual.length == arity
        converted = actual.map { |a| "(float)(#{expr(a).first})" }
        ["#{mid}(#{converted.join(', ')})", :float]
      end

      def builtin(node)
        name = node.children[0]
        macro = BUILTIN_IDS[name] or unsupported!(node, "unknown identifier '#{name}'")
        [macro, :int]
      end

      def ternary(node)
        cond_node, then_node, else_node = node.children
        unsupported!(node, "if expression without else") if then_node.nil? || else_node.nil?
        cond, = expr(cond_node)
        t, tt = expr(then_node)
        e, et = expr(else_node)
        ["(#{cond} ? #{t} : #{e})", tt == :float || et == :float ? :float : :int]
      end

      # Handles buf[i] (1D) and buf[r, c] (2D, expanded to a row-major flat
      # index) alike. Returns [C index expression (e.g. "buf[i]"), element type,
      # storage type]. **No cast is inserted** — it is also used as an assignment
      # target; the read side (#call) widens to float based on the storage type.
      def resolve_index(recv, idx_nodes, ctx)
        case idx_nodes.length
        when 1
          target, elem, storage = buffer_ref(recv, ctx)
          idx, idx_type = expr(idx_nodes[0])
          unsupported!(ctx, "non-integer buffer index") unless idx_type == :int
          ["#{target}[#{idx}]", elem, storage]
        when 2
          target, elem, cols, storage = buffer_ref_2d(recv, ctx)
          r, r_type = expr(idx_nodes[0])
          c, c_type = expr(idx_nodes[1])
          unsupported!(ctx, "non-integer buffer index") unless r_type == :int && c_type == :int
          ["#{target}[((#{r}) * #{cols}) + (#{c})]", elem, storage]
        else
          unsupported!(ctx, "buffer indexing takes 1 or 2 indices, got #{idx_nodes.length}")
        end
      end

      def buffer_ref(node, ctx)
        unsupported!(ctx, "indexing a non-variable") unless %i[LVAR DVAR].include?(node.type)
        name = node.children[0]
        param = @env[name]
        unsupported!(ctx, "'#{name}' is not a buffer parameter") unless param.is_a?(Param) && param.kind == :buffer
        [name.to_s, param.type, param.storage]
      end

      # For buf[r, c]; only arguments called with a 2D-shaped Buffer.
      def buffer_ref_2d(node, ctx)
        name, type, storage = buffer_ref(node, ctx)
        [name, type, buffer_shape(node, ctx, "index it as buf[r, c]")[1], storage]
      end

      # For buf.rows / buf.cols; only arguments called with a 2D-shaped Buffer.
      def buffer_shape(node, ctx, purpose)
        name = node.children[0]
        param = @env[name]
        unless param&.shape
          raise CompileError,
                "'#{name}' is not a 2D buffer for this call (pass a Buffer with a 2D shape, " \
                "e.g. Buffer.from_matrix, to #{purpose})"
        end
        param.shape
      end
    end
  end

  # A callable kernel generated from a Ruby DSL block.
  #
  # A specialized compiled kernel is cached for each argument signature.
  class DSLKernel
    attr_reader :name

    # Creates a callable DSL kernel.
    #
    # @param name [String] kernel name
    # @param block [Proc] supported Tilework DSL body
    # @param out [Integer, nil] output parameter index to allocate automatically
    # @param out_dtype [Symbol, nil] dtype for an automatically allocated output
    # @return [DSLKernel]
    #
    # out:       the output buffer's position (0-based index of the block
    #            parameter). When given, call/.() allocates the output buffer and
    #            returns it (the output is omitted from the call arguments).
    #            Otherwise every argument is passed.
    # out_dtype: the type of the auto-allocated output (:float32 / :int32).
    #            Inferred from the input Buffers when omitted (int32 if all are
    #            int32, float32 otherwise). For kernels whose output type differs
    #            from the inputs.
    def initialize(name, block, out: nil, out_dtype: nil)
      raise ArgumentError, "Tilework.kernel requires a block" if block.nil?

      @name = name
      @block = block
      @out = out
      @out_dtype = out_dtype
      @compiled = {}
    end

    # Launches the kernel over an inferred or explicit domain.
    #
    # @param args [Array<Buffer, Numeric>] kernel inputs
    # @param over [Integer, Array<Integer>, nil] explicit one- or two-dimensional domain
    # @return [Buffer, nil] an allocated output when `out:` was configured
    #
    # Automatic grid/block launch. The domain is, by default:
    #   - over: when given (an Integer for 1D, [cols, rows] for 2D)
    #   - otherwise the shape of a 2D-shaped Buffer argument, as a 2D domain
    #   - otherwise the length of the longest Buffer argument, as a 1D domain
    # A 2D domain launches with global_id_x = column and global_id_y = row.
    #
    # The generator inserts the bounds guard automatically. A grid launches
    # whole blocks only (Metal's dispatchThreadgroups and CUDA's cuLaunchKernel
    # alike), so extra edge threads always run, but they never execute the body
    # with an index outside the domain. This structurally prevents out-of-range
    # writes from a forgotten guard — harmless padding on Metal, a neighbor
    # buffer corrupted on CUDA.
    def call(*args, over: nil)
      domain = resolve_domain(args, over)

      if @out
        unless (0..args.length).cover?(@out)
          raise LaunchError, "kernel '#{@name}' out: #{@out} is out of range for #{args.length} call arguments"
        end

        dtype = @out_dtype || (args.grep(Buffer).all? { |b| b.dtype == :int32 } ? :int32 : :float32)
        out_buf = domain.new_buffer(dtype)
        full_args = args.dup.insert(@out, out_buf)
        kernel_for(full_args, block: domain.block, domain: domain)
          .launch(grid: domain.grid, block: domain.block, args: full_args)
        out_buf
      else
        kernel_for(args, block: domain.block, domain: domain)
          .launch(grid: domain.grid, block: domain.block, args: args)
      end
    end

    # Launches with explicitly supplied dimensions and waits for completion.
    #
    # @param grid [Integer, Array<Integer>]
    # @param block [Integer, Array<Integer>]
    # @param args [Array<Buffer, Numeric>]
    # @return [nil]
    def launch(grid:, block:, args: [])
      kernel_for(args, block: block).launch(grid: grid, block: block, args: args)
    end

    # Launches with explicitly supplied dimensions without waiting.
    #
    # @param grid [Integer, Array<Integer>]
    # @param block [Integer, Array<Integer>]
    # @param args [Array<Buffer, Numeric>]
    # @return [nil]
    def launch_async(grid:, block:, args: [])
      kernel_for(args, block: block).launch_async(grid: grid, block: block, args: args)
    end

    # Returns generated dialect source for inspection.
    #
    # @param args [Array<Buffer, Numeric>] representative argument values
    # @param block [Integer, Array<Integer>, nil] block dimensions for source generation
    # @param over [Integer, Array<Integer>, nil] domain used for generated bounds guards
    # @return [String]
    #
    # With over: the source includes the same bounds guard as a real launch.
    # Otherwise it tries the same rules as .(), and when mixed shapes make that
    # impossible it returns guard-free source (for debugging that only wants the
    # signature).
    def source_for(args, block: nil, over: nil)
      domain = begin
        resolve_domain(args, over)
      rescue LaunchError
        nil
      end
      generate_source(args, block, domain)
    end

    private

    # A value object holding the launch domain (1D or 2D) and the grid / block /
    # output allocation derived from it.
    Domain1D = Struct.new(:n) do
      def grid = (n / DSL::BLOCK.to_f).ceil
      def block = DSL::BLOCK
      def new_buffer(dtype) = Buffer.new(dtype, n)
      # [x range, y range] for the automatic bounds guard
      def extents = [n, 1]
    end

    Domain2D = Struct.new(:cols, :rows) do
      TILE = 16 # block: [16, 16] = 256 threads (matching the 1D DSL::BLOCK)

      def grid = [(cols / TILE.to_f).ceil, (rows / TILE.to_f).ceil]
      def block = [TILE, TILE]
      def new_buffer(dtype) = Buffer.new(dtype, cols * rows, shape: [rows, cols])
      def extents = [cols, rows]
    end

    def resolve_domain(args, over)
      case over
      when Array
        unless over.length == 2 && over.all? { |d| d.is_a?(Integer) && d.positive? }
          raise LaunchError, "over: must be a positive Integer or [cols, rows], got #{over.inspect}"
        end
        Domain2D.new(over[0], over[1])
      when Integer
        Domain1D.new(over)
      when nil
        matrices = args.grep(Buffer).select(&:matrix?)
        shapes = matrices.map(&:shape).uniq
        if shapes.length > 1
          raise LaunchError, "kernel '#{@name}' got 2D buffer arguments with different shapes: #{shapes.inspect}; " \
                              "pass over: explicitly to disambiguate"
        elsif shapes.length == 1
          Domain2D.new(matrices.first.cols, matrices.first.rows)
        else
          n = args.grep(Buffer).map(&:length).max or
            raise LaunchError, "kernel '#{@name}' needs a Buffer argument or over: to infer launch size"
          Domain1D.new(n)
        end
      else
        raise LaunchError, "over: must be a positive Integer or [cols, rows], got #{over.inspect}"
      end
    end

    # Dialect backends' generated source does not depend on block, so the
    # signature alone is the key. The domain is baked into the source as the
    # bounds guard's literals, so like the shape (signature) it is part of the
    # cache key. In practice over: almost always follows from argument shapes,
    # so caching is just as effective.
    def kernel_for(args, block:, domain: nil)
      signature = DSL.signature_for(args)
      extents = domain&.extents
      # DSLKernel instances (like NN.head_slice's @head_gather) may be reused
      # across backends, so backend is part of the key to avoid returning a
      # pipeline built for another backend.
      backend = Tilework.backend
      # Backends that cannot use the dialect (:cpu) do not bake grid/block into
      # the source, so pass raw: true.
      raw = !Tilework.capabilities[:dialect]
      @compiled[[backend, signature, extents]] ||=
        Tilework.compile(generate_source(args, block, domain), name: @name, raw: raw)
    end

    def generate_source(args, block, domain = nil)
      extents = domain&.extents
      return DSL::Transpiler.new(@block, DSL.signature_for(args), @name, domain: extents).transpile if Tilework.capabilities[:dialect]

      unless Tilework.bridge.respond_to?(:dsl_codegen)
        raise Error, "the #{Tilework.backend.inspect} backend doesn't support dialect kernel strings " \
                     "and doesn't implement dsl_codegen either"
      end
      Tilework.bridge.dsl_codegen.transpile(@block, args, name: @name, domain: extents)
    end
  end
end
