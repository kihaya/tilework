# frozen_string_literal: true

require_relative "../../dsl"

module Tilework
  module CPU
    # The :cpu backend for the Ruby kernel DSL. Runs the same
    # Tilework.kernel {} calls without a GPU so that correctness can always be
    # checked against values computed on the CPU.
    #
    # It reuses Tilework::DSL::Transpiler's Ruby subset and type
    # specialization (Tilework::DSL.signature_for / Param / Const / Alias are
    # host-side transpile-time machinery, independent of the target language)
    # and emits Ruby source instead of the CUDA-flavored dialect. dsl.rb is
    # left untouched.
    #
    # The generated source is the Hash literal
    # `{ arg_kinds: [...], proc: lambda { |...| ... } }`, which
    # Tilework::CPUBridge#pipeline_create evals once and uses as the pipeline
    # handle (see lib/tilework/backend/cpu.rb). The proc's calling convention is
    # fixed: `proc.call(bufs, scalars, gx, gy, gz, lx, ly, lz, bx, by, bz)`.
    # bufs/scalars are arrays indexed by argument position (signature order);
    # buffer positions have a value in bufs, scalar positions in scalars (only
    # one of the two is used per position).
    #
    # Ruby's native numeric semantics are used as is (a design choice: C int32
    # wraparound and negative-division rounding are not reproduced). Explicit
    # casts (f32(x)/i32(x)) are therefore unnecessary and arithmetic relies on
    # Ruby's implicit promotion. Only comparison results need care: Ruby's
    # true/false must be told apart from 0/1 (Integer) used as values (true/false
    # cannot go to Array#pack("l*")), so the "value context" (expr) and the
    # "condition context" (bool_expr) are separate.
    class Transpiler
      BUILTIN_IDS = {
        global_id_x: ["gx", :int],
        global_id_y: ["gy", :int],
        local_id_x: ["lx", :int],
        group_id_x: ["bx", :int],
      }.freeze

      def initialize(block, signature, name, domain: nil)
        @block = block
        @signature = signature
        @name = name
        @domain = domain # [x range, y range]; nil means no guard
        @binding = block.binding
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
        preamble = param_names.zip(@signature).each_with_index.map do |(pname, param), idx|
          param.kind == :buffer ? "#{pname} = bufs[#{idx}]" : "#{pname} = scalars[#{idx}]"
        end

        arg_kinds = @signature.map { |param| arg_kind(param) }
        lines = body.nil? ? [] : statements(body, 1)
        <<~RUBY
          {
            arg_kinds: #{arg_kinds.inspect},
            proc: lambda do |bufs, scalars, gx, gy, gz, lx, ly, lz, bx, by, bz|
              #{preamble.join("\n    ")}
          #{domain_guard}#{lines.join("\n")}
            end,
          }
        RUBY
      end

      private

      # Same role as dsl.rb's Transpiler#domain_guard (see the comment there).
      # The CPU backend does not run the extra edge threads a GPU does, but
      # inserts the same guard to keep the semantics aligned.
      def domain_guard
        return "" if @domain.nil?

        x, y = @domain
        cond = ["gx >= #{x}"]
        cond << "gy >= #{y}" if y > 1
        "    return if #{cond.join(' || ')}\n"
      end

      def arg_kind(param)
        return :"#{param.type == :int ? 'int32' : 'float32'}_scalar" unless param.kind == :buffer

        return :bfloat16_buffer if param.storage == "GPU_BFLOAT"

        :"#{param.type == :int ? 'int32' : 'float32'}_buffer"
      end

      def ast_of(block)
        RubyVM::AbstractSyntaxTree.of(block)
      rescue StandardError => e
        raise CompileError, "Tilework::CPU::Transpiler requires a block defined in a readable source file: #{e.message}"
      end

      def unsupported!(node, what = nil)
        raise CompileError,
              "unsupported Ruby syntax in CPU kernel at line #{node.first_lineno}: #{what || node.type}"
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
          cond = bool_expr(node.children[0])
          [pad + "while #{cond}", *statements(node.children[1], depth + 1), pad + "end"]
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
          when DSL::Const
            unsupported!(node, "cannot assign to unrolled constant '#{name}'")
          when DSL::Alias
            unsupported!(node, "cannot assign to read-only '#{name}' (:#{existing.name})") unless existing.assignable
            unsupported!(node, "reassigning #{type} value to #{existing.type} variable '#{name}'") if
              existing.type == :int && type == :float
            "#{existing.name} = #{code}"
          else
            unsupported!(node, "reassigning #{type} value to #{existing.type} variable '#{name}'") if
              existing.type == :int && type == :float
            "#{name} = #{code}"
          end
        else
          @env[name] = DSL::Param.new(:var, type)
          "#{name} = #{code}"
        end
      end

      def index_assign(node)
        recv, mid, args = node.children
        unsupported!(node, "#{mid} assignment") unless mid == :[]=
        parts = args.children.compact
        value_node = parts.pop
        target, elem = resolve_index(recv, parts, node)
        val_code, val_type = expr(value_node)
        val_code = "(#{val_code}).#{elem == :int ? 'to_i' : 'to_f'}" unless val_type == elem
        "#{target} = #{val_code}"
      end

      def if_statement(node, depth)
        pad = "  " * depth
        cond_node, then_node, else_node = node.children
        cond = bool_expr(cond_node)
        cond = "!(#{cond})" if node.type == :UNLESS
        lines = [pad + "if #{cond}", *statements(then_node, depth + 1)]
        lines += [pad + "else", *statements(else_node, depth + 1)] if else_node
        lines << (pad + "end")
      end

      def loop_statement(node, depth)
        pad = "  " * depth
        call, scope = node.children

        return unroll_statement(call, scope, depth) if call.type == :FCALL && call.children[0] == :unroll

        var = scope.children[0].first
        recv, mid = call.children
        recv = unwrap_block(recv)
        header =
          case mid
          when :times
            count, type = expr(recv)
            unsupported!(node, "non-integer .times receiver") unless type == :int
            "(#{count}).times do |#{var}|"
          when :each
            unsupported!(node, "each on #{recv.type}") unless %i[DOT2 DOT3].include?(recv.type)
            lo, lo_t = expr(recv.children[0])
            hi, hi_t = expr(recv.children[1])
            unsupported!(node, "non-integer range bounds") unless lo_t == :int && hi_t == :int
            op = recv.type == :DOT3 ? "..." : ".."
            "(#{lo}#{op}#{hi}).each do |#{var}|"
          else
            unsupported!(node, "block call .#{mid} (only .times and range .each loop)")
          end

        outer = @env.key?(var) ? @env[var] : nil
        @env[var] = DSL::Param.new(:var, :int)
        body = scope.children[2] ? statements(scope.children[2], depth + 1) : []
        outer ? @env[var] = outer : @env.delete(var)
        [pad + header, *body, pad + "end"]
      end

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

      def with_bindings(pairs)
        saved = pairs.map { |name, _| [name, @env.fetch(name, :__absent)] }
        pairs.each { |name, value| @env[name] = bind_value(value) }
        yield
      ensure
        saved.each { |name, old| old == :__absent ? @env.delete(name) : (@env[name] = old) }
      end

      def bind_value(value)
        case value
        when Integer then DSL::Const.new(value.to_s, :int)
        when Float   then DSL::Const.new(value.to_s, :float)
        when Symbol
          target = @env[value] or
            raise CompileError, "unroll symbol :#{value} does not name a kernel parameter or variable"
          case target
          when DSL::Param then DSL::Alias.new(value.to_s, target.type, target.kind == :var)
          else target
          end
        else
          raise CompileError, "unroll values must be Integer, Float, or Symbol, got #{value.class} (#{value.inspect})"
        end
      end

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

      def host_const(node)
        value = host_eval(node)
        case value
        when Integer then [value.to_s, :int]
        when Float   then [value.to_s, :float]
        else unsupported!(node, "constant #{const_name(node)} is #{value.class}, not a number")
        end
      end

      # --- Expressions (value context) → [Ruby code, type (:int | :float)] ---

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
          ["(#{bool_expr(node)} ? 1 : 0)", :int]
        when :CALL
          call(node)
        when :VCALL
          builtin(node)
        when :CONST, :COLON2, :COLON3
          host_const(node)
        when :IF
          ternary(node)
        when :BLOCK
          unwrap_parenthesized(node) { |inner| expr(inner) }
        else
          unsupported!(node)
        end
      end

      def unwrap_parenthesized(node)
        yield unwrap_block(node)
      end

      # For places that want the AST node itself rather than translated code,
      # such as loop_statement's receiver (n.times / (lo...hi).each). Returns
      # node as is unless it is a :BLOCK.
      def unwrap_block(node)
        return node unless node.type == :BLOCK

        children = node.children.compact
        unsupported!(node, "expected a single parenthesized expression") unless children.length == 1
        children.first
      end

      # --- Expressions (condition context) → code returning a Ruby boolean ---
      #
      # Conditions of if/while/unless/ternaries, && and || go through here.
      # Only comparisons, &&, || and unary ! are allowed; using an int/float
      # value directly as a condition is rejected (0 is truthy in Ruby, which
      # would disagree with the C-family backends).

      def bool_expr(node)
        case node.type
        when :OPCALL
          recv, op, args = node.children
          if args.nil? && op == :!
            "!(#{bool_expr(recv)})"
          elsif args && DSL::COMPARE_OPS.include?(op)
            l, = expr(recv)
            r, = expr(args.children[0])
            "(#{l} #{op} #{r})"
          else
            unsupported!(node, "expected a boolean expression (comparison, &&/||, or !)")
          end
        when :AND, :OR
          op = node.type == :AND ? "&&" : "||"
          parts = node.children.compact.map { |child| bool_expr(child) }
          "(#{parts.join(" #{op} ")})"
        when :BLOCK
          unwrap_parenthesized(node) { |inner| bool_expr(inner) }
        else
          unsupported!(node, "expected a boolean expression (comparison, &&/||, or !)")
        end
      end

      def literal(node, value)
        case value
        when Integer then [value.to_s, :int]
        when Float   then [value.to_s, :float]
        else unsupported!(node, "#{value.class} literal")
        end
      end

      def variable(node)
        name = node.children[0]
        entry = @env[name]
        case entry
        when DSL::Const then [entry.code, entry.type]
        when DSL::Alias then [entry.name, entry.type]
        when DSL::Param
          unsupported!(node, "buffer '#{name}' must be indexed") if entry.kind == :buffer
          [name.to_s, entry.type]
        else
          unsupported!(node, "unknown variable '#{name}'")
        end
      end

      def opcall(node)
        recv, op, args = node.children
        if args.nil?
          case op
          when :-@
            code, type = expr(recv)
            ["(-#{code})", type]
          when :!
            ["(#{bool_expr(node)} ? 1 : 0)", :int]
          else
            unsupported!(node, "unary #{op}")
          end
        elsif DSL::COMPARE_OPS.include?(op)
          ["(#{bool_expr(node)} ? 1 : 0)", :int]
        elsif DSL::ARITH_OPS.include?(op)
          l, lt = expr(recv)
          r, rt = expr(args.children[0])
          unsupported!(node, "% on float operands") if op == :% && (lt == :float || rt == :float)
          type = lt == :float || rt == :float ? :float : :int
          ["(#{l} #{op} #{r})", type]
        else
          unsupported!(node, "operator #{op}")
        end
      end

      def call(node)
        recv, mid, args = node.children
        case
        when mid == :[]
          resolve_index(recv, args.children.compact, node)
        when recv.type == :CONST && recv.children[0] == :Math
          math_call(node, mid, args)
        when mid == :abs && args.nil?
          code, type = expr(recv)
          ["(#{code}).abs", type]
        when (mid == :rows || mid == :cols) && args.nil? && %i[LVAR DVAR].include?(recv.type)
          shape = buffer_shape(recv, node, "call .#{mid}")
          [(mid == :rows ? shape[0] : shape[1]).to_s, :int]
        else
          unsupported!(node, "method call .#{mid}")
        end
      end

      def math_call(node, mid, args)
        arity = DSL::MATH_FNS[mid] or unsupported!(node, "Math.#{mid}")
        actual = args ? args.children.compact : []
        unsupported!(node, "Math.#{mid} takes #{arity} arguments") unless actual.length == arity
        converted = actual.map { |a| "(#{expr(a).first}).to_f" }
        code = mid == :pow ? "(#{converted[0]} ** #{converted[1]})" : "Math.#{mid}(#{converted.join(', ')})"
        [code, :float]
      end

      def builtin(node)
        name = node.children[0]
        macro = BUILTIN_IDS[name] or unsupported!(node, "unknown identifier '#{name}'")
        macro
      end

      def ternary(node)
        cond_node, then_node, else_node = node.children
        unsupported!(node, "if expression without else") if then_node.nil? || else_node.nil?
        cond = bool_expr(cond_node)
        t, tt = expr(then_node)
        e, et = expr(else_node)
        ["(#{cond} ? #{t} : #{e})", tt == :float || et == :float ? :float : :int]
      end

      def resolve_index(recv, idx_nodes, ctx)
        case idx_nodes.length
        when 1
          target, elem = buffer_ref(recv, ctx)
          idx, idx_type = expr(idx_nodes[0])
          unsupported!(ctx, "non-integer buffer index") unless idx_type == :int
          ["#{target}[#{idx}]", elem]
        when 2
          target, elem, cols = buffer_ref_2d(recv, ctx)
          r, r_type = expr(idx_nodes[0])
          c, c_type = expr(idx_nodes[1])
          unsupported!(ctx, "non-integer buffer index") unless r_type == :int && c_type == :int
          ["#{target}[((#{r}) * #{cols}) + (#{c})]", elem]
        else
          unsupported!(ctx, "buffer indexing takes 1 or 2 indices, got #{idx_nodes.length}")
        end
      end

      def buffer_ref(node, ctx)
        unsupported!(ctx, "indexing a non-variable") unless %i[LVAR DVAR].include?(node.type)
        name = node.children[0]
        param = @env[name]
        unsupported!(ctx, "'#{name}' is not a buffer parameter") unless param.is_a?(DSL::Param) && param.kind == :buffer
        [name.to_s, param.type]
      end

      def buffer_ref_2d(node, ctx)
        name, type = buffer_ref(node, ctx)
        [name, type, buffer_shape(node, ctx, "index it as buf[r, c]")[1]]
      end

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

    module_function

    # Generates the Ruby source for CPU execution from the given block and
    # arguments (eval'ing it yields the { arg_kinds:, proc: } Hash).
    def transpile(block, args, name: "dsl_kernel", domain: nil)
      signature = DSL.signature_for(args)
      Transpiler.new(block, signature, name, domain: domain).transpile
    end
  end
end
