# frozen_string_literal: true

module Tilework
  # Elementwise and reduction operations mixed into {Buffer}.
  # Generates and launches CUDA-flavored dialect kernels. The generated source
  # rides Kernel's PSO cache, so repeating an operation does not recompile.
  module Ops
    # float16 / bfloat16 appear **only as storage types**. Arithmetic always
    # widens to float, so they never become a compute dtype (see result_dtypes).
    CTYPE = { float32: "float", int32: "int", float16: "GPU_HALF", bfloat16: "GPU_BFLOAT" }.freeze
    LOW_PRECISION_DTYPES = %i[float16 bfloat16].freeze
    ZERO = { float32: "0.0f", int32: "0" }.freeze
    BLOCK = 256
    OPERATORS = %i[+ - * /].freeze
    COMPARISONS = %i[> < >= <=].freeze
    # Unary math functions. int32 input is promoted to float32
    UNARY_MATH = %i[exp log sqrt sin cos tanh floor ceil round].freeze

    # Reduction combine expressions and identities. The identity is the value
    # padding elements take so they do not affect the result
    REDUCTIONS = {
      sum: {
        combine: "sh[lid] += sh[lid + s];",
        identity: ZERO,
      },
      min: {
        combine: "sh[lid] = min(sh[lid], sh[lid + s]);",
        identity: { float32: "INFINITY", int32: "2147483647" },
      },
      max: {
        combine: "sh[lid] = max(sh[lid], sh[lid + s]);",
        identity: { float32: "-INFINITY", int32: "(-2147483647 - 1)" },
      },
    }.freeze

    OPERATORS.each do |op|
      define_method(op) { |other| Ops.arithmetic(op, self, other) }
    end

    COMPARISONS.each do |op|
      define_method(op) { |other| Ops.compare(op, self, other) }
    end

    UNARY_MATH.each do |fn|
      define_method(fn) { Ops.unary_math(fn, self) }
    end

    # Compares elements for equality and returns an int32 mask.
    # @param other [Buffer, Numeric]
    # @return [Buffer]
    def eq(other) = Ops.compare(:==, self, other)
    def ne(other) = Ops.compare(:!=, self, other)

    # Returns elementwise absolute values.
    # @return [Buffer]
    def abs = Ops.unary_abs(self)
    def clamp(lo, hi) = Ops.clamp_op(self, lo, hi)
    def maximum(other) = Ops.binary_fn("max", self, other)
    def minimum(other) = Ops.binary_fn("min", self, other)

    # Multiplies two two-dimensional matrices.
    # @param other [Buffer]
    # @param out_dtype [Symbol, nil] optional low-precision output storage dtype
    # @return [Buffer] `[rows, other.cols]` matrix; float inputs accumulate in float32
    def matmul(other, out_dtype: nil) = Ops.matmul_op(self, other, out_dtype: out_dtype)

    # Makes scalar-on-the-left expressions like 2.0 * buffer work
    def coerce(num)
      [Scalar.new(num), self]
    end

    # Reduces all elements to a sum.
    # @param device [Boolean] return a one-element buffer instead of a host scalar
    # @return [Numeric, Buffer]
    #
    # Returns a host Ruby number by default. **device: true returns a
    # one-element Buffer holding the result** (see Ops.reduce).
    def sum(device: false) = Ops.reduce(:sum, self, device: device)
    def min(device: false) = Ops.reduce(:min, self, device: device)
    def max(device: false) = Ops.reduce(:max, self, device: device)
    def mean(device: false) = Ops.mean_op(self, device: device)
    def argmax(device: false) = Ops.arg_reduce(:max, self, device: device)
    def argmin(device: false) = Ops.arg_reduce(:min, self, device: device)

    # The scalar-side proxy returned by coerce. Non-commutative ops (- /)
    # generate a kernel with swapped operands; comparisons flip the operator
    # (s > a[i] ≡ a[i] < s).
    class Scalar
      def initialize(value)
        @value = value
      end

      def +(other) = Ops.arithmetic(:+, other, @value)
      def *(other) = Ops.arithmetic(:*, other, @value)
      def -(other) = Ops.arithmetic(:-, other, @value, reversed: true)
      def /(other) = Ops.arithmetic(:/, other, @value, reversed: true)
      def >(other) = Ops.compare(:<, other, @value)
      def <(other) = Ops.compare(:>, other, @value)
      def >=(other) = Ops.compare(:<=, other, @value)
      def <=(other) = Ops.compare(:>=, other, @value)
    end

    # The execution context an upper layer (the graph's Plan) plugs in while
    # running. **Without one (= eager use) behavior is unchanged.**
    #
    #   #allocate(dtype, length, shape) — takes over output allocation; a
    #                                     caller that knows lifetimes can hand
    #                                     out pooled buffers instead
    #   #async?                         — true makes launches launch_async
    #
    # It is a thread-local context rather than an argument to keep the branch
    # out of eager code (threading out:/async: through every op signature would
    # make everyone carry arguments no caller uses).
    CONTEXT_KEY = :tilework_ops_context

    class << self
      def context = Thread.current[CONTEXT_KEY]

      def with_context(context)
        previous = Thread.current[CONTEXT_KEY]
        Thread.current[CONTEXT_KEY] = context
        yield
      ensure
        Thread.current[CONTEXT_KEY] = previous
      end
    end

    module_function

    def allocate_output(dtype, length, shape)
      context = Ops.context
      return Buffer.new(dtype, length, shape: shape) unless context

      context.allocate(dtype, length, shape)
    end

    # Called from `Buffer#free`: the third part of the context contract (after
    # where to allocate and whether to go async — who decides freeing).
    # Releasing memory a running kernel reads is undefined behavior on
    # CUDA/HIP, so with a context the free is handed to it. With no context
    # (eager) it returns false and nothing changes.
    def defer_free(buffer)
      context = Ops.context
      return false unless context.respond_to?(:defer_free)

      # The context decides whether it took the buffer (things with no point in
      # deferring, like views, return false and are freed on the spot).
      !!context.defer_free(buffer)
    end

    def launch_kernel(kernel, grid:, block:, args:)
      return kernel.launch(grid: grid, block: block, args: args) unless Ops.context&.async?

      kernel.launch_async(grid: grid, block: block, args: args)
    end

    # The source generator for backends that cannot use the dialect
    # (capabilities[:dialect] == false). The bridge declares it itself (the same
    # pattern as Tilework.host_fallback?), so no backend name is written here.
    def codegen
      Tilework.bridge.ops_codegen
    end

    def require_computable!(*buffers)
      caps = Tilework.capabilities
      bad = buffers.compact.select do |buffer|
        next false unless buffer.is_a?(Buffer)

        (buffer.dtype == :float16 && !caps.fp16_storage) ||
          (buffer.dtype == :bfloat16 && !caps.bf16_storage)
      end
      return if bad.empty?

      raise Error, "#{bad.map(&:dtype).uniq.join(' / ')} is not computable on the #{Tilework.backend.inspect} backend; " \
                   "use #cast(:float32) to compute with them"
    end

    # The compute dtype. **float16 is never a compute dtype** — accumulating in
    # half precision piles up error, so it always widens to float.
    def promoted_dtype(buffer, operand)
      other_int = operand.is_a?(Buffer) ? operand.dtype == :int32 : operand.is_a?(Integer)
      buffer.dtype == :int32 && other_int ? :int32 : :float32
    end

    # Returns [compute dtype, output dtype]. half with half **keeps a float16
    # output**: half precision exists to save memory bandwidth, and widening to
    # float32 here would defeat it. Rounding happens once on write-back, so the
    # precision is the same as "compute in float, round at the end".
    # Half is a storage type, so it is widened to float before entering an expression.
    def read(buffer, expr)
      buffer.is_a?(Buffer) ? read_dtype(buffer.dtype, expr) : expr
    end

    def read_dtype(dtype, expr)
      case dtype
      when :float16  then "(float)#{expr}"
      when :bfloat16 then "GPU_BFLOAT_TO_FLOAT(#{expr})"
      else expr
      end
    end

    def result_dtypes(buffer, operand)
      compute = promoted_dtype(buffer, operand)
      same_storage = operand.is_a?(Buffer) ? operand.dtype == buffer.dtype : operand.is_a?(Numeric)
      out = LOW_PRECISION_DTYPES.include?(buffer.dtype) && same_storage ? buffer.dtype : compute
      [compute, out]
    end

    # cpu_expr is passed as a thunk (a zero-argument lambda) so codegen
    # (= Tilework.bridge.ops_codegen) need not be called. When compiled takes the
    # dialect branch (always, on dialect backends) it is never called, so
    # bridges without ops_codegen (MockBridge, Metal, CUDA) do not fail here.
    def arithmetic(op, buffer, operand, reversed: false)
      compute, out = result_dtypes(buffer, operand)
      binary(buffer, operand, compute_dtype: compute, out_dtype: out, reversed: reversed,
             cpu_expr: -> { codegen.arithmetic_expr(op) }) do |lhs, rhs|
        "#{lhs} #{op} #{rhs}"
      end
    end

    # Comparisons evaluate in the promoted type and return an int32 0/1 mask
    def compare(op, buffer, operand)
      binary(buffer, operand, compute_dtype: promoted_dtype(buffer, operand), out_dtype: :int32,
             cpu_expr: -> { codegen.compare_expr(op) }) do |lhs, rhs|
        "#{lhs} #{op} #{rhs} ? 1 : 0"
      end
    end

    def binary_fn(fn, buffer, operand)
      compute, out = result_dtypes(buffer, operand)
      binary(buffer, operand, compute_dtype: compute, out_dtype: out,
             cpu_expr: -> { codegen.binary_fn_expr(fn) }) do |lhs, rhs|
        "#{fn}(#{lhs}, #{rhs})"
      end
    end

    def binary(buffer, operand, compute_dtype:, out_dtype:, cpu_expr:, reversed: false, &expr)
      require_computable!(buffer, operand)
      case operand
      when Buffer  then buffer_binary(buffer, operand, compute_dtype, out_dtype, cpu_expr, &expr)
      when Numeric then scalar_binary(buffer, operand, compute_dtype, out_dtype, reversed, cpu_expr, &expr)
      else
        raise TypeError, "cannot combine Tilework::Buffer with #{operand.class}"
      end
    end

    def buffer_binary(a, b, compute_dtype, out_dtype, cpu_expr, &expr)
      # If either side has a 2D-or-higher shape, require equal shapes (to catch
      # mix-ups); two 1D sides compare by length only.
      if a.shape.length > 1 || b.shape.length > 1
        raise ArgumentError, "buffer shape mismatch: #{a.shape.inspect} vs #{b.shape.inspect}" unless a.shape == b.shape
      elsif a.length != b.length
        raise ArgumentError, "buffer length mismatch: #{a.length} vs #{b.length}"
      end

      ct = CTYPE.fetch(compute_dtype)
      src = <<~KERNEL
        GPU_KERNEL void ew(GPU_ARGS,
                           device const #{CTYPE.fetch(a.dtype)}* a,
                           device const #{CTYPE.fetch(b.dtype)}* b,
                           device #{CTYPE.fetch(out_dtype)}* c,
                           constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) c[i] = #{expr.call("(#{ct})#{read_dtype(a.dtype, 'a[i]')}", "(#{ct})#{read_dtype(b.dtype, 'b[i]')}")};
        }
      KERNEL

      kernel = compiled(
        dialect_name: "ew", dialect_src: src,
        cpu_name: "ew",
        cpu_src: -> { codegen.buffer_binary("ew", a_dtype: a.dtype, b_dtype: b.dtype, out_dtype: out_dtype,
                                             compute_dtype: compute_dtype, &cpu_expr.call) },
      )
      launch_into(kernel, a.length, [a, b], out_dtype: out_dtype, shape: a.shape)
    end

    def scalar_binary(a, scalar, compute_dtype, out_dtype, reversed, cpu_expr, &expr)
      ct = CTYPE.fetch(compute_dtype)
      lhs, rhs = "(#{ct})#{read_dtype(a.dtype, 'a[i]')}", "s"
      lhs, rhs = rhs, lhs if reversed
      src = <<~KERNEL
        GPU_KERNEL void ew(GPU_ARGS,
                           device const #{CTYPE.fetch(a.dtype)}* a,
                           constant #{ct}& s,
                           device #{CTYPE.fetch(out_dtype)}* c,
                           constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) c[i] = #{expr.call(lhs, rhs)};
        }
      KERNEL

      kernel = compiled(
        dialect_name: "ew", dialect_src: src,
        cpu_name: "ew",
        cpu_src: -> { codegen.scalar_binary("ew", a_dtype: a.dtype, out_dtype: out_dtype,
                                             compute_dtype: compute_dtype, reversed: reversed, &cpu_expr.call) },
      )
      converted = compute_dtype == :float32 ? scalar.to_f : scalar
      launch_into(kernel, a.length, [a, converted], out_dtype: out_dtype, shape: a.shape)
    end

    def unary(a, out_dtype:, cpu_name:, cpu_src:, &expr)
      require_computable!(a)
      src = <<~KERNEL
        GPU_KERNEL void ew(GPU_ARGS,
                           device const #{CTYPE.fetch(a.dtype)}* a,
                           device #{CTYPE.fetch(out_dtype)}* c,
                           constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) c[i] = #{expr.call(read(a, 'a[i]'))};
        }
      KERNEL

      kernel = compiled(dialect_name: "ew", dialect_src: src, cpu_name: cpu_name, cpu_src: cpu_src)
      launch_into(kernel, a.length, [a], out_dtype: out_dtype, shape: a.shape)
    end

    def unary_math(fn, a)
      unary(a, out_dtype: LOW_PRECISION_DTYPES.include?(a.dtype) ? a.dtype : :float32,
            cpu_name: "ew_#{fn}",
            cpu_src: -> { codegen.unary_math(fn, a_dtype: a.dtype) }) { |x| "#{fn}((float)#{x})" }
    end

    def unary_abs(a)
      fn = a.dtype == :int32 ? "abs" : "fabs"
      unary(a, out_dtype: a.dtype, cpu_name: "abs_#{a.dtype}",
            cpu_src: -> { codegen.unary_abs(a_dtype: a.dtype) }) { |x| "#{fn}(#{x})" }
    end

    def clamp_op(a, lo, hi)
      require_computable!(a)
      unless lo.is_a?(Numeric) && hi.is_a?(Numeric)
        raise TypeError, "clamp bounds must be Numeric"
      end

      dtype = a.dtype == :int32 && lo.is_a?(Integer) && hi.is_a?(Integer) ? :int32 : :float32
      out_dtype = LOW_PRECISION_DTYPES.include?(a.dtype) ? a.dtype : dtype
      ct = CTYPE.fetch(dtype)
      src = <<~KERNEL
        GPU_KERNEL void ew(GPU_ARGS,
                           device const #{CTYPE.fetch(a.dtype)}* a,
                           constant #{ct}& lo,
                           constant #{ct}& hi,
                           device #{CTYPE.fetch(out_dtype)}* c,
                           constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < n) c[i] = clamp((#{ct})#{read_dtype(a.dtype, 'a[i]')}, lo, hi);
        }
      KERNEL

      kernel = compiled(
        dialect_name: "ew", dialect_src: src,
        cpu_name: "clamp_ew", cpu_src: -> { codegen.clamp(a_dtype: a.dtype, dtype: dtype) },
      )
      convert = ->(v) { dtype == :float32 ? v.to_f : v }
      launch_into(kernel, a.length, [a, convert.(lo), convert.(hi)], out_dtype: out_dtype, shape: a.shape)
    end

    def launch_into(kernel, n, inputs, out_dtype:, shape: [n])
      out = allocate_output(out_dtype, n, shape)
      launch_kernel(kernel, grid: (n / BLOCK.to_f).ceil, block: BLOCK, args: [*inputs, out, n])
      out
    rescue StandardError
      out&.free
      raise
    end

    # Backends that cannot use the dialect use the bridge's own ops_codegen
    # (lib/tilework/backend/cpu/ops_codegen.rb on :cpu), passed with raw: true.
    # The dialect backends' path (dialect_src straight into Tilework.compile) is
    # unchanged (the same pattern as DSLKernel in lib/tilework/dsl.rb).
    def compiled(dialect_name:, dialect_src:, cpu_name:, cpu_src:)
      return Tilework.compile(dialect_src, name: dialect_name) if Tilework.capabilities[:dialect]

      unless Tilework.bridge.respond_to?(:ops_codegen)
        raise Error, "the #{Tilework.backend.inspect} backend doesn't support dialect kernel strings " \
                     "and doesn't implement ops_codegen either"
      end
      Tilework.compile(cpu_src.call, name: cpu_name, raw: true)
    end

    # Matrix multiply. Requires 2D shapes a: [m, k] and b: [k, n]; the output is
    # [m, n]. A naive implementation where each thread on a 1D grid computes one
    # output element's dot product (no tiling or shared-memory optimization).
    def matmul_op(a, b, out_dtype: nil)
      raise TypeError, "matmul requires a Tilework::Buffer, got #{b.class}" unless b.is_a?(Buffer)
      unless a.matrix? && b.matrix?
        raise ArgumentError, "matmul requires 2D buffers (shape), got #{a.shape.inspect} and #{b.shape.inspect}"
      end
      unless a.cols == b.rows
        raise ArgumentError, "matmul shape mismatch: #{a.shape.inspect} x #{b.shape.inspect}"
      end

      m, k, n = a.rows, a.cols, b.cols
      dtype = promoted_dtype(a, b)
      out_dtype ||= dtype
      allowed = [dtype]
      allowed << a.dtype if a.dtype == b.dtype && LOW_PRECISION_DTYPES.include?(a.dtype)
      unless allowed.include?(out_dtype)
        raise ArgumentError, "matmul out_dtype must be #{allowed.map(&:inspect).join(' or ')} " \
                             "for #{a.dtype.inspect} x #{b.dtype.inspect} inputs"
      end

      require_computable!(a, b)

      # Fast GEMM paths. This is the only branch point; anything that takes
      # none of them falls through to the SIMT kernel below.
      #   1. a verified bridge-specific matrix-core kernel (tile-aligned shapes)
      #   2. a GEMM library (no shape constraints)
      #   3. CUDA's WMMA kernel
      #   4. otherwise SIMT
      result = if TensorCore.matrix_core_eligible?(a, b, dtype)
                 TensorCore.matrix_core_gemm(a, b, m, k, n)
               elsif TensorCore.gemm_eligible?(a, b, dtype)
                 TensorCore.gemm(a, b, m, k, n)
               elsif TensorCore.eligible?(a, b, dtype)
                 TensorCore.matmul(a, b, m, k, n)
               end
      return cast_matmul_output(result, out_dtype) if result

      ct = CTYPE.fetch(dtype)
      src = <<~KERNEL
        GPU_KERNEL void matmul(GPU_ARGS,
                               device const #{CTYPE.fetch(a.dtype)}* a,
                               device const #{CTYPE.fetch(b.dtype)}* b,
                               device #{ct}* c,
                               constant int& m,
                               constant int& k,
                               constant int& n) {
          int i = GLOBAL_ID_X;
          if (i < m * n) {
            int row = i / n;
            int col = i % n;
            #{ct} acc = #{ZERO.fetch(dtype)};
            for (int t = 0; t < k; t++) {
              acc += (#{ct})#{read_dtype(a.dtype, 'a[row * k + t]')} * (#{ct})#{read_dtype(b.dtype, 'b[t * n + col]')};
            }
            c[i] = acc;
          }
        }
      KERNEL

      kernel = compiled(
        dialect_name: "matmul", dialect_src: src,
        cpu_name: "matmul", cpu_src: -> { codegen.matmul(a_dtype: a.dtype, b_dtype: b.dtype, dtype: dtype) },
      )
      out = allocate_output(dtype, m * n, [m, n])
      launch_kernel(kernel, grid: (m * n / BLOCK.to_f).ceil, block: BLOCK, args: [a, b, out, m, k, n])
      cast_matmul_output(out, out_dtype)
    end

    def cast_matmul_output(out, out_dtype)
      return out if out.dtype == out_dtype

      converted = out.cast(out_dtype)
      out.free
      converted
    rescue StandardError
      out.free
      raise
    end

    # Builds per-threadgroup partial reductions on the GPU.
    #
    # There are two ways to receive the result:
    #
    #   buffer.sum                # => Float. Runs one pass and folds the
    #                             #    partials on the host
    #   buffer.sum(device: true)  # => a one-element Buffer, folded all the
    #                             #    way on the device
    #
    # **Receiving on the host drains the queue** — Buffer#to_a waits for
    # pending kernels, so every reduction becomes a full synchronization.
    # Reductions come at the end of loops (loss, sampling, normalizer
    # denominators), so dropping to the host each time rules out "queue the
    # launches and wait once at the end". Use device: true to feed the result
    # straight into the next operation.
    #
    # device: true reapplies the kernel until one partial is left (folding 256
    # at a time, so even a million elements take three passes).
    #
    # Backends with Tilework.host_fallback? (:cpu) bypass kernel generation and
    # compute directly in Ruby. GPU_SHARED/GPU_BARRIER assume every thread in a
    # group runs the body at once and synchronizes at barriers; reproducing that
    # correctly on a CPU would need an interpreter emulating barrier phases.
    # Only these two functions (reduce/arg_reduce) use GPU_SHARED/GPU_BARRIER,
    # internally to Ops (not exposed to user Tilework.kernel {}), so a dedicated
    # interpreter is not worth it (see the top of
    # lib/tilework/backend/cpu/ops_codegen.rb).
    def reduce(kind, buffer, device: false)
      require_computable!(buffer)
      if Tilework.host_fallback?
        value = buffer.to_a.public_send(kind)
        return device ? scalar_buffer(value, accumulator_dtype(buffer)) : value
      end

      partial = reduce_tree(kind, buffer, device: device)
      return partial if device

      value = partial.to_a.public_send(kind)
      partial.free
      value
    end

    # With device: false it runs one pass and returns the partials (the caller
    # folds them); with device: true it reapplies the kernel until one is left.
    def reduce_tree(kind, buffer, device:)
      acc_dtype = accumulator_dtype(buffer)
      source = buffer
      loop do
        groups = (source.length / BLOCK.to_f).ceil
        # Take the partials and the final value from the execution context
        # (borrow and return symmetrically). The context takes over frees, so a
        # Buffer.new temporary would become a slot that is returned but never
        # borrowed, growing the pool every plan run.
        partial = allocate_output(acc_dtype, groups, [groups])
        launch_kernel(reduce_kernel(kind, source.dtype, acc_dtype),
                      grid: groups, block: BLOCK, args: [source, partial, source.length])
        source.free unless source.equal?(buffer) # intermediate partials are ours to free
        source = partial
        break if !device || groups == 1
      end
      source
    rescue StandardError
      partial&.free unless partial&.equal?(source)
      source&.free unless source&.equal?(buffer)
      raise
    end

    # Accumulating in half precision piles up error as the count grows, so
    # reduce in float and return float32 (accumulation is always FP32).
    def accumulator_dtype(buffer) = LOW_PRECISION_DTYPES.include?(buffer.dtype) ? :float32 : buffer.dtype

    def scalar_buffer(value, dtype)
      allocate_output(dtype, 1, [1]).tap { |buffer| buffer.write([value]) }
    end

    # From the second pass the input is partials (in the accumulator type), so
    # the input type is a parameter.
    def reduce_kernel(kind, in_dtype, acc_dtype)
      spec = REDUCTIONS.fetch(kind)
      ctype = CTYPE.fetch(acc_dtype)
      element = read_dtype(in_dtype, "a[gid]")
      src = <<~KERNEL
        GPU_KERNEL void reduce_#{kind}(GPU_ARGS,
                                       device const #{CTYPE.fetch(in_dtype)}* a,
                                       device #{ctype}* partial,
                                       constant int& n) {
          GPU_SHARED #{ctype} sh[#{BLOCK}];
          int gid = GLOBAL_ID_X;
          int lid = LOCAL_ID_X;
          sh[lid] = gid < n ? #{element} : #{spec[:identity].fetch(acc_dtype)};
          GPU_BARRIER;
          for (int s = #{BLOCK / 2}; s > 0; s >>= 1) {
            if (lid < s) #{spec[:combine]}
            GPU_BARRIER;
          }
          if (lid == 0) partial[GROUP_ID_X] = sh[0];
        }
      KERNEL

      compiled(
        dialect_name: "reduce_#{kind}", dialect_src: src,
        cpu_name: "reduce_#{kind}", cpu_src: -> { raise NotImplementedError },
      )
    end

    # Mean. With device: true both the sum and the division finish on the device.
    def mean_op(buffer, device: false)
      return reduce(:sum, buffer).fdiv(buffer.length) unless device

      total = reduce(:sum, buffer, device: true)
      mean = arithmetic(:/, total, buffer.length.to_f)
      total.free
      mean
    rescue StandardError
      total&.free
      raise
    end

    # Reduction over (value, index) pairs. Ties keep the smaller index (strict
    # comparison, left-biased). Later passes keep indices ascending, so the same
    # strict comparison picks the same winners — device: true and host folding
    # agree. See #reduce about the early return on :cpu.
    def arg_reduce(kind, buffer, device: false)
      require_computable!(buffer)
      if Tilework.host_fallback?
        index = host_arg_reduce(kind, buffer.to_a)
        return device ? scalar_buffer(index, :int32) : index
      end

      values, indices = arg_reduce_pass(kind, buffer, nil)
      unless device
        index = host_arg_reduce(kind, values.to_a, indices.to_a)
        values.free
        indices.free
        return index
      end

      until values.length == 1
        next_values, next_indices = arg_reduce_pass(kind, values, indices)
        values.free
        indices.free
        values, indices = next_values, next_indices
      end
      values.free
      indices # only the indices are the answer
    end

    # One pass. With indices nil the indices come from thread ids (the first
    # pass); otherwise the indices chosen by the previous pass are carried along.
    def arg_reduce_pass(kind, source, indices)
      groups = (source.length / BLOCK.to_f).ceil
      # Borrow and return symmetrically (the same reason as reduce_tree). pval
      # is always a temporary freed within this operation; only the final pidx
      # goes to the caller and becomes the Plan's.
      pval = allocate_output(source.dtype, groups, [groups])
      pidx = allocate_output(:int32, groups, [groups])
      args = indices ? [source, indices, pval, pidx] : [source, pval, pidx]
      launch_kernel(arg_reduce_kernel(kind, source.dtype, carry: !indices.nil?),
                    grid: groups, block: BLOCK, args: [*args, source.length])
      [pval, pidx]
    rescue StandardError
      pval&.free
      pidx&.free
      raise
    end

    def arg_reduce_kernel(kind, dtype, carry:)
      cmp = kind == :max ? ">" : "<"
      ctype = CTYPE.fetch(dtype)
      identity = REDUCTIONS.fetch(kind)[:identity].fetch(dtype)
      name = carry ? "arg_#{kind}_pairs" : "arg_#{kind}"
      params = ["device const #{ctype}* a"]
      params << "device const int* idx" if carry
      params += ["device #{ctype}* pval", "device int* pidx", "constant int& n"]
      src = <<~KERNEL
        GPU_KERNEL void #{name}(GPU_ARGS, #{params.join(', ')}) {
          GPU_SHARED #{ctype} sv[#{BLOCK}];
          GPU_SHARED int si[#{BLOCK}];
          int gid = GLOBAL_ID_X;
          int lid = LOCAL_ID_X;
          sv[lid] = gid < n ? a[gid] : #{identity};
          si[lid] = #{carry ? 'gid < n ? idx[gid] : 0' : 'gid'};
          GPU_BARRIER;
          for (int s = #{BLOCK / 2}; s > 0; s >>= 1) {
            if (lid < s && sv[lid + s] #{cmp} sv[lid]) {
              sv[lid] = sv[lid + s];
              si[lid] = si[lid + s];
            }
            GPU_BARRIER;
          }
          if (lid == 0) {
            pval[GROUP_ID_X] = sv[0];
            pidx[GROUP_ID_X] = si[0];
          }
        }
      KERNEL

      compiled(
        dialect_name: name, dialect_src: src,
        cpu_name: name, cpu_src: -> { raise NotImplementedError },
      )
    end

    # Host side of index reductions. Without indices, indices count from 0.
    def host_arg_reduce(kind, values, indices = nil)
      best_value = best_index = nil
      values.each_with_index do |value, i|
        next unless best_value.nil? || (kind == :max ? value > best_value : value < best_value)

        best_value = value
        best_index = indices ? indices[i] : i
      end
      best_index
    end
  end

  class Buffer
    include Ops
  end
end
