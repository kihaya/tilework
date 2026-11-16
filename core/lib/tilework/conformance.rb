# frozen_string_literal: true

module Tilework
  # The conformance suite.
  #
  # Ships the proof of the layering — "Ops, the DSL and with_gpu give the same
  # numbers on every backend" — as library code in core rather than as a test
  # file. A backend gem's CI gets the same checks as :cpu without writing a
  # single assertion of its own:
  #
  #   require "tilework"
  #   require "tilework/conformance"
  #   report = Tilework::Conformance.run!(backend: :metal)
  #   raise report.summary unless report.success?
  #
  # A check that did not run must never turn green by skipping. So each check
  # records exceptions in a Result instead of swallowing them, and
  # Report#success? is true only when **all** of these hold:
  #
  #   1. the backend was usable          (otherwise a non-green unavailable?)
  #   2. at least one check actually ran (zero is never green)
  #   3. every check that ran passed
  #
  # 2 is needed because if success? were true with an empty CHECKS, "green with
  # nothing checked" would be possible, defeating the suite's purpose.
  module Conformance
    Result = Struct.new(:name, :error) do
      def passed? = error.nil?
    end

    class Report
      attr_reader :backend, :results, :unavailable_reason

      def initialize(backend, results, unavailable_reason: nil)
        @backend = backend
        @results = results
        @unavailable_reason = unavailable_reason
      end

      # The backend itself was unusable, so not a single check could run.
      def unavailable? = !@unavailable_reason.nil?

      def failures = results.reject(&:passed?)

      # Green only when at least one check actually ran and all of them passed.
      # A missing runtime and zero checks run are never green — reporting checks
      # that did not run as passing would defeat the suite's purpose.
      def success?
        return false if unavailable? || results.empty?

        results.all?(&:passed?)
      end

      def summary
        if unavailable?
          return "conformance suite did not run on #{backend.inspect} " \
                 "(no execution environment): #{unavailable_reason}"
        end
        return "no conformance checks ran on #{backend.inspect}" if results.empty?

        base = "#{results.count(&:passed?)}/#{results.length} conformance checks passed on #{backend.inspect}"
        return base if failures.empty?

        "#{base}\n#{failures.map { |r| "  #{r.name}: #{r.error.message}" }.join("\n")}"
      end
    end

    CHECKS = []

    class << self
      def register(name, &block)
        CHECKS << [name, block]
      end

      def names = CHECKS.map(&:first)

      # Selects backend and runs every registered check.
      #
      # Even where the backend is unusable (extension not built, no device, ...)
      # it returns a **non-green Report** instead of raising, so the caller's
      # boilerplate
      #
      #   raise report.summary unless report.success?
      #
      # doubles as the report that the runtime is missing, keeping CI from
      # passing while believing it ran the suite. With the backend registry,
      # backend= raises the same DeviceError (a failed require) for an unknown
      # backend name and an uninstalled backend gem, so both become a non-green
      # Report here.
      def run!(backend:)
        begin
          Tilework.backend = backend
        rescue DeviceError => e
          return Report.new(backend, [], unavailable_reason: e.message)
        end

        results = CHECKS.map { |name, _| Result.new(name, run_check_error(name)) }
        Report.new(backend, results)
      end

      # Runs a single check (the Minitest adapter turns each into its own
      # test_ method). Failures raise as is (to combine with Minitest's
      # assert_raises and the like).
      def run_check!(name)
        _, block = CHECKS.find { |n, _| n == name }
        raise ArgumentError, "unknown conformance check: #{name.inspect}" unless block

        Tilework::Kernel.clear_cache!
        instance_eval(&block)
        nil
      ensure
        # Native modules and buffers are TypedData objects.  A full backend
        # conformance run intentionally compiles many short-lived pipelines;
        # clear the cache on both sides of a check and collect them here so a
        # long run does not rely on an implementation-specific GC threshold
        # to release device modules, streams, or signal-pool entries.
        Tilework::Kernel.clear_cache!
        GC.start
      end

      private

      def run_check_error(name)
        run_check!(name)
        nil
      rescue StandardError => e
        e
      end
    end

    # --- Small helpers used by the checks ------------------------------------

    def self.f32(array) = array.pack("f*").unpack("f*")

    def self.assert_close(expected, actual, tol = 1e-5)
      unless expected.length == actual.length
        raise "length mismatch: expected #{expected.length}, got #{actual.length}"
      end

      expected.zip(actual).each_with_index do |(e, a), i|
        raise "element #{i}: expected #{e.inspect}, got #{a.inspect} (tolerance #{tol})" if (e - a).abs > tol
      end
    end

    # A comparison relative to the array's magnitude. **assert_close checks
    # nothing on arrays of small values** — an absolute tolerance of 1e-5 on
    # values around 1e-7, like gradients, passes everything.
    def self.assert_close_relative(expected, actual, tol = 1e-5)
      scale = expected.map(&:abs).max
      # All zeros falls back to absolute comparison (relative error is undefined).
      assert_close(expected, actual, scale.zero? ? tol : tol * scale)
    end

    def self.assert_equal_arrays(expected, actual)
      raise "expected #{expected.inspect}, got #{actual.inspect}" unless expected == actual
    end

    # Runs with the portable RNG, then restores the original mode.
    def self.with_portable_random
      previous = Random.mode
      Random.portable!
      yield
    ensure
      Random.mode = previous
    end

    # An n x n Buffer that is not a view. Building from a nested array returns
    # an owning buffer (reshape gives a view, which is no use for paging checks).
    def self.square(n) = Tilework[Array.new(n) { f32(Array.new(n) { rand }) }]

    def self.assert(cond, message)
      raise message unless cond
    end

    # --- Ops (operators) -----------------------------------------------------

    register(:ops_buffer_addition) do
      a = Conformance.f32(Array.new(64) { rand })
      b = Conformance.f32(Array.new(64) { rand })
      c = Tilework[a] + Tilework[b]
      Conformance.assert_close(a.zip(b).map { |x, y| x + y }, c.to_a)
    end

    register(:bfloat16_storage_ops_and_graph_forward) do
      values = [1.0, -2.0, 4.0, 8.0]
      input = Tilework[values, dtype: :bfloat16]
      Conformance.assert(input.bytesize == 8, "bfloat16 must use two bytes per element")
      Conformance.assert_equal_arrays(values, input.to_a)
      doubled = input * 2.0
      Conformance.assert(doubled.dtype == :bfloat16, "bfloat16 scalar arithmetic must preserve storage dtype")
      Conformance.assert_equal_arrays(values.map { |value| value * 2.0 }, doubled.to_a)
      total = input.sum(device: true)
      Conformance.assert(total.dtype == :float32, "bfloat16 reductions must accumulate into float32")
      Conformance.assert_close([values.sum], total.to_a)

      matrix = Tilework[[[1.0, 2.0], [3.0, 4.0]], dtype: :bfloat16]
      plan = Graph.build do |g|
        x = g.input(:x, [2, 2], dtype: :bfloat16)
        g.output(x.matmul(g.param(matrix), out_dtype: :bfloat16) + 1.0)
      end.compile
      output = plan.run(x: matrix)
      Conformance.assert(output.dtype == :bfloat16, "graph output must preserve requested bfloat16 dtype")
      Conformance.assert_equal_arrays([8.0, 11.0, 16.0, 23.0], output.to_a)

      probs = NN.softmax(matrix)
      Conformance.assert(probs.dtype == :bfloat16, "NN forward must preserve bfloat16 storage")
      probs.to_a.each_slice(2) do |row|
        Conformance.assert_close([1.0], [row.sum], 0.01)
      end

      cache = Buffer.new(:bfloat16, 4, shape: [2, 2])
      state_plan = Graph.build do |g|
        state = g.state(cache, name: :bf16_cache)
        value = g.input(:value, [2, 2], dtype: :bfloat16)
        g.output(g.read(g.write_into(state, value: value)))
      end.compile
      Conformance.assert_equal_arrays(matrix.to_a, state_plan.run(value: matrix).to_a)
      Conformance.assert_equal_arrays(matrix.to_a, cache.to_a)
    end

    register(:ops_scalar_multiplication_and_coerce) do
      a = Conformance.f32(Array.new(64) { rand })
      buf = Tilework[a]
      Conformance.assert_close(a.map { |x| x * 2.0 }, (buf * 2.0).to_a)
      Conformance.assert_close(a.map { |x| x * 2.0 }, (2.0 * buf).to_a)
      Conformance.assert_close(a.map { |x| 10.0 - x }, (10.0 - buf).to_a)
    end

    register(:ops_int32_arithmetic_is_exact) do
      a = Array.new(64) { rand(-1000..1000) }
      b = Array.new(64) { rand(1..100) }
      da = Tilework[a]
      db = Tilework[b]
      Conformance.assert_equal_arrays(a.zip(b).map { |x, y| x - y }, (da - db).to_a)
      Conformance.assert_equal_arrays(a.map { |x| x * 3 }, (da * 3).to_a)
    end

    register(:ops_unary_math) do
      a = Conformance.f32(Array.new(64) { (rand * 4) + 0.1 })
      buf = Tilework[a]
      Conformance.assert_close(a.map { |x| Math.exp(x) }, buf.exp.to_a, 1e-3)
      Conformance.assert_close(a.map { |x| Math.log(x) }, buf.log.to_a)
      Conformance.assert_close(a.map { |x| Math.sqrt(x) }, buf.sqrt.to_a)
    end

    register(:ops_abs_and_clamp) do
      a = Conformance.f32(Array.new(64) { (rand * 2) - 1 })
      buf = Tilework[a]
      Conformance.assert_close(a.map(&:abs), buf.abs.to_a)
      Conformance.assert_close(a.map { |x| x.clamp(-0.5, 0.5) }, buf.clamp(-0.5, 0.5).to_a)
    end

    register(:ops_comparison_masks) do
      a = Conformance.f32(Array.new(64) { rand })
      b = Conformance.f32(Array.new(64) { rand })
      da = Tilework[a]
      Conformance.assert_equal_arrays(a.zip(b).map { |x, y| x > y ? 1 : 0 }, (da > Tilework[b]).to_a)
      Conformance.assert_equal_arrays(a.map { |x| x <= 0.5 ? 1 : 0 }, (da <= 0.5).to_a)
    end

    register(:ops_elementwise_maximum_minimum) do
      a = Conformance.f32(Array.new(64) { rand })
      b = Conformance.f32(Array.new(64) { rand })
      Conformance.assert_close(a.zip(b).map(&:max), Tilework[a].maximum(Tilework[b]).to_a)
      Conformance.assert_close(a.map { |x| [x, 0.5].min }, Tilework[a].minimum(0.5).to_a)
    end

    register(:ops_min_max_mean_reductions) do
      a = Conformance.f32(Array.new(500) { (rand * 200) - 100 })
      buf = Tilework[a]
      Conformance.assert_equal_arrays(a.min, buf.min)
      Conformance.assert_equal_arrays(a.max, buf.max)
      Conformance.assert_close([a.sum / a.length], [buf.mean], (a.sum.abs / a.length * 1e-4) + 1e-4)
    end

    register(:ops_sum_reduction) do
      a = Conformance.f32(Array.new(2_000) { rand })
      Conformance.assert_close([a.sum], [Tilework[a].sum], a.sum.abs * 1e-4)
      ints = Array.new(2_000) { rand(-100..100) }
      Conformance.assert_equal_arrays(ints.sum, Tilework[ints].sum)
    end

    register(:ops_argmax_argmin) do
      a = Conformance.f32(Array.new(500) { rand })
      buf = Tilework[a]
      Conformance.assert_equal_arrays(a.index(a.max), buf.argmax)
      Conformance.assert_equal_arrays(a.index(a.min), buf.argmin)
    end

    # device: true folds all the way without reaching the host; the same input
    # gives the same answer (the element count exceeds one threadgroup = 256,
    # so real backends take two or more passes).
    register(:ops_device_resident_reductions_match_the_host_values) do
      a = Conformance.f32(Array.new(5_000) { (rand * 200) - 100 })
      buf = Tilework[a]
      tolerance = (a.sum.abs * 1e-4) + 1e-3
      Conformance.assert_close([buf.sum], buf.sum(device: true).to_a, tolerance)
      Conformance.assert_close([buf.min], buf.min(device: true).to_a)
      Conformance.assert_close([buf.max], buf.max(device: true).to_a)
      Conformance.assert_close([buf.mean], buf.mean(device: true).to_a, (tolerance / a.length) + 1e-4)

      ints = Array.new(5_000) { rand(-100..100) }
      device_sum = Tilework[ints].sum(device: true)
      Conformance.assert(device_sum.dtype == :int32, "an int32 sum must stay int32 on the device")
      Conformance.assert_equal_arrays([ints.sum], device_sum.to_a)
    end

    # Even in later passes carrying indices, ties keep the smaller index (the
    # same strict comparison as host folding decides winners).
    register(:ops_device_resident_arg_reductions_match_the_host_index) do
      a = Conformance.f32(Array.new(5_000) { rand })
      a[1_000] = 2.0
      a[4_000] = 2.0  # the max appears twice; the smaller index is the answer
      a[2_500] = -1.0
      buf = Tilework[a]

      argmax = buf.argmax(device: true)
      Conformance.assert(argmax.dtype == :int32, "argmax returns an index, so the buffer is int32")
      Conformance.assert_equal_arrays([buf.argmax], argmax.to_a)
      Conformance.assert_equal_arrays([1_000], argmax.to_a)
      Conformance.assert_equal_arrays([buf.argmin], buf.argmin(device: true).to_a)
      Conformance.assert_equal_arrays([2_500], buf.argmin(device: true).to_a)
    end

    register(:ops_matmul) do
      m, k, n = 12, 9, 7
      a_rows = Array.new(m) { Conformance.f32(Array.new(k) { (rand * 2) - 1 }) }
      b_rows = Array.new(k) { Conformance.f32(Array.new(n) { (rand * 2) - 1 }) }
      da = Tilework::Buffer.from_matrix(a_rows)
      db = Tilework::Buffer.from_matrix(b_rows)
      c = da.matmul(db)
      Conformance.assert_equal_arrays([m, n], c.shape)
      expected = Array.new(m) { |i| Array.new(n) { |j| (0...k).sum { |t| a_rows[i][t] * b_rows[t][j] } } }.flatten
      Conformance.assert_close(expected, c.to_a, 1e-2)
    end

    register(:ops_matmul_int32_is_exact) do
      a = Tilework::Buffer.from_matrix([[1, 2, 3], [4, 5, 6]])
      b = Tilework::Buffer.from_matrix([[7, 8], [9, 10], [11, 12]])
      Conformance.assert_equal_arrays([58, 64, 139, 154], a.matmul(b).to_a)
    end

    # --- NN (fused kernels) --------------------------------------------------

    def self.nn_matrix(rows, cols, values)
      buf = Buffer.new(:float32, rows * cols, shape: [rows, cols])
      buf.write(values)
      buf
    end

    register(:nn_softmax_matches_the_closed_form) do
      out = NN.softmax(Conformance.nn_matrix(1, 3, [1.0, 2.0, 3.0])).to_a
      exps = [1.0, 2.0, 3.0].map { |v| Math.exp(v - 3.0) }
      Conformance.assert_close(exps.map { |e| e / exps.sum }, out)
    end

    register(:nn_softmax_survives_large_logits) do
      out = NN.softmax(Conformance.nn_matrix(1, 4, [1000.0, 1001.0, 1002.0, 1003.0])).to_a
      Conformance.assert(out.none? { |v| v.nan? || v.infinite? }, "large logits overflowed")
      Conformance.assert_close([1.0], [out.sum])
    end

    register(:nn_layernorm_gives_zero_mean_and_unit_variance) do
      out = NN.layernorm(Conformance.nn_matrix(1, 4, [1.0, 2.0, 3.0, 4.0])).to_a
      Conformance.assert_close([0.0], [out.sum / out.length], 1e-5)
      Conformance.assert_close([1.0], [out.sum { |v| v * v } / out.length], 1e-3)
    end

    register(:nn_layernorm_applies_gamma_and_beta) do
      x = Conformance.nn_matrix(1, 4, [1.0, 2.0, 3.0, 4.0])
      gamma = Buffer.new(:float32, 4)
      gamma.write([2.0, 2.0, 2.0, 2.0])
      beta = Buffer.new(:float32, 4)
      beta.write([1.0, 1.0, 1.0, 1.0])
      plain = NN.layernorm(x).to_a
      scaled = NN.layernorm(x, gamma: gamma, beta: beta).to_a
      Conformance.assert_close(plain.map { |v| (v * 2.0) + 1.0 }, scaled, 1e-5)
    end

    register(:nn_rmsnorm_divides_by_the_root_mean_square_without_centering) do
      x = Conformance.nn_matrix(2, 4, [3.0, 3.0, 3.0, 3.0, 1.0, -2.0, 3.0, -4.0])
      got = NN.rmsnorm(x, eps: 0.0).to_a
      want = [3.0, 3.0, 3.0, 3.0, 1.0, -2.0, 3.0, -4.0].each_slice(4).flat_map do |row|
        rms = Math.sqrt(row.sum { |v| v * v } / 4.0)
        row.map { |v| v / rms }
      end
      Conformance.assert_close(want, got, 1e-5)
    end

    register(:nn_groupnorm_normalizes_within_each_group) do
      n, c, h, w = 1, 4, 2, 2
      x = Buffer.new(:float32, n * c * h * w, shape: [n, c, h, w])
      x.write((0...16).map { |i| (i * 3 % 7).to_f })
      out = NN.groupnorm(x, groups: 2).to_a
      out.each_slice(8) do |group|
        Conformance.assert_close([0.0], [group.sum / 8.0], 1e-5)
        Conformance.assert_close([1.0], [group.sum { |v| v * v } / 8.0], 1e-3)
      end
    end

    register(:nn_relu_clamps_negatives_to_zero) do
      x = Buffer.new(:float32, 4)
      x.write([-2.0, -0.5, 0.5, 2.0])
      Conformance.assert_close([0.0, 0.0, 0.5, 2.0], NN.relu(x).to_a)
    end

    register(:nn_sigmoid_and_silu_match_their_definitions) do
      x = Buffer.new(:float32, 3)
      x.write([-1.0, 0.0, 2.0])
      want = [-1.0, 0.0, 2.0].map { |v| 1.0 / (1.0 + Math.exp(-v)) }
      Conformance.assert_close(want, NN.sigmoid(x).to_a)
      Conformance.assert_close([-1.0, 0.0, 2.0].each_with_index.map { |v, i| v * want[i] }, NN.silu(x).to_a)
    end

    register(:nn_gelu_variants_agree_closely_and_pass_through_zero) do
      x = Buffer.new(:float32, 5)
      x.write([-2.0, -0.5, 0.0, 0.5, 2.0])
      approx = NN.gelu(x).to_a
      exact = NN.gelu(x, exact: true).to_a
      Conformance.assert_close([0.0], [approx[2]], 1e-9)
      Conformance.assert_close(exact, approx, 2e-3) # the known gap of the tanh approximation
    end

    # Losses end in a reduction, so check that the host Float and the shape-[1]
    # Buffer left on the device agree (training loops use the latter).
    register(:nn_mse_matches_the_closed_form) do
      pred = Conformance.nn_matrix(2, 3, [0.5, -1.0, 2.0, 0.0, 1.0, -0.5])
      target = Conformance.nn_matrix(2, 3, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
      errors = [0.25, 1.0, 1.0, 1.0, 0.0, 2.25]

      Conformance.assert_close(errors, NN.mse(pred, target, reduction: :none).to_a)
      Conformance.assert_close([errors.sum / 6.0], [NN.mse(pred, target)])
      Conformance.assert_close([errors.sum], [NN.mse(pred, target, reduction: :sum)])
      Conformance.assert_close([errors.sum / 6.0], NN.mse(pred, target, device: true).to_a, 1e-5)
    end

    # Cross entropy **takes logits**. A naive log(softmax(x)) hits -inf where a
    # probability underflows to 0, so this pins the log-sum-exp computation
    # (the large-logits check below is the core of it).
    register(:nn_cross_entropy_matches_the_log_sum_exp_form) do
      rows = [[0.1, 0.5, -0.3], [2.0, 0.0, 1.0]]
      logits = Conformance.nn_matrix(2, 3, rows.flatten)
      labels = Buffer.new(:int32, 2, shape: [2])
      labels.write([1, 0])

      want = rows.each_with_index.map do |row, i|
        m = row.max
        Math.log(row.sum { |v| Math.exp(v - m) }) - (row[[1, 0][i]] - m)
      end
      Conformance.assert_close(want, NN.cross_entropy(logits, labels, reduction: :none).to_a, 1e-5)
      Conformance.assert_close([want.sum / 2.0], [NN.cross_entropy(logits, labels)], 1e-5)
      Conformance.assert_close([want.sum], [NN.cross_entropy(logits, labels, reduction: :sum)], 1e-5)
      Conformance.assert_close([want.sum / 2.0], NN.cross_entropy(logits, labels, device: true).to_a, 1e-5)
    end

    register(:nn_cross_entropy_survives_large_logits) do
      logits = Conformance.nn_matrix(1, 4, [1000.0, 1001.0, 1002.0, 1003.0])
      labels = Buffer.new(:int32, 1, shape: [1])
      labels.write([3])
      value = NN.cross_entropy(logits, labels)
      Conformance.assert(!value.nan? && !value.infinite?, "large logits overflowed (got #{value})")
      # The correct class is the largest, so the loss settles at log(1 + e^-1 + e^-2 + e^-3).
      want = Math.log([0, 1, 2, 3].sum { |k| Math.exp(-k) })
      Conformance.assert_close([want], [value], 1e-4)
    end

    register(:nn_transpose_of_a_non_square_matrix_fills_every_element) do
      out = NN.transpose(Conformance.nn_matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
      Conformance.assert_equal_arrays([3, 2], out.shape)
      Conformance.assert_equal_arrays([1.0, 4.0, 2.0, 5.0, 3.0, 6.0], out.to_a)
    end

    register(:nn_multi_head_attention_matches_a_hand_written_reference) do
      heads, dh, n, m = 2, 3, 4, 5
      d = heads * dh
      qa = (0...(n * d)).map { |i| Math.sin(i * 0.7) }
      ka = (0...(m * d)).map { |i| Math.cos(i * 0.3) }
      va = (0...(m * d)).map { |i| Math.sin(i * 0.11) - 0.2 }

      got = NN.multi_head_attention(Conformance.nn_matrix(n, d, qa), Conformance.nn_matrix(m, d, ka),
                                     Conformance.nn_matrix(m, d, va), heads: heads).to_a

      scale = 1.0 / Math.sqrt(dh)
      want = Array.new(n * d, 0.0)
      heads.times do |h|
        (0...n).each do |r|
          s = (0...m).map { |c| (0...dh).sum { |j| qa[(r * d) + (h * dh) + j] * ka[(c * d) + (h * dh) + j] } * scale }
          e = s.map { |x| Math.exp(x - s.max) }
          (0...dh).each do |j|
            want[(r * d) + (h * dh) + j] = (0...m).sum { |c| (e[c] / e.sum) * va[(c * d) + (h * dh) + j] }
          end
        end
      end
      Conformance.assert_close(want, got, 1e-4)
    end

    # --- DSL (Ruby blocks) / with_gpu ----------------------------------------
    #
    # Each check builds its Tilework.kernel {} inside the check, so every run
    # uses a fresh DSLKernel and no generated source is carried across backend
    # switches.

    register(:dsl_vec_add_matches_reference) do
      vec_add = Tilework.kernel("conformance_vec_add") do |a, b, c, n|
        i = global_id_x
        c[i] = a[i] + b[i] if i < n
      end
      n = 500
      a = Conformance.f32(Array.new(n) { rand })
      b = Conformance.f32(Array.new(n) { rand })
      c = Buffer.new(:float32, n)
      vec_add.(Tilework[a], Tilework[b], c, n)
      Conformance.assert_close(a.zip(b).map { |x, y| x + y }, c.to_a)
    end

    register(:dsl_specializes_for_int_buffers) do
      vec_add = Tilework.kernel("conformance_vec_add_int") do |a, b, c, n|
        i = global_id_x
        c[i] = a[i] + b[i] if i < n
      end
      n = 200
      a = Array.new(n) { rand(-100..100) }
      b = Array.new(n) { rand(-100..100) }
      c = Buffer.new(:int32, n)
      vec_add.(Tilework[a], Tilework[b], c, n)
      Conformance.assert_equal_arrays(a.zip(b).map(&:sum), c.to_a)
    end

    register(:dsl_kernel_with_loop_computes_row_sums) do
      rows, cols = 8, 10
      m = Conformance.f32(Array.new(rows * cols) { rand })
      row_sum = Tilework.kernel("conformance_row_sum", out: 1) do |mat, out, rows, cols|
        r = global_id_x
        if r < rows
          acc = 0.0
          cols.times { |c| acc = acc + mat[(r * cols) + c] }
          out[r] = acc
        end
      end
      out = row_sum.(Tilework[m], rows, cols, over: rows)
      Conformance.assert_close(Array.new(rows) { |r| m[r * cols, cols].sum }, out.to_a, 1e-3)
    end

    register(:dsl_2d_launch_transpose) do
      size = 8
      m = Conformance.f32(Array.new(size * size) { rand })
      transpose = Tilework.kernel("conformance_transpose") do |src, dst, size|
        x = global_id_x
        y = global_id_y
        dst[(x * size) + y] = src[(y * size) + x] if x < size && y < size
      end
      dst = Buffer.new(:float32, size * size)
      transpose.launch(grid: [1, 1], block: [size, size], args: [Tilework[m], dst, size])
      expected = Array.new(size * size) { |i| x, y = i.divmod(size); m[(y * size) + x] }
      Conformance.assert_close(expected, dst.to_a)
    end

    register(:dsl_2d_indexing_transpose_with_nested_arrays) do
      rows, cols = 4, 5
      m = Array.new(rows) { |r| Array.new(cols) { |c| ((r * cols) + c).to_f } }
      transpose = Tilework.kernel("conformance_transpose2d", out: 1) do |mat, out|
        r = global_id_y
        c = global_id_x
        out[r, c] = mat[c, r] if c < mat.rows && r < mat.cols
      end
      result = Tilework.run(m) { |mat| transpose.(mat, over: [mat.rows, mat.cols]) }
      expected = cols.times.map { |c| rows.times.map { |r| m[r][c] } }
      Conformance.assert_equal_arrays(expected, result)
    end

    register(:dsl_unroll_expands_at_transpile_time) do
      weighted = Tilework.kernel("conformance_weighted", out: 1) do |a, out, n|
        i = global_id_x
        if i < n
          total = 0.0
          unroll([1.0, 2.0, 3.0]) { |w| total = total + (a[i] * w) }
          out[i] = total
        end
      end
      a = Buffer.from([1.0, 2.0, 3.0, 4.0])
      out = weighted.(a, 4)
      Conformance.assert_close([6.0, 12.0, 18.0, 24.0], out.to_a)
    end

    register(:dsl_math_functions) do
      sqrt_k = Tilework.kernel("conformance_sqrt") do |a, out, n|
        i = global_id_x
        out[i] = Math.sqrt(a[i]) if i < n
      end
      a = Buffer.from([4.0, 9.0, 16.0])
      out = Buffer.new(:float32, 3)
      sqrt_k.(a, out, 3)
      Conformance.assert_close([2.0, 3.0, 4.0], out.to_a)
    end

    # An end-to-end check that with_gpu (Tilework.run) bridges Array <-> Buffer
    # conversion, DSL kernels and operator chains.
    register(:with_gpu_and_dsl_end_to_end) do
      vec_add = Tilework.kernel("conformance_with_gpu_vec_add") do |a, b, c, n|
        i = global_id_x
        c[i] = a[i] + b[i] if i < n
      end
      n = 200
      xs = Conformance.f32(Array.new(n) { rand })
      ys = Conformance.f32(Array.new(n) { rand })

      result = Tilework.run(xs, ys) do |x, y|
        c = Buffer.new(:float32, n)
        vec_add.(x, y, c, n)
        (c * 2.0).clamp(0.0, 1.5)
      end

      expected = xs.zip(ys).map { |a, b| ((a + b) * 2.0).clamp(0.0, 1.5) }
      Conformance.assert_close(expected, result, 1e-5)
    end
    # --- Random (the portable path) ------------------------------------------
    #
    # Native RNG (cuRAND / MPS) sequences differ per backend, so only the
    # portable path (Random.portable!) can be checked here. **The expected
    # values are hard-coded** — rather than using :cpu as the oracle, this pins
    # that every backend produces these exact numbers.
    PORTABLE_UNIFORM_SEED_42 = [
      0.6129599213600159, 0.4685865640640259, 0.0732317566871643,
      0.34086155891418457, 0.9877187013626099, 0.3270633816719055,
    ].freeze

    PORTABLE_NORMAL_SEED_42 = [
      -0.9701898097991943, -1.2356207370758057, -0.07318136096000671, -1.106665015220642,
    ].freeze

    # Uniform is bit-identical (integer-to-float conversion uses only 24 bits and
    # powers of two, so no backend rounds).
    register(:random_portable_uniform_is_bit_identical) do
      Conformance.with_portable_random do
        got = Tilework.random(6, seed: 42).to_a
        Conformance.assert_equal_arrays(Conformance.f32(Conformance::PORTABLE_UNIFORM_SEED_42), got)
      end
    end

    # Normal goes through Box-Muller's sqrt/log/cos and agrees within their
    # precision (no bit identity — the last ulp depends on the math library).
    register(:random_portable_normal_matches_across_backends) do
      Conformance.with_portable_random do
        got = Tilework.randn(4, seed: 42).to_a
        Conformance.assert_close(Conformance::PORTABLE_NORMAL_SEED_42, got, 1e-5)
      end
    end

    # Counter-based, so element i does not depend on the request size.
    register(:random_portable_is_independent_of_the_request_size) do
      Conformance.with_portable_random do
        long = Tilework.random(64, seed: 7).to_a
        short = Tilework.random(5, seed: 7).to_a
        Conformance.assert_equal_arrays(long.first(5), short)
      end
    end

    register(:random_native_is_reproducible_within_a_process) do
      a = Tilework.random(32, seed: 1234).to_a
      b = Tilework.random(32, seed: 1234).to_a
      Conformance.assert_equal_arrays(a, b)
      Conformance.assert(a.all? { |x| x > 0.0 && x <= 1.0 }, "uniform values must lie in (0, 1]")
    end

    # Mixing async launches with host writes does not change a seed's result.
    # On unified-memory backends buffer_write is a raw memcpy, so racing an
    # in-flight kernel changes results intermittently.
    register(:random_survives_interleaved_async_work_and_host_writes) do
      run = lambda do
        noise = Tilework.random(512, seed: 4321)
        chain = noise
        3.times do |k|
          # Interleave a host write (Tilework[...] is buffer_write)
          weights = Tilework[Array.new(512) { |i| ((i + k) % 7) * 0.25 }]
          chain = ((chain * 1.5) + weights).clamp(-4.0, 4.0)
        end
        chain.to_a
      end
      Conformance.assert_equal_arrays(run.call, run.call)
    end

    # --- Graph layer (Tensor / Graph / Plan) ---

    register(:graph_matches_the_eager_path) do
      a = Tilework[Conformance.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])].reshape(2, 3)
      w = Tilework[Conformance.f32([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])].reshape(3, 2)
      plan = Tilework::Graph.build do |g|
        x = g.input(:x, [2, 3])
        g.output(((x.matmul(g.param(w)) * 2.0) + 1.0).tanh)
      end.compile
      expected = ((a.matmul(w) * 2.0) + 1.0).tanh.to_a
      Conformance.assert_close(expected, plan.run(x: a).to_a)
    end

    # Async launch + buffer reuse is bit-identical to waiting per node and
    # allocating every time. :cpu dispatches synchronously, so this matters only
    # on hardware.
    register(:graph_async_and_pooled_execution_matches_the_serial_one) do
      values = Conformance.f32(Array.new(512) { rand })
      build = lambda do
        plan = Tilework::Graph.build do |g|
          v = g.input(:x, [512])
          5.times { v = ((v * 1.5) + 0.25).tanh }
          g.output(v)
        end.compile
        plan.run(x: Tilework[values]).to_a
      end

      async = Tilework::Graph.async
      pool = Tilework::Graph.buffer_pool
      begin
        Tilework::Graph.async = true
        Tilework::Graph.buffer_pool = true
        pipelined = build.call
        Tilework::Graph.async = false
        Tilework::Graph.buffer_pool = false
        serial = build.call
      ensure
        Tilework::Graph.async = async
        Tilework::Graph.buffer_pool = pool
      end
      Conformance.assert_equal_arrays(serial, pipelined)
    end

    # Reductions are shape-[1] values inside a graph; also covers a plan with two outputs.
    register(:graph_reductions_match_the_eager_path) do
      values = Conformance.f32(Array.new(1_000) { (rand * 2) - 1 })
      buf = Tilework[values]
      plan = Tilework::Graph.build do |g|
        x = g.input(:x, [1_000])
        g.output((x * x).mean)
        g.output(x.argmax)
      end.compile

      mean, argmax = plan.run(x: buf)
      Conformance.assert_close([(buf * buf).mean], mean.to_a, 1e-4)
      Conformance.assert_equal_arrays([buf.argmax], argmax.to_a)
    end

    def self.nd(shape, values)
      buffer = Buffer.new(:float32, values.length, shape: shape)
      buffer.write(Conformance.f32(values))
      buffer
    end

    # Batched matmul becomes one strided batched GEMM when available and per-batch
    # 2D matmuls otherwise; this pins that **the path does not change values**.
    # A wrong column-major reinterpretation (swapped transposes or ld) breaks
    # only the delegated path, so m / k / n are all different to rule out shapes
    # matching by accident.
    register(:graph_batched_matmul_matches_per_batch_matmuls) do
      batch = 2
      m = 2
      k = 3
      n = 4
      a = Conformance.nd([batch, m, k], Array.new(batch * m * k) { |i| ((i % 5) - 2) * 0.5 })
      b = Conformance.nd([batch, k, n], Array.new(batch * k * n) { |i| ((i % 7) - 3) * 0.25 })

      plan = Tilework::Graph.build do |g|
        g.output(g.input(:a, [batch, m, k]).matmul(g.param(b)))
      end.compile

      expected = batch.times.flat_map do |i|
        a.slice(i).reshape(m, k).matmul(b.slice(i).reshape(k, n)).to_a
      end
      result = plan.run(a: a)
      Conformance.assert(result.shape == [batch, m, n],
                         "batched matmul should be #{[batch, m, n].inspect}, got #{result.shape.inspect}")
      Conformance.assert_close(expected, result.to_a, 1e-4)
    end

    # A broadcast gives the same answer as materializing the stretched side
    # first. The oracle is a same-size constant rearranged on the host, so value
    # mix-ups (rows and columns swapped, a misplaced stride 0) always show here.
    # The arithmetic is identical, so bit identity is required.
    register(:graph_broadcast_matches_an_explicitly_expanded_operand) do
      rows = 4
      cols = 3
      x = Conformance.nd([rows, cols], Array.new(rows * cols) { |i| ((i % 7) - 3) * 0.5 })
      row_vector = Conformance.f32([0.25, -1.5, 2.0])
      col_values = Conformance.f32([1.0, -2.0, 0.5, 3.0])
      tiled_rows = Conformance.nd([rows, cols], row_vector * rows)
      tiled_cols = Conformance.nd([rows, cols], col_values.flat_map { |v| [v] * cols })

      broadcast = Tilework::Graph.build do |g|
        t = g.input(:x, [rows, cols])
        g.output((t + g.param(Tilework[row_vector])) * g.param(Conformance.nd([rows, 1], col_values)))
      end.compile.run(x: x)

      expanded = Tilework::Graph.build do |g|
        t = g.input(:x, [rows, cols])
        g.output((t + g.param(tiled_rows)) * g.param(tiled_cols))
      end.compile.run(x: x)

      Conformance.assert(broadcast.shape == [rows, cols],
                         "broadcast should be #{[rows, cols].inspect}, got #{broadcast.shape.inspect}")
      Conformance.assert_equal_arrays(expanded.to_a, broadcast.to_a)
    end

    # Select by condition. An inverted mask or a scalar argument in the wrong
    # position shows up here as swapped values.
    register(:graph_where_selects_elementwise) do
      values = Conformance.f32(Array.new(64) { |i| ((i % 9) - 4) * 0.5 })
      x = Tilework[values]
      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [64])
        g.output(g.where(t > 0, t, 0.0))                 # tensor / scalar values
        g.output(g.where(t.eq(0.0), t + 1.0, t * 10.0))  # tensors on both sides
        g.output(g.where(t > 0, 1, 0))                   # scalars on both sides (an int32 mask)
      end.compile

      relu, patched, mask = plan.run(x: x)
      Conformance.assert_close(values.map { |v| v.positive? ? v : 0.0 }, relu.to_a)
      Conformance.assert_close(values.map { |v| v.zero? ? v + 1.0 : v * 10.0 }, patched.to_a, 1e-4)
      Conformance.assert_equal_arrays(values.map { |v| v.positive? ? 1 : 0 }, mask.to_a)
    end

    # Casting truncates toward zero (like a C cast). floor would be off by one
    # on the negative side, so negative values are always included.
    register(:graph_cast_truncates_toward_zero) do
      values = Conformance.f32([-2.75, -1.5, -0.5, 0.0, 0.5, 1.5, 2.75])
      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [values.length])
        g.output(t.cast(:int32))
        g.output(t.cast(:int32).cast(:float32))
      end.compile

      ints, back = plan.run(x: Tilework[values])
      Conformance.assert_equal_arrays(values.map(&:to_i), ints.to_a)
      Conformance.assert_close(values.map { |v| v.to_i.to_f }, back.to_a)
    end

    register(:graph_iota_writes_the_index_along_its_axis) do
      plan = Tilework::Graph.build do |g|
        g.output(g.iota([2, 3], axis: 0))
        g.output(g.iota([2, 3], axis: 1))
        g.output(g.iota([2, 2, 2], axis: 1, dtype: :float32))
      end.compile

      rows, cols, middle = plan.run
      Conformance.assert_equal_arrays([0, 0, 0, 1, 1, 1], rows.to_a)
      Conformance.assert_equal_arrays([0, 1, 2, 0, 1, 2], cols.to_a)
      Conformance.assert_close([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], middle.to_a)
    end

    # reshape is a view, so the source buffer must live until the view's readers
    # finish. Miscounting the lifetime returns memory still being read to the
    # pool and the next writer — which shows when the same plan reruns, so run
    # three times and expect the same values each time.
    register(:graph_reshape_is_a_view_whose_source_outlives_its_readers) do
      x = Conformance.nd([2, 2], [1.0, 2.0, 3.0, 4.0])
      y = Tilework[Conformance.f32([10.0, 20.0, 30.0, 40.0])]
      plan = Tilework::Graph.build do |g|
        a = g.input(:x, [2, 2]) * 2.0
        g.output((a.reshape(4) + g.param(y)) * 0.5)
        g.output(a.reshape(1, 4))
      end.compile

      3.times do
        summed, viewed = plan.run(x: x)
        Conformance.assert_close([6.0, 12.0, 18.0, 24.0], summed.to_a)
        Conformance.assert_close([2.0, 4.0, 6.0, 8.0], viewed.to_a)
        Conformance.assert(viewed.shape == [1, 4], "reshape should be [1, 4], got #{viewed.shape.inspect}")
      end
    end

    # Axis reductions. One thread folds in order along the axis, a different
    # order from the tree-folded whole-buffer reduction — so the reference is a
    # host fold in the same order, and the tolerance covers the accumulator gap
    # (float on the dialect side, Ruby double on :cpu).
    register(:graph_axis_reductions_match_a_host_computation) do
      batch = 2
      rows = 3
      cols = 4
      values = Array.new(batch * rows * cols) { |i| ((i % 11) - 5) * 0.375 }
      x = Conformance.nd([batch, rows, cols], values)
      flat = Conformance.f32(values)

      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [batch, rows, cols])
        g.output(t.sum(axis: 2))
        g.output(t.max(axis: 1))
        g.output(t.mean(axis: 0))
        g.output(t.argmax(axis: 2))
      end.compile
      sums, maxes, means, positions = plan.run(x: x)

      # The host-side answer, folding as [outer, axis, inner].
      fold = lambda do |axis, &combine|
        axis_len = [batch, rows, cols][axis]
        inner = [batch, rows, cols].drop(axis + 1).inject(1, :*)
        outer = [batch, rows, cols].take(axis).inject(1, :*)
        (0...(outer * inner)).map do |i|
          base = ((i / inner) * axis_len * inner) + (i % inner)
          combine.call((0...axis_len).map { |t| flat[base + (t * inner)] })
        end
      end

      Conformance.assert_close(fold.call(2) { |v| v.sum }, sums.to_a, 1e-4)
      Conformance.assert_close(fold.call(1) { |v| v.max }, maxes.to_a, 1e-6)
      Conformance.assert_close(fold.call(0) { |v| v.sum / v.length }, means.to_a, 1e-4)
      Conformance.assert_equal_arrays(fold.call(2) { |v| v.index(v.max) }, positions.to_a)
      Conformance.assert(sums.shape == [batch, rows],
                         "sum(axis: 2) should be #{[batch, rows].inspect}, got #{sums.shape.inspect}")
    end

    # Row-wise fusion agrees with the unfused path. The fold order and the
    # write-back rounding match the axis-reduction kernel, so the tolerance is
    # only for the same reason as elementwise fusion (FMA contraction).
    register(:graph_row_wise_fusion_matches_the_unfused_path) do
      rows = 256
      cols = 8
      x = Conformance.nd([rows, cols], Array.new(rows * cols) { |i| ((i % 13) - 6) * 0.5 })
      build = lambda do
        plan = Tilework::Graph.build do |g|
          t = g.input(:x, [rows, cols])
          # softmax in primitives; two reductions in a row.
          e = (t - t.max(axis: -1, keepdims: true)).exp
          g.output(e / e.sum(axis: -1, keepdims: true))
          # The folded value itself is an output too (a reduction that is an output cannot join a cluster).
          g.output(t.mean(axis: -1, keepdims: true))
        end.compile
        [plan, plan.run(x: x).map(&:to_a)]
      end

      previous = Tilework::Graph.row_fusion
      begin
        Tilework::Graph.row_fusion = true
        fused_plan, fused = build.call
        Tilework::Graph.row_fusion = false
        plain_plan, plain = build.call
      ensure
        Tilework::Graph.row_fusion = previous
      end

      Conformance.assert(fused_plan.fusion_stats[:nodes] < plain_plan.fusion_stats[:nodes],
                         "row fusion should launch fewer kernels than the unfused path")
      fused.zip(plain).each { |a, b| Conformance.assert_close(b, a, 1e-6) }
    end

    register(:graph_permute_reorders_axes) do
      values = Array.new(24) { |i| i.to_f }
      x = Conformance.nd([2, 3, 4], values)
      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [2, 3, 4])
        g.output(t.permute(1, 0, 2))
        g.output(t.permute(2, 1, 0))
      end.compile

      swapped, reversed = plan.run(x: x)
      Conformance.assert(swapped.shape == [3, 2, 4], "permute(1, 0, 2) should be [3, 2, 4]")
      Conformance.assert(reversed.shape == [4, 3, 2], "permute(2, 1, 0) should be [4, 3, 2]")
      at = ->(i, j, k) { values[(i * 12) + (j * 4) + k] }
      Conformance.assert_close((0...3).flat_map { |j| (0...2).flat_map { |i| (0...4).map { |k| at.(i, j, k) } } },
                               swapped.to_a)
      Conformance.assert_close((0...4).flat_map { |k| (0...3).flat_map { |j| (0...2).map { |i| at.(i, j, k) } } },
                               reversed.to_a)
    end

    # Table lookup. Out-of-range indices are clamped to the ends — what an
    # out-of-range read does on a GPU is undefined, so clamping was chosen over
    # breaking silently.
    register(:graph_gather_reads_along_an_axis) do
      table = Conformance.nd([4, 2], [10.0, 11.0, 20.0, 21.0, 30.0, 31.0, 40.0, 41.0])
      ids = Buffer.new(:int32, 4, shape: [4])
      ids.write([2, 0, 3, 99])
      plan = Tilework::Graph.build do |g|
        g.output(g.gather(g.param(table), g.param(ids)))
      end.compile

      rows = plan.run
      Conformance.assert(rows.shape == [4, 2], "gather should be [4, 2], got #{rows.shape.inspect}")
      Conformance.assert_close([30.0, 31.0, 10.0, 11.0, 40.0, 41.0, 40.0, 41.0], rows.to_a)
    end

    # Border fill, with different before/after amounts per dim, covering inner and outer dims.
    register(:graph_pad_fills_the_border) do
      x = Conformance.nd([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      plan = Tilework::Graph.build do |g|
        g.output(g.pad(g.input(:x, [2, 3]), before: [1, 1], after: [0, 2], value: -1.0))
      end.compile

      padded = plan.run(x: x)
      Conformance.assert(padded.shape == [3, 6], "pad should be [3, 6], got #{padded.shape.inspect}")
      Conformance.assert_close([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                                -1.0, 1.0, 2.0, 3.0, -1.0, -1.0,
                                -1.0, 4.0, 5.0, 6.0, -1.0, -1.0], padded.to_a)
    end

    # Concatenating along an inner axis scatters pieces, so it takes a different path from the outermost dim.
    register(:graph_concat_joins_along_an_inner_axis) do
      a = Conformance.nd([2, 2], [1.0, 2.0, 3.0, 4.0])
      b = Conformance.nd([2, 1], [5.0, 6.0])
      plan = Tilework::Graph.build do |g|
        g.output(g.concat(g.input(:a, [2, 2]), g.param(b), axis: 1))
      end.compile

      Conformance.assert_close([1.0, 2.0, 5.0, 3.0, 4.0, 6.0], plan.run(a: a).to_a)
    end

    # An outermost-dim slice is a view (no kernel). Returning the source to the
    # pool early corrupts values, so like reshape it is rerun to check.
    register(:graph_slice_is_a_view_of_the_outermost_dimension) do
      x = Conformance.nd([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      plan = Tilework::Graph.build do |g|
        rows = g.input(:x, [3, 2]) * 2.0
        g.output(rows.slice(1..2) + 1.0)
        g.output(rows.slice(0))
      end.compile

      3.times do
        tail, head = plan.run(x: x)
        Conformance.assert(tail.shape == [2, 2], "slice(1..2) should be [2, 2], got #{tail.shape.inspect}")
        Conformance.assert_close([7.0, 9.0, 11.0, 13.0], tail.to_a)
        Conformance.assert_close([2.0, 4.0], head.to_a)
      end
    end

    # Convolution. Delegated to cuDNN / MPSGraph when present, otherwise (:cpu)
    # expanded into im2col + matmul at record time. It is compared with a
    # direct convolution on the host, so both paths face the same check.
    register(:graph_conv2d_matches_a_direct_convolution) do
      n = 2
      cin = 2
      h = 5
      w = 4
      cout = 3
      kh = 3
      kw = 2
      pad = [1, 0]
      stride = [2, 1]
      x = Conformance.nd([n, cin, h, w], Array.new(n * cin * h * w) { |i| ((i % 11) - 5) * 0.25 })
      weight = Conformance.nd([cout, cin, kh, kw], Array.new(cout * cin * kh * kw) { |i| ((i % 7) - 3) * 0.125 })

      plan = Tilework::Graph.build do |g|
        g.output(Conv.forward(g.input(:x, [n, cin, h, w]), g.param(weight), pad: pad, stride: stride))
      end.compile
      out = plan.run(x: x)

      oh = Conv.output_size(h, kh, pad[0], stride[0])
      ow = Conv.output_size(w, kw, pad[1], stride[1])
      Conformance.assert(out.shape == [n, cout, oh, ow],
                         "conv2d should be #{[n, cout, oh, ow].inspect}, got #{out.shape.inspect}")

      xs = x.to_a
      ws = weight.to_a
      expected = (0...n).flat_map do |b|
        (0...cout).flat_map do |co|
          (0...oh).flat_map do |y|
            (0...ow).map do |t|
              (0...cin).sum do |ci|
                (0...kh).sum do |ky|
                  (0...kw).sum do |kx|
                    iy = (y * stride[0]) + ky - pad[0]
                    ix = (t * stride[1]) + kx - pad[1]
                    next 0.0 if iy.negative? || iy >= h || ix.negative? || ix >= w

                    xs[(((b * cin) + ci) * h * w) + (iy * w) + ix] *
                      ws[(((co * cin) + ci) * kh * kw) + (ky * kw) + kx]
                  end
                end
              end
            end
          end
        end
      end
      Conformance.assert_close(expected, out.to_a, 1e-4)
    end

    # The same check with dilation; the only difference is reading taps with
    # (dilation - 1) gaps — Conv.output_size computes the effective kernel width.
    register(:graph_conv2d_dilation_matches_a_direct_convolution) do
      n = 1
      cin = 2
      h = 8
      w = 7
      cout = 3
      kh = 3
      kw = 3
      pad = [2, 1]
      stride = [1, 2]
      dilation = [2, 3]
      x = Conformance.nd([n, cin, h, w], Array.new(n * cin * h * w) { |i| ((i % 11) - 5) * 0.25 })
      weight = Conformance.nd([cout, cin, kh, kw], Array.new(cout * cin * kh * kw) { |i| ((i % 7) - 3) * 0.125 })

      plan = Tilework::Graph.build do |g|
        g.output(Conv.forward(g.input(:x, [n, cin, h, w]), g.param(weight),
                              pad: pad, stride: stride, dilation: dilation))
      end.compile
      out = plan.run(x: x)

      oh = Conv.output_size(h, kh, pad[0], stride[0], dilation[0])
      ow = Conv.output_size(w, kw, pad[1], stride[1], dilation[1])
      Conformance.assert(out.shape == [n, cout, oh, ow],
                         "dilated conv2d should be #{[n, cout, oh, ow].inspect}, got #{out.shape.inspect}")

      xs = x.to_a
      ws = weight.to_a
      expected = (0...n).flat_map do |b|
        (0...cout).flat_map do |co|
          (0...oh).flat_map do |y|
            (0...ow).map do |t|
              (0...cin).sum do |ci|
                (0...kh).sum do |ky|
                  (0...kw).sum do |kx|
                    iy = (y * stride[0]) + (ky * dilation[0]) - pad[0]
                    ix = (t * stride[1]) + (kx * dilation[1]) - pad[1]
                    next 0.0 if iy.negative? || iy >= h || ix.negative? || ix >= w

                    xs[(((b * cin) + ci) * h * w) + (iy * w) + ix] *
                      ws[(((co * cin) + ci) * kh * kw) + (ky * kw) + kx]
                  end
                end
              end
            end
          end
        end
      end
      Conformance.assert_close(expected, out.to_a, 1e-4)
    end

    # groups > 1 (including depthwise). Each output channel sees only its own
    # group's input channel range, and the reference sums only that range.
    register(:graph_conv2d_groups_matches_a_direct_convolution) do
      n = 1
      cin = 4
      h = 6
      w = 5
      cout = 4
      kh = 3
      kw = 3
      pad = [1, 1]
      stride = [1, 1]
      groups = 4 # depthwise: one input channel each
      cin_per_group = cin / groups
      cout_per_group = cout / groups
      x = Conformance.nd([n, cin, h, w], Array.new(n * cin * h * w) { |i| ((i % 11) - 5) * 0.25 })
      weight = Conformance.nd([cout, cin_per_group, kh, kw],
                              Array.new(cout * cin_per_group * kh * kw) { |i| ((i % 7) - 3) * 0.125 })

      plan = Tilework::Graph.build do |g|
        g.output(Conv.forward(g.input(:x, [n, cin, h, w]), g.param(weight),
                              pad: pad, stride: stride, groups: groups))
      end.compile
      out = plan.run(x: x)

      oh = Conv.output_size(h, kh, pad[0], stride[0])
      ow = Conv.output_size(w, kw, pad[1], stride[1])
      Conformance.assert(out.shape == [n, cout, oh, ow],
                         "grouped conv2d should be #{[n, cout, oh, ow].inspect}, got #{out.shape.inspect}")

      xs = x.to_a
      ws = weight.to_a
      expected = (0...n).flat_map do |b|
        (0...cout).flat_map do |co|
          gi = co / cout_per_group
          (0...oh).flat_map do |y|
            (0...ow).map do |t|
              (0...cin_per_group).sum do |ci|
                (0...kh).sum do |ky|
                  (0...kw).sum do |kx|
                    iy = (y * stride[0]) + ky - pad[0]
                    ix = (t * stride[1]) + kx - pad[1]
                    next 0.0 if iy.negative? || iy >= h || ix.negative? || ix >= w

                    xs[(((b * cin) + (gi * cin_per_group) + ci) * h * w) + (iy * w) + ix] *
                      ws[(((co * cin_per_group) + ci) * kh * kw) + (ky * kw) + kx]
                  end
                end
              end
            end
          end
        end
      end
      Conformance.assert_close(expected, out.to_a, 1e-4)
    end

    # General groups (not depthwise: groups == 2 with several channels per
    # group) are covered by the same form of check.
    register(:graph_conv2d_general_groups_matches_a_direct_convolution) do
      n = 1
      cin = 4
      h = 6
      w = 6
      cout = 6
      kh = 3
      kw = 3
      pad = [1, 1]
      stride = [1, 1]
      groups = 2
      cin_per_group = cin / groups
      cout_per_group = cout / groups
      x = Conformance.nd([n, cin, h, w], Array.new(n * cin * h * w) { |i| ((i % 13) - 6) * 0.2 })
      weight = Conformance.nd([cout, cin_per_group, kh, kw],
                              Array.new(cout * cin_per_group * kh * kw) { |i| ((i % 9) - 4) * 0.1 })

      plan = Tilework::Graph.build do |g|
        g.output(Conv.forward(g.input(:x, [n, cin, h, w]), g.param(weight),
                              pad: pad, stride: stride, groups: groups))
      end.compile
      out = plan.run(x: x)

      oh = Conv.output_size(h, kh, pad[0], stride[0])
      ow = Conv.output_size(w, kw, pad[1], stride[1])

      xs = x.to_a
      ws = weight.to_a
      expected = (0...n).flat_map do |b|
        (0...cout).flat_map do |co|
          gi = co / cout_per_group
          (0...oh).flat_map do |y|
            (0...ow).map do |t|
              (0...cin_per_group).sum do |ci|
                (0...kh).sum do |ky|
                  (0...kw).sum do |kx|
                    iy = (y * stride[0]) + ky - pad[0]
                    ix = (t * stride[1]) + kx - pad[1]
                    next 0.0 if iy.negative? || iy >= h || ix.negative? || ix >= w

                    xs[(((b * cin) + (gi * cin_per_group) + ci) * h * w) + (iy * w) + ix] *
                      ws[(((co * cin_per_group) + ci) * kh * kw) + (ky * kw) + kx]
                  end
                end
              end
            end
          end
        end
      end
      Conformance.assert_close(expected, out.to_a, 1e-4)
    end

    # High-level ops agree with their decompositions (Tier 1 in docs/graph-ir.md).
    #
    # Bit identity is not available — the decomposition reduces in a different
    # order and accumulates in a different type than the hand-written kernel.
    # Measured on :cpu the gap is about 1e-7; the tolerance is 1e-5 (for
    # backend exp / tanh differences and FMA contraction). A wrong axis, a
    # dropped gamma and a missing square are all caught. Differences like moving
    # eps (1e-6) outside the square root fall below that, so **those are not checked**.
    register(:graph_high_level_ops_match_their_decomposition) do
      rows = 3
      cols = 8
      values = Array.new(rows * cols) { |i| ((i % 13) - 6) * 0.5 }
      x = Conformance.nd([rows, cols], values)
      gamma = Tilework[Conformance.f32(Array.new(cols) { |i| 0.5 + (i * 0.25) })]
      beta = Tilework[Conformance.f32(Array.new(cols) { |i| (i % 3) - 1.0 })]
      kv = Conformance.nd([5, cols], Array.new(5 * cols) { |i| ((i % 7) - 3) * 0.25 })
      target = Conformance.nd([rows, cols], Array.new(rows * cols) { |i| ((i % 5) - 2) * 0.5 })
      labels = Buffer.new(:int32, rows, shape: [rows])
      labels.write(Array.new(rows) { |i| (i * 3) % cols })

      build = lambda do
        plan = Tilework::Graph.build do |g|
          t = g.input(:x, [rows, cols])
          g.output(NN.softmax(t))
          g.output(NN.layernorm(t, gamma: g.param(gamma), beta: g.param(beta)))
          g.output(NN.layernorm(t))                       # the path without affine transform
          g.output(NN.rmsnorm(t, gamma: g.param(gamma)))
          g.output(NN.relu(t))
          g.output(NN.sigmoid(t))
          g.output(NN.silu(t))
          g.output(NN.gelu(t))
          g.output(NN.transpose(t))
          # Needs head splitting, axis permutation and batched matmul all at once.
          g.output(NN.multi_head_attention(t, g.param(kv), g.param(kv), heads: 2))
          # A loss. It ends in a reduction, so a decomposition with the wrong axis
          # or divisor shows a difference once folded into one number.
          g.output(NN.mse(t, g.param(target)))
          g.output(NN.cross_entropy(t, g.param(labels)))
          g.output(NN.cross_entropy(t, g.param(labels), reduction: :none))
        end.compile
        [plan, plan.run(x: x).map(&:to_a)]
      end

      previous = Tilework::Graph.lower_high_level
      begin
        Tilework::Graph.lower_high_level = false
        fused_plan, fused = build.call
        Tilework::Graph.lower_high_level = true
        lowered_plan, lowered = build.call
      ensure
        Tilework::Graph.lower_high_level = previous
      end

      Conformance.assert(fused_plan.nodes.count { |n| n.op == :nn } == 13,
                         "the default path should record 13 :nn nodes")
      Conformance.assert(lowered_plan.nodes.none? { |n| n.op == :nn },
                         "the lowered path should have no :nn node left")
      names = %w[softmax layernorm layernorm_plain rmsnorm relu sigmoid silu gelu
                 transpose multi_head_attention mse cross_entropy cross_entropy_rows]
      names.each_with_index do |name, i|
        begin
          Conformance.assert_close(fused[i], lowered[i], 1e-5)
        rescue RuntimeError => e
          raise "#{name}: the decomposition disagrees with the kernel (#{e.message})"
        end
      end
    end

    # softmax written in primitives alone agrees with the hand-written fused
    # kernel (NN.softmax) — the smallest evidence of Tier 1 in docs/graph-ir.md.
    register(:graph_softmax_in_primitives_matches_the_nn_kernel) do
      rows = 4
      cols = 8
      x = Conformance.nd([rows, cols], Array.new(rows * cols) { |i| ((i % 13) - 6) * 0.5 })
      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [rows, cols])
        e = (t - t.max(axis: -1, keepdims: true)).exp
        g.output(e / e.sum(axis: -1, keepdims: true))
      end.compile

      Conformance.assert_close(NN.softmax(x).to_a, plan.run(x: x).to_a, 1e-5)
    end

    # Concatenation just lays pieces along the outermost dim. If a piece were not
    # contiguous in the output its contents would swap, so values themselves are checked.
    register(:graph_concat_joins_along_the_outermost_dimension) do
      a = Tilework[Conformance.f32([1.0, 2.0, 3.0, 4.0])].reshape(2, 2)
      b = Tilework[Conformance.f32([5.0, 6.0])].reshape(1, 2)
      plan = Tilework::Graph.build do |g|
        g.output(g.concat(g.input(:a, [2, 2]), g.param(b)))
      end.compile

      Conformance.assert_equal_arrays([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], plan.run(a: a).to_a)
    end

    # Row-wise reductions look only at the last dim, so 3D and up give the same
    # answer as the 2D with leading dims folded; the dims must also come back intact.
    register(:nn_row_wise_ops_treat_leading_dims_as_rows) do
      values = Array.new(12) { |i| ((i % 4) - 1.5) * 0.75 }
      cube = Conformance.nd([3, 2, 2], values)
      flat = Conformance.nn_matrix(6, 2, Conformance.f32(values))

      out = NN.softmax(cube)
      Conformance.assert(out.shape == [3, 2, 2], "softmax should keep #{cube.shape.inspect}, got #{out.shape.inspect}")
      Conformance.assert_equal_arrays(NN.softmax(flat).to_a, out.to_a)
      Conformance.assert_equal_arrays(NN.rmsnorm(flat).to_a, NN.rmsnorm(cube).to_a)
    end

    # Fused elementwise ops agree with unfused execution.
    #
    # Fusion wraps each expression-tree node in a cast to its output type,
    # reproducing each node's write-back exactly. So **the arithmetic is
    # identical**, and :cpu agrees bit for bit (graph_cpu_test.rb checks that).
    #
    # The tolerance here exists for one reason only: a shader compiler may
    # contract `a * b + c` within one expression into a single FMA, which
    # differs in the low bits from two unfused kernels (the contracted one is
    # actually more accurate). There is no portable way to forbid contraction,
    # so the tolerance passes identical arithmetic but fails a wrong order or
    # truncation.
    #
    # int32 and comparisons are mixed in because integer truncation is easy to
    # lose in fusion (unnoticeable when looking at floats only).
    register(:graph_fused_elementwise_matches_the_unfused_path) do
      values = Conformance.f32(Array.new(256) { |i| ((i % 17) - 8) * 0.375 })
      bias = Tilework[Conformance.f32(Array.new(16) { |i| ((i % 5) - 2) * 0.5 })]
      build = lambda do
        plan = Tilework::Graph.build do |g|
          x = g.input(:x, [256])
          v = x
          4.times { v = ((v * 1.5) + 0.25).tanh.abs }
          # Mix a round trip through integers with a 0/1 mask
          ints = g.input(:i, [256], dtype: :int32)
          # cast and where are expression-tree nodes too, so the same property
          # (fusion does not change values) is pinned for them here.
          v = g.where(v > 0.5, v * 2.0, v.cast(:int32).cast(:float32))
          v += ((ints * 3) / 2).eq(4)
          # Fused, a broadcast folds into **an index**; unfused, it writes out the
          # stretched values. The values read are the same, so these must agree too.
          v = v.reshape(16, 16) + g.param(bias)
          g.output(v.reshape(256))
        end.compile
        plan.run(x: Tilework[values], i: Tilework::Buffer.from(Array.new(256) { |i| i % 5 }))
      end

      previous = Tilework::Graph.fusion
      begin
        Tilework::Graph.fusion = true
        fused = build.call.to_a
        Tilework::Graph.fusion = false
        plain = build.call.to_a
      ensure
        Tilework::Graph.fusion = previous
      end
      Conformance.assert_close(plain, fused, 1e-6)
    end

    # NN's fused kernels run the same function in a graph (no graph-specific
    # implementation), so this requires bit identity, not closeness.
    register(:graph_nn_matches_the_eager_path) do
      x = Tilework[Conformance.f32(Array.new(24) { |i| ((i % 7) - 3) * 0.5 })].reshape(4, 6)
      gamma = Tilework[Conformance.f32(Array.new(6) { |i| 1.0 + (i * 0.1) })]

      plan = Tilework::Graph.build do |g|
        t = g.input(:x, [4, 6])
        # gamma can be passed as a Buffer (lifted like g.param).
        h = Tilework::NN.rmsnorm(t, gamma: gamma)
        g.output(Tilework::NN.softmax(Tilework::NN.gelu(h) + 1.0))
        g.output(Tilework::NN.transpose(h))
      end.compile

      probs, transposed = plan.run(x: x)
      expected = Tilework::NN.rmsnorm(x, gamma: gamma)
      Conformance.assert_equal_arrays(Tilework::NN.softmax(Tilework::NN.gelu(expected) + 1.0).to_a, probs.to_a)
      Conformance.assert_equal_arrays(Tilework::NN.transpose(expected).to_a, transposed.to_a)
    end

    # With gemm_library, attention is delegated to a vendor library, so inside a
    # graph it waits on both sides (like matmul). Whether those waits are right
    # is checked by agreeing with eager while running asynchronously.
    register(:graph_attention_matches_the_eager_path) do
      heads = 2
      n = 3
      d = 4
      q = Conformance.nn_matrix(n, d, Conformance.f32(Array.new(n * d) { |i| ((i % 5) - 2) * 0.25 }))
      k = Conformance.nn_matrix(n, d, Conformance.f32(Array.new(n * d) { |i| ((i % 3) - 1) * 0.5 }))
      v = Conformance.nn_matrix(n, d, Conformance.f32(Array.new(n * d) { |i| (i % 4) * 0.125 }))

      plan = Tilework::Graph.build do |g|
        qt = g.input(:q, [n, d])
        g.output(Tilework::NN.multi_head_attention(qt, g.param(k), g.param(v), heads: heads))
      end.compile

      expected = Tilework::NN.multi_head_attention(q, k, v, heads: heads).to_a
      Conformance.assert_equal_arrays(expected, plan.run(q: q).to_a)
    end

    # RNG inside a graph **depends only on the seed**. The key comes from (plan
    # seed, the node's static index), so it depends neither on execution order
    # nor on run count — the basis for not making RNG mutable state; breaking it
    # brings back bugs where async execution changes the sequence.
    register(:graph_random_is_pure_in_the_plan_seed) do
      plan = Tilework::Graph.build do |g|
        g.output(g.random([64]))
        g.output(g.randn([64]))
      end.compile

      first = plan.run(seed: 1234).map(&:to_a)
      again = plan.run(seed: 1234).map(&:to_a)
      advanced = plan.run(seed: plan.next_seed(1234)).map(&:to_a)

      Conformance.assert_equal_arrays(first[0], again[0])
      Conformance.assert_equal_arrays(first[1], again[1])
      Conformance.assert(first[0] != first[1], "two draws in one plan must not share a sequence")
      Conformance.assert(first[0] != advanced[0], "advancing the seed must change the sequence")
    end

    # Placement (writing a write_into value straight into the state's view) is
    # bit-identical to producing the value and copying it. A mismatch means the
    # write position or the write/read order is broken.
    register(:graph_placed_writes_match_the_copying_path) do
      weight = Tilework[Conformance.f32([2.0, 0.0, 0.0, 3.0])].reshape(2, 2)
      run = lambda do
        cache = Tilework[Conformance.f32(Array.new(8, 0.0))].reshape(4, 2)
        plan = Tilework::Graph.build do |g|
          kv0 = g.state(cache, name: :kv)
          k = g.input(:k, [1, 2])
          kv1 = g.write_into(kv0, at: g.index(:pos), value: k.matmul(g.param(weight)))
          g.output(g.read(kv1) * 1.0)
        end.compile
        3.times { |step| plan.run(k: Tilework[Conformance.f32([step + 1.0, 1.0])].reshape(1, 2), pos: step) }
        cache.to_a
      end

      previous = Tilework::Graph.placement
      begin
        Tilework::Graph.placement = true
        placed = run.call
        Tilework::Graph.placement = false
        copied = run.call
      ensure
        Tilework::Graph.placement = previous
      end
      Conformance.assert_equal_arrays(copied, placed)
    end

    # Mutable-state ordering. Version edges give only RAW, so check numerically
    # that version N+1's write does not run while version N's readers still read.
    register(:graph_state_write_waits_for_the_readers_of_the_old_version) do
      cache = Tilework[Conformance.f32([1.0, 1.0, 2.0, 2.0])].reshape(2, 2)
      plan = Tilework::Graph.build do |g|
        s0 = g.state(cache, name: :s)
        stale = g.read(s0, 0)
        g.write_into(s0, at: 0, value: g.param(Tilework[Conformance.f32([9.0, 9.0])].reshape(1, 2)))
        g.output(stale * 1.0)
      end.compile
      Conformance.assert_close([1.0, 1.0], plan.run.to_a)
      Conformance.assert_close([9.0, 9.0, 2.0, 2.0], cache.to_a)
    end

    # A write position decided at run time (a KV-cache append), advanced by
    # rerunning the same plan.
    register(:graph_state_appends_across_runs) do
      cache = Tilework::Buffer.new(:float32, 8, shape: [4, 2])
      cache.write([0.0] * 8)
      plan = Tilework::Graph.build do |g|
        kv0 = g.state(cache, name: :kv)
        k = g.input(:k, [1, 2])
        g.output(g.read(g.write_into(kv0, at: g.index(:pos), value: k)) * 1.0)
      end.compile
      3.times do |step|
        plan.run(k: Tilework[Conformance.f32([step + 1.0, (step + 1) * 10.0])].reshape(1, 2), pos: step)
      end
      Conformance.assert_close([1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 0.0, 0.0], cache.to_a)
    end

    # Two writes to disjoint regions need not serialize (rejoined with g.merge).
    register(:graph_state_disjoint_writes_fill_the_buffer) do
      out = Tilework::Buffer.new(:float32, 8, shape: [2, 4])
      out.write([0.0] * 8)
      plan = Tilework::Graph.build do |g|
        s0 = g.state(out, name: :out)
        left = g.write_into(s0, at: s0.axis(1, 0...2), value: g.input(:l, [2, 2]))
        right = g.write_into(s0, at: s0.axis(1, 2...4), value: g.input(:r, [2, 2]))
        g.output(g.read(g.merge(left, right)))
      end.compile
      l = Tilework[Conformance.f32([1.0, 2.0, 5.0, 6.0])].reshape(2, 2)
      r = Tilework[Conformance.f32([3.0, 4.0, 7.0, 8.0])].reshape(2, 2)
      Conformance.assert_close([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], plan.run(l: l, r: r).to_a)
    end

    # strict_state = :copy materializes reads, making the graph a pure DAG. The
    # default view path producing the same values verifies the mutable-state
    # machinery itself (an oracle inside the graph layer).
    register(:graph_state_view_reads_agree_with_materialized_reads) do
      build = lambda do
        cache = Tilework[Conformance.f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])].reshape(3, 2)
        plan = Tilework::Graph.build do |g|
          s0 = g.state(cache, name: :kv)
          head = g.read(s0, 0..1)
          s1 = g.write_into(s0, at: 2, value: g.param(Tilework[Conformance.f32([7.0, 8.0])].reshape(1, 2)))
          g.output((head * 2.0) + g.read(s1, 1..2))
        end.compile
        plan.run.to_a
      end

      previous = Tilework::Graph.strict_state
      begin
        Tilework::Graph.strict_state = :view
        view = build.call
        Tilework::Graph.strict_state = :copy
        copied = build.call
      ensure
        Tilework::Graph.strict_state = previous
      end
      Conformance.assert_close(copied, view)
    end
    # --- Automatic offload (#17 Phase 4 / docs/graph-offload.md) -------------
    #
    # Paging must not change values, so it is held to bit identity at any
    # budget. Without a staging ABI Offload.transport falls back to :posix, so
    # the policy itself is checkable on :cpu.
    register(:graph_offload_matches_the_resident_path) do
      # Bind **non-view** Buffers: reshape returns a view, and views cannot be
      # paged because restore! allocates a new handle (they are just counted as
      # pinned, but then this check would page nothing).
      weights = Array.new(4) { Conformance.square(4) }
      x = Conformance.square(4)
      build = lambda do |vram|
        Tilework::Graph.build do |g|
          value = g.input(:x, [4, 4])
          weights.each { |w| value = value.matmul(g.param(w)) }
          g.output(value)
        end.compile(vram: vram)
      end

      resident = build.call(nil).run(x: x).to_a
      # Squeeze the budget to "two weights + reserve". One node needs one weight
      # at a time, so it passes, but four do not fit, so paging always happens.
      probe = build.call(1 << 30).residency
      slot = probe.weights / weights.length
      paged = build.call(probe.reserved + (2 * slot))
      begin
        Conformance.assert(paged.residency.floor.positive?, "the budget must force paging")
        Conformance.assert_equal_arrays(resident, paged.run(x: x).to_a)
        Conformance.assert(paged.offload_stats[:fetches].positive?, "no weight was ever paged in")
      ensure
        paged.free_offload!
      end
    end

    # Rerunning the same plan does not change values. Eviction is buffer_free,
    # so releasing VRAM a not-yet-executed kernel reads breaks things as
    # "occasionally different values" (the same hazard that makes Pool return
    # rather than free). One run cannot hit it, so run three times.
    register(:graph_offload_is_stable_across_runs) do
      weights = Array.new(3) { Conformance.square(4) }
      x = Conformance.square(4)
      build = lambda do |vram|
        Tilework::Graph.build do |g|
          value = g.input(:x, [4, 4])
          weights.each { |w| value = (value.matmul(g.param(w)) * 1.5) }
          g.output(value)
        end.compile(vram: vram)
      end

      want = build.call(nil).run(x: x).to_a
      probe = build.call(1 << 30).residency
      plan = build.call(probe.reserved + (probe.weights / weights.length))
      begin
        3.times { |i| Conformance.assert_equal_arrays(want, plan.run(x: x).to_a) && i }
      ensure
        plan.free_offload!
      end
    end

    # --- Backward (#22) ------------------------------------------------------

    # Gradients are correct, checked against **finite differences** — backward
    # easily returns plausible but wrong values, and forward agreement
    # guarantees nothing.
    #
    # It checks **directional derivatives**, not per-element values. float32
    # central differences agree only to about 1e-3 per element (cancellation),
    # so a per-element tolerance would constrain nothing. <grad, v> against
    # (f(x+hv) - f(x-hv)) / 2h for a direction v averages the error down to
    # about 1e-4. The tolerance is 3e-2 (measured on :cpu: 1e-5 to 1e-4).
    #
    # v is fixed (no random draw) — a failure that cannot be reproduced is useless.
    register(:graph_gradients_match_finite_differences) do
      rows, cols, out = 3, 4, 2
      # Fixed values that **stay away from 0 and where's thresholds** — abs and
      # where have kinks where the derivative jumps, and differences straddling
      # one disagree with the analytic value (a limit of finite differences,
      # not a vjp error).
      x = Array.new(rows * cols) { |i| Math.sin((i + 0.6) * 1.7) * 0.8 }
      w = Array.new(cols * out) { |i| Math.cos(i * 2.3) * 0.6 }
      dx = Array.new(rows * cols) { |i| Math.cos(i * 0.9) * 0.5 }
      dw = Array.new(cols * out) { |i| Math.sin(i * 1.3) * 0.5 }
      # gamma (the norms' affine scale) is differentiated too. Hand-written vjps
      # use a separate expression for it, so checking x alone tests only half.
      gm = Array.new(cols) { |i| 0.5 + (i * 0.25) }
      dg = Array.new(cols) { |i| Math.cos((i + 1) * 1.1) * 0.5 }
      labels = Buffer.new(:int32, rows, shape: [rows])
      labels.write(Array.new(rows) { |i| i % out })

      # One plan per expression. Summing everything into one loss lets one wrong
      # vjp dilute into the other terms and fall under the tolerance.
      cases = {
        arithmetic: ->(_g, xt, wt, _gt) {  (((xt * xt) / (xt + 3.0)) - (2.0 / (xt + 3.0)) - (xt * 0.5)).sum + (wt * wt).sum },
        unary: ->(_g, xt, _wt, _gt) {  ((((xt * xt) + 1.5).sqrt.log + xt.sin + xt.cos.exp + xt.tanh).mean) },
        abs_where: ->(g, xt, _wt, _gt) {  (xt.abs + g.where(xt > 0.5, xt * 2.0, xt * 0.25)).sum },
        matmul: ->(_g, xt, wt, _gt) {  (xt.matmul(wt) * 1.5).sum },
        broadcast: ->(_g, xt, wt, _gt) {  (xt + wt.permute(1, 0).slice(0..0).reshape(cols)).sum },
        rowwise: ->(_g, xt, _wt, gt) { NN.silu(Tilework::NN.rmsnorm(xt, gamma: gt) * 1.5).sum },
        # layernorm's rows always sum to 0, so summing directly makes the
        # derivative 0 too (checking nothing). Weight by x before folding.
        layernorm: ->(_g, xt, _wt, gt) { (NN.layernorm(xt, gamma: gt, beta: gt * 0.5) * xt).sum },
        softmax: ->(_g, xt, _wt, _gt) {  (NN.softmax(xt) * NN.softmax(xt)).sum },
        gelu: ->(_g, xt, _wt, _gt) {  NN.gelu(xt).sum },
        cross_entropy: ->(g, xt, wt, _gt) { NN.cross_entropy(xt.matmul(wt), g.param(labels), reduction: :sum) * 5.0 },
        # The reduction's scale differs between :mean and :sum, so run both.
        cross_entropy_mean: ->(g, xt, wt, _gt) { NN.cross_entropy(xt.matmul(wt), g.param(labels)) * 20.0 },
        mse: ->(_g, xt, _wt, _gt) {  NN.mse(xt, xt * 2.0) },
        max_reduce: ->(_g, xt, _wt, _gt) {  xt.max(axis: -1).sum },
        # Shape ops. permute needs 3D for an inverse-permutation mistake to show
        # (a 2D transpose is its own inverse, so a mix-up gives the same answer).
        # iota weights each position differently so ordering differences reach the value.
        layout: lambda { |g, xt, _wt, _gt|
          t = xt.reshape(rows, 2, 2).permute(1, 2, 0)
          weighted = (t * g.iota([2, 2, rows], axis: 2).cast(:float32)).sum
          pieces = g.concat(xt.permute(1, 0).slice(0..1) / 3.0, xt.permute(1, 0).slice(1..2))
          weighted + pieces.reshape(4 * rows).sum
        },
        pad_concat: lambda { |g, xt, _wt, _gt|
          padded = g.pad(xt, before: [1, 2], after: [2, 1], value: 0.0)
          # A linear expression would make the derivative the sum of the
          # direction vector (≈0), so square before folding.
          joined = g.concat(padded, padded, axis: 1)
          (joined * joined).sum
        },
      }

      # h is 2e-3: smaller suffers cancellation, larger truncation error, so for
      # float32 this is about the floor.
      step = 2e-3
      cases.each do |name, body|
        build = lambda do |with_grad|
          Tilework::Graph.build(differentiable: true) do |g|
            xt = g.input(:x, [rows, cols])
            wt = g.input(:w, [cols, out])
            gt = g.input(:g, [cols])
            loss = body.call(g, xt, wt, gt)
            g.output(loss)
            g.grad(loss, wrt: [xt, wt, gt]).each { |grad| g.output(grad) } if with_grad
          end.compile
        end

        plan = build.call(true)
        plain = build.call(false)
        bind = lambda do
          { x: Conformance.nd([rows, cols], x), w: Conformance.nd([cols, out], w), g: Conformance.nd([cols], gm) }
        end
        grads = plan.run(**bind.call).drop(1).map(&:to_a)

        [[:x, x, dx, [rows, cols], 0], [:w, w, dw, [cols, out], 1],
         [:g, gm, dg, [cols], 2]].each do |arg, base, dir, shape, index|
          shifted = lambda do |sign|
            bindings = bind.call
            bindings[arg] = Conformance.nd(shape, base.each_with_index.map { |v, i| v + (sign * step * dir[i]) })
            plain.run(**bindings).to_a[0]
          end
          numeric = (shifted.call(1.0) - shifted.call(-1.0)) / (2.0 * step)
          analytic = grads[index].each_with_index.sum { |v, i| v * dir[i] }
          # An expression with a near-zero derivative checks nothing, so first
          # check that it moves (expressions not using :w excepted).
          next if numeric.abs < 1e-6 && analytic.abs < 1e-6

          Conformance.assert(numeric.abs > 1e-2,
                             "#{name} d/d#{arg}: the directional derivative is #{numeric}, too small to check")
          scale = [numeric.abs, analytic.abs].max
          Conformance.assert((numeric - analytic).abs / scale < 3e-2,
                             "#{name} d/d#{arg}: finite difference #{numeric}, analytic #{analytic}")
        end
      end
    end



    # Checkpointing lowers the peak without changing gradients (#22 Step 5).
    #
    # The pool's allocated is the number of slots alive at once, so it stands in
    # for the peak. If it does not drop, the scheduler has hoisted the copies
    # and nothing is saved.
    register(:graph_checkpointing_keeps_the_gradients_and_lowers_the_peak) do
      n, d, layers = 8, 16, 4
      x = Conformance.nd([n, d], Array.new(n * d) { |i| Math.sin((i + 0.3) * 0.7) * 0.5 })
      weights = Array.new(layers) do |l|
        Conformance.nd([d, d], Array.new(d * d) { |i| Math.cos((i + l) * 0.31) * 0.25 })
      end

      build = lambda do |use_checkpoint|
        Tilework::Graph.build(differentiable: true) do |g|
          xt = g.input(:x, [n, d])
          h = xt
          weights.each do |w|
            wt = g.param(w)
            h = if use_checkpoint
                  g.checkpoint { NN.gelu(h.matmul(wt)) }
                else
                  NN.gelu(h.matmul(wt))
                end
          end
          loss = (h * h).sum
          g.output(loss)
          g.grad(loss, wrt: [xt]).each { |grad| g.output(grad) }
        end.compile
      end

      plain = build.call(false)
      checkpointed = build.call(true)
      want = plain.run(x: x).map(&:to_a)
      got = checkpointed.run(x: x).map(&:to_a)
      # Bit identity is not available. checkpoint changes the fusion structure,
      # and inside a fused expression a shader compiler may fold a*b+c into an
      # FMA (measured on :metal: 4.5e-7 relative to the array max).
      # graph_grad_cpu_test.rb checks bit identity on :cpu.
      want.each_with_index { |values, i| Conformance.assert_close_relative(values, got[i], 1e-5) }

      # Run once more until the pool stabilizes before comparing.
      plain.run(x: x)
      checkpointed.run(x: x)
      peak = plain.pool_stats[:allocated]
      lower = checkpointed.pool_stats[:allocated]
      Conformance.assert(lower < peak,
                         "checkpointing did not lower the peak (#{lower} slots against #{peak})")
    end

    # Optimizers apply the updates they promise (#22 Step 3).
    #
    # The MLP check below sees training progress, but that **passes even with a
    # slightly wrong formula** (the loss drops as long as the gradient points
    # the right way). Here one step runs and the update itself is compared with
    # the closed form.
    #
    # Adam's first step starts from m = v = 0, so after bias correction
    # m̂/√v̂ = g/|g| — **it moves by lr regardless of the gradient's magnitude**.
    # Dropping the correction breaks this, so mishandling t fails here.
    register(:graph_optimizers_apply_the_updates_they_promise) do
      values = Conformance.f32([0.5, -1.5, 2.0, -0.25])
      target = Conformance.f32([0.0, 0.0, 0.0, 0.0])
      lr = 0.1

      # loss = Σ(p - t)², so the gradient is 2(p - t) = 2p.
      want_grad = values.map { |v| 2.0 * v }

      sgd_param = Tilework[values.dup]
      sgd = Tilework::Graph::SGD.new({ w: sgd_param }, lr: lr)
      sgd_plan = Tilework::Graph.build(differentiable: true) do |g|
        p = sgd.attach(g)[:w]
        loss = ((p - g.param(Tilework[target])) * (p - g.param(Tilework[target]))).sum
        sgd.step(loss)
        g.output(loss)
      end.compile
      sgd_plan.run
      Conformance.assert_close(values.each_with_index.map { |v, i| v - (lr * want_grad[i]) },
                               sgd_param.to_a, 1e-5)

      adam_param = Tilework[values.dup]
      adam = Tilework::Graph::Adam.new({ w: adam_param }, lr: lr)
      adam_plan = Tilework::Graph.build(differentiable: true) do |g|
        p = adam.attach(g)[:w]
        loss = ((p - g.param(Tilework[target])) * (p - g.param(Tilework[target]))).sum
        adam.step(loss)
        g.output(loss)
      end.compile
      begin
        adam_plan.run
        # Only the sign matters on the first step (slightly less, by eps).
        want = values.each_with_index.map { |v, i| v - (lr * (want_grad[i].negative? ? -1.0 : 1.0)) }
        Conformance.assert_close(want, adam_param.to_a, 1e-4)
      ensure
        adam.free!
        sgd.free!
      end
    end

    # gather's gradient (scatter-add) agrees with the form written as a matmul
    # against a match matrix, and gives the same value every run. It scans the
    # indices per output element without atomics, so the fold order is fixed
    # and bit identity can be required.
    #
    # It also checks that duplicate indices **accumulate** (not overwrite).
    register(:graph_scatter_add_matches_a_matmul_reference) do
      rows, inner, count = 5, 3, 6
      table = Conformance.nd([rows, inner], Array.new(rows * inner) { |i| Math.sin(i * 1.4) })
      ids = Buffer.new(:int32, count, shape: [count])
      # Duplicates (2 three times), a row never drawn (1), and out-of-range (clamped to the ends).
      ids.write([0, 2, 2, 2, 4, 9])

      plan = Tilework::Graph.build(differentiable: true) do |g|
        t = g.input(:t, [rows, inner])
        picked = g.gather(t, g.param(ids))
        g.output(g.grad((picked * picked).sum, wrt: [t])[0])
      end.compile

      # Reference: build mask[j, i] = (ids[i] == j) and multiply dy by it.
      reference = Tilework::Graph.build(differentiable: true) do |g|
        t = g.input(:t, [rows, inner])
        picked = g.gather(t, g.param(ids))
        dy = 2.0 * picked
        index = g.param(ids).reshape(1, count)
        clamped = g.where(index > (rows - 1), rows - 1, g.where(index < 0, 0, index))
        mask = g.where(g.iota([rows, count], axis: 0).eq(clamped), 1.0, 0.0)
        g.output(mask.matmul(dy))
      end.compile

      got = plan.run(t: table).to_a
      Conformance.assert_equal_arrays(reference.run(t: table).to_a, got)
      2.times { Conformance.assert_equal_arrays(got, plan.run(t: table).to_a) }
    end

    # Convolution gradients. **The only gradient taking a different path per
    # backend**, so both are checked against finite differences — the delegated
    # backward kernels (:conv2d_backward_data / _filter) with cuDNN / MPSGraph,
    # otherwise the primitive vjps of the im2col decomposition (where the pad and
    # gather rows come in).
    register(:graph_convolution_gradients_match_finite_differences) do
      n, cin, h, w = 1, 2, 5, 5
      cout, kh, kw = 2, 3, 3
      x = Array.new(n * cin * h * w) { |i| Math.sin((i + 0.4) * 1.3) * 0.7 }
      k = Array.new(cout * cin * kh * kw) { |i| Math.cos((i + 0.2) * 0.9) * 0.4 }
      dx = Array.new(x.length) { |i| Math.cos(i * 1.7) * 0.5 }
      dk = Array.new(k.length) { |i| Math.sin(i * 2.1) * 0.5 }

      build = lambda do |with_grad|
        Tilework::Graph.build(differentiable: true) do |g|
          xt = g.input(:x, [n, cin, h, w])
          kt = g.input(:k, [cout, cin, kh, kw])
          y = Tilework::Conv.forward(xt, kt, pad: 1, stride: 1)
          loss = (y * y).sum
          g.output(loss)
          g.grad(loss, wrt: [xt, kt]).each { |grad| g.output(grad) } if with_grad
        end.compile
      end

      plan = build.call(true)
      plain = build.call(false)
      bind = lambda do
        { x: Conformance.nd([n, cin, h, w], x), k: Conformance.nd([cout, cin, kh, kw], k) }
      end
      grads = plan.run(**bind.call).drop(1).map(&:to_a)

      step = 2e-3
      [[:x, x, dx, [n, cin, h, w], 0], [:k, k, dk, [cout, cin, kh, kw], 1]].each do |arg, base, dir, shape, index|
        shifted = lambda do |sign|
          bindings = bind.call
          bindings[arg] = Conformance.nd(shape, base.each_with_index.map { |v, i| v + (sign * step * dir[i]) })
          plain.run(**bindings).to_a[0]
        end
        numeric = (shifted.call(1.0) - shifted.call(-1.0)) / (2.0 * step)
        analytic = grads[index].each_with_index.sum { |v, i| v * dir[i] }
        Conformance.assert(numeric.abs > 1e-2, "conv d/d#{arg}: derivative #{numeric} is too small to check")
        Conformance.assert((numeric - analytic).abs / [numeric.abs, analytic.abs].max < 3e-2,
                           "conv d/d#{arg}: finite difference #{numeric}, analytic #{analytic}")
      end
    end

    # The same finite-difference check with dilation.
    register(:graph_dilated_convolution_gradients_match_finite_differences) do
      n, cin, h, w = 1, 2, 6, 6
      cout, kh, kw = 2, 3, 3
      dilation = [2, 2]
      x = Array.new(n * cin * h * w) { |i| Math.sin((i + 0.4) * 1.3) * 0.7 }
      k = Array.new(cout * cin * kh * kw) { |i| Math.cos((i + 0.2) * 0.9) * 0.4 }
      dx = Array.new(x.length) { |i| Math.cos(i * 1.7) * 0.5 }
      dk = Array.new(k.length) { |i| Math.sin(i * 2.1) * 0.5 }

      build = lambda do |with_grad|
        Tilework::Graph.build(differentiable: true) do |g|
          xt = g.input(:x, [n, cin, h, w])
          kt = g.input(:k, [cout, cin, kh, kw])
          y = Tilework::Conv.forward(xt, kt, pad: 2, stride: 1, dilation: dilation)
          loss = (y * y).sum
          g.output(loss)
          g.grad(loss, wrt: [xt, kt]).each { |grad| g.output(grad) } if with_grad
        end.compile
      end

      plan = build.call(true)
      plain = build.call(false)
      bind = lambda do
        { x: Conformance.nd([n, cin, h, w], x), k: Conformance.nd([cout, cin, kh, kw], k) }
      end
      grads = plan.run(**bind.call).drop(1).map(&:to_a)

      step = 2e-3
      [[:x, x, dx, [n, cin, h, w], 0], [:k, k, dk, [cout, cin, kh, kw], 1]].each do |arg, base, dir, shape, index|
        shifted = lambda do |sign|
          bindings = bind.call
          bindings[arg] = Conformance.nd(shape, base.each_with_index.map { |v, i| v + (sign * step * dir[i]) })
          plain.run(**bindings).to_a[0]
        end
        numeric = (shifted.call(1.0) - shifted.call(-1.0)) / (2.0 * step)
        analytic = grads[index].each_with_index.sum { |v, i| v * dir[i] }
        Conformance.assert(numeric.abs > 1e-2, "dilated conv d/d#{arg}: derivative #{numeric} is too small to check")
        Conformance.assert((numeric - analytic).abs / [numeric.abs, analytic.abs].max < 3e-2,
                           "dilated conv d/d#{arg}: finite difference #{numeric}, analytic #{analytic}")
      end
    end

    # The same finite-difference check with groups (depthwise: groups == cin == cout).
    register(:graph_grouped_convolution_gradients_match_finite_differences) do
      n, cin, h, w = 1, 4, 5, 5
      cout, kh, kw = 4, 3, 3
      groups = 4
      cin_per_group = cin / groups
      x = Array.new(n * cin * h * w) { |i| Math.sin((i + 0.4) * 1.3) * 0.7 }
      k = Array.new(cout * cin_per_group * kh * kw) { |i| Math.cos((i + 0.2) * 0.9) * 0.4 }
      dx = Array.new(x.length) { |i| Math.cos(i * 1.7) * 0.5 }
      dk = Array.new(k.length) { |i| Math.sin(i * 2.1) * 0.5 }

      build = lambda do |with_grad|
        Tilework::Graph.build(differentiable: true) do |g|
          xt = g.input(:x, [n, cin, h, w])
          kt = g.input(:k, [cout, cin_per_group, kh, kw])
          y = Tilework::Conv.forward(xt, kt, pad: 1, stride: 1, groups: groups)
          loss = (y * y).sum
          g.output(loss)
          g.grad(loss, wrt: [xt, kt]).each { |grad| g.output(grad) } if with_grad
        end.compile
      end

      plan = build.call(true)
      plain = build.call(false)
      bind = lambda do
        { x: Conformance.nd([n, cin, h, w], x), k: Conformance.nd([cout, cin_per_group, kh, kw], k) }
      end
      grads = plan.run(**bind.call).drop(1).map(&:to_a)

      step = 2e-3
      [[:x, x, dx, [n, cin, h, w], 0],
       [:k, k, dk, [cout, cin_per_group, kh, kw], 1]].each do |arg, base, dir, shape, index|
        shifted = lambda do |sign|
          bindings = bind.call
          bindings[arg] = Conformance.nd(shape, base.each_with_index.map { |v, i| v + (sign * step * dir[i]) })
          plain.run(**bindings).to_a[0]
        end
        numeric = (shifted.call(1.0) - shifted.call(-1.0)) / (2.0 * step)
        analytic = grads[index].each_with_index.sum { |v, i| v * dir[i] }
        Conformance.assert(numeric.abs > 1e-2, "grouped conv d/d#{arg}: derivative #{numeric} is too small to check")
        Conformance.assert((numeric - analytic).abs / [numeric.abs, analytic.abs].max < 3e-2,
                           "grouped conv d/d#{arg}: finite difference #{numeric}, analytic #{analytic}")
      end
    end

    # Hand-written vjps agree with differentiating the decomposition.
    #
    # Finite differences can only say one of the two is wrong, not which, so the
    # fast path needs another reference. The decomposition's gradient follows
    # automatically from primitive vjps, so it is the oracle — the same trick as
    # comparing high-level ops' forward pass with their decompositions.
    #
    # Bit identity is not available (different fold order and accumulator
    # type); the tolerance is the forward pass's 1e-5.
    register(:graph_hand_written_vjps_match_the_decomposed_ones) do
      rows, cols = 3, 4
      x = Conformance.nd([rows, cols], Array.new(rows * cols) { |i| Math.sin((i + 0.6) * 1.7) * 0.8 })
      gamma = Tilework[Conformance.f32(Array.new(cols) { |i| 0.5 + (i * 0.25) })]
      beta = Tilework[Conformance.f32(Array.new(cols) { |i| (i % 3) - 1.0 })]
      labels = Buffer.new(:int32, rows, shape: [rows])
      labels.write(Array.new(rows) { |i| i % cols })

      build = lambda do
        Tilework::Graph.build(differentiable: true) do |g|
          t = g.input(:x, [rows, cols])
          gm = g.param(gamma)
          bt = g.param(beta)
          # For each of the four functions, fold to a scalar and take gradients
          # with respect to x and gamma. Weighting before folding keeps the
          # derivative from vanishing for expressions whose row sums are
          # constant (layernorm).
          losses = [
            (NN.softmax(t) * t).sum,
            (NN.layernorm(t, gamma: gm, beta: bt) * t).sum,
            (NN.rmsnorm(t, gamma: gm) * t).sum,
            NN.cross_entropy(t, g.param(labels), reduction: :sum),
            NN.cross_entropy(t, g.param(labels)),
          ]
          losses.each do |loss|
            g.output(loss)
            g.grad(loss, wrt: [t, gm, bt]).each { |grad| g.output(grad) }
          end
        end.compile
      end

      previous = Tilework::Graph.lower_high_level
      begin
        Tilework::Graph.lower_high_level = false
        fast_plan = build.call
        fast = fast_plan.run(x: x).map(&:to_a)
        Tilework::Graph.lower_high_level = true
        lowered_plan = build.call
        lowered = lowered_plan.run(x: x).map(&:to_a)
      ensure
        Tilework::Graph.lower_high_level = previous
      end

      # On the fast path all four stay :nn nodes (not decomposed). Otherwise both
      # sides would be decompositions and the comparison would pass trivially.
      Conformance.assert(fast_plan.nodes.count { |n| n.op == :nn } >= 4,
                         "the fast path should keep the hand-written kernels as :nn nodes")
      Conformance.assert(lowered_plan.nodes.none? { |n| n.op == :nn },
                         "the lowered path should have no :nn node left")

      names = %w[softmax layernorm rmsnorm cross_entropy_sum cross_entropy_mean]
      fast.each_index do |i|
        what = "#{names[i / 4]} #{%w[loss d/dx d/dgamma d/dbeta][i % 4]}"
        begin
          Conformance.assert_close(lowered[i], fast[i], 1e-5)
        rescue RuntimeError => e
          raise "#{what}: the hand-written vjp disagrees with the decomposed one (#{e.message})"
        end
      end
    end

    # Row fusion works even when a row reduction's result is read in two places.
    #
    # If two clusters pull in the same broadcast, only the first holds the
    # subtree, and the later one references an absorbed node as input and fails
    # with a nil buffer. Row fusion only kicks in above MIN_ROWS, so small-matrix
    # checks never show it. Norm vjps produce this shape.
    register(:graph_row_fusion_handles_a_reduction_read_twice) do
      rows = Tilework::Graph::Fusion::MIN_ROWS
      cols = 4
      x = Conformance.nd([rows, cols], Array.new(rows * cols) { |i| Math.sin((i + 0.7) * 0.13) })

      build = lambda do
        Tilework::Graph.build do |g|
          t = g.input(:x, [rows, cols])
          # **An expression before the reduction** is needed; it has one consumer,
          # so the first cluster absorbs it.
          mean = (t * t).mean(axis: -1, keepdims: true)
          # Then two expressions read the folded value (= two readers of the
          # broadcast). The later cluster used to reference the absorbed `t * t`
          # as input.
          g.output((t - mean) * 2.0)
          g.output((t + mean) * 0.5)
        end.compile
      end

      previous = Tilework::Graph.row_fusion
      begin
        Tilework::Graph.row_fusion = true
        fused = build.call.run(x: x).map(&:to_a)
        Tilework::Graph.row_fusion = false
        plain = build.call.run(x: x).map(&:to_a)
      ensure
        Tilework::Graph.row_fusion = previous
      end

      plain.each_with_index { |values, i| Conformance.assert_close_relative(values, fused[i], 1e-5) }
    end

    # Gradients point "the right way". Global mistakes that per-op checks miss
    # (the reduction's divisor, the update's sign, writing back to a stale
    # version) only show as training not progressing. Run a small MLP for a
    # few steps and check only that the loss decreases monotonically.
    register(:graph_a_training_step_reduces_the_loss) do
      n, d, hidden, classes = 8, 3, 6, 2
      xs = Conformance.nd([n, d], Array.new(n * d) { |i| Math.sin(i * 1.1) })
      labels = Buffer.new(:int32, n, shape: [n])
      labels.write(Array.new(n) { |i| i % classes })
      w1 = Conformance.nd([d, hidden], Array.new(d * hidden) { |i| Math.cos(i * 0.7) * 0.5 })
      w2 = Conformance.nd([hidden, classes], Array.new(hidden * classes) { |i| Math.sin(i * 0.5) * 0.5 })

      plan = Tilework::Graph.build(differentiable: true) do |g|
        x = g.input(:x, [n, d])
        s1 = g.state(w1, name: :w1)
        s2 = g.state(w2, name: :w2)
        p1 = g.read(s1)
        p2 = g.read(s2)
        loss = NN.cross_entropy(NN.gelu(x.matmul(p1)).matmul(p2), g.param(labels))
        d1, d2 = g.grad(loss, wrt: [p1, p2])
        # Gradients and updates live in the same plan (the very reason for a
        # graph rather than a tape).
        g.write_into(s1, value: p1 - (0.5 * d1))
        g.write_into(s2, value: p2 - (0.5 * d2))
        g.output(loss)
      end.compile

      losses = Array.new(12) { plan.run(x: xs).to_a[0] }
      Conformance.assert(losses.each_cons(2).all? { |a, b| b <= a + 1e-6 },
                         "the loss did not fall monotonically: #{losses.inspect}")
      Conformance.assert(losses.last < losses.first * 0.9,
                         "the loss barely moved: #{losses.first} -> #{losses.last}")
    end

    # Whether the budget holds. peak_bytes is measured right after everything
    # needed is loaded (the point of peak const occupancy).
    register(:graph_offload_respects_the_budget) do
      weights = Array.new(5) { Conformance.square(4) }
      x = Conformance.square(4)
      build = lambda do |vram|
        Tilework::Graph.build do |g|
          value = g.input(:x, [4, 4])
          weights.each { |w| value = value.matmul(g.param(w)) }
          g.output(value)
        end.compile(vram: vram)
      end

      probe = build.call(1 << 30).residency
      slot = probe.weights / weights.length
      budget = probe.reserved + (2 * slot)
      plan = build.call(budget)
      begin
        2.times { plan.run(x: x) }
        peak = plan.offload_stats[:peak_bytes]
        Conformance.assert(peak <= budget - probe.reserved,
                           "paged weights peaked at #{peak} bytes, over the #{budget - probe.reserved} allowed")
      ensure
        plan.free_offload!
      end
    end

    # Tilework.submit (#71) changes only how often the host waits. Device-side
    # order stays on the same queue, so it should be **bit-identical** to
    # synchronous eager. A mismatch means an op that must not go async did, or
    # a call leaving the queue was not waited on.
    #
    # A reduction (freeing intermediates as it goes), an NN fused kernel and a
    # free inside submit are mixed into one chain because that is exactly how
    # they combine in eager code.
    register(:submit_matches_the_synchronous_eager_path) do
      x = Tilework[Conformance.f32(Array.new(24) { |i| ((i % 7) - 3) * 0.5 })].reshape(4, 6)
      gamma = Tilework[Conformance.f32(Array.new(6) { |i| 1.0 + (i * 0.1) })]

      expression = lambda do
        h = Tilework::NN.rmsnorm(x, gamma: gamma)
        scaled = h * 2.0
        probs = Tilework::NN.softmax(Tilework::NN.gelu(scaled) + 1.0)
        scaled.free # deferred until completion inside submit
        [probs.to_a, Tilework::Ops.reduce(:sum, probs)]
      end

      expected_probs, expected_sum = expression.call
      actual_probs, actual_sum = Tilework.submit(&expression).wait

      Conformance.assert_equal_arrays(expected_probs, actual_probs)
      Conformance.assert(expected_sum == actual_sum,
                         "submit changed a reduction: #{expected_sum} vs #{actual_sum}")
    end

    # A matmul between two queued kernels. Where the GEMM is delegated it is
    # submitted separately from our kernels, so it has to land **behind** the
    # work that produces its inputs and ahead of the work that reads its
    # output. A bridge that batches asynchronous launches gets this wrong by
    # submitting the GEMM while the producing kernel is still unsubmitted —
    # the GEMM then reads the operand before it is written.
    register(:submit_orders_a_matmul_between_queued_kernels) do
      n = 8
      x = Tilework[Conformance.f32(Array.new(n * n) { |i| ((i % 7) - 3) * 0.25 })].reshape(n, n)
      w = Tilework[Conformance.f32(Array.new(n * n) { |i| ((i % 5) - 2) * 0.5 })].reshape(n, n)

      expression = -> { ((x * 3.0) + 1.0).matmul(w) * 0.5 }

      expected = expression.call.to_a
      actual = Tilework.submit(&expression).wait.to_a
      Conformance.assert_equal_arrays(expected, actual)
    end

    # Buffers made inside submit must not be freed before completion. Releasing
    # memory a running kernel reads is undefined behavior on CUDA/HIP, so this
    # pins not only that values match but that the buffer lives until the boundary.
    register(:submit_defers_a_free_to_the_boundary) do
      a = Tilework[Conformance.f32([1.0, 2.0, 3.0, 4.0])]
      temp = nil

      job = Tilework.submit do
        temp = a * 2.0
        temp.free
        Conformance.assert(!temp.freed?, "a buffer freed inside submit must stay alive until the boundary")
        temp + 1.0
      end

      Conformance.assert(!temp.freed?, "the deferred free must not run before wait")
      out = job.wait
      Conformance.assert(temp.freed?, "the deferred free must run at the boundary")
      Conformance.assert_equal_arrays([3.0, 5.0, 7.0, 9.0], out.to_a)
    end

    # #86: ops that free their own temporaries mid-call (:nn / whole-buffer
    # reductions / batched matmul without a delegate) no longer run
    # synchronously, since the execution context takes over those frees.
    # **Values must not change** — if they do, a freed slot is being lent to
    # the next node while still in use.
    #
    # Three consecutive runs because double-lending a slot does not show on the first.
    register(:graph_deferred_free_matches_synchronous_execution) do
      state_buffer = -> { Tilework[Conformance.f32([0.5, -1.5, 2.0, 0.25, 1.0, -0.5])].reshape(3, 2) }

      build = lambda do |cache|
        Tilework::Graph.build do |g|
          x = g.input(:x, [4, 6])
          h = Tilework::NN.silu(Tilework::NN.rmsnorm(x))
          total = h.sum # whole-buffer reduction (tree fold, freeing partials midway)
          s0 = g.state(cache, name: :acc)
          s1 = g.write_into(s0, at: 0..1, value: (g.read(s0, 0..1) * 1.5))
          g.output(h + total)
          g.output(g.read(s1, 0..2).reshape(6))
        end.compile
      end

      x = Tilework[Conformance.f32(Array.new(24) { |i| ((i % 5) - 2) * 0.75 })].reshape(4, 6)
      previous = Tilework::Graph.async
      begin
        Tilework::Graph.async = false
        serial_cache = state_buffer.call
        serial = build.call(serial_cache)
        expected = 3.times.map { serial.run(x: x).map(&:to_a) }

        Tilework::Graph.async = true
        cache = state_buffer.call
        plan = build.call(cache)
        actual = 3.times.map { plan.run(x: x).map(&:to_a) }
      ensure
        Tilework::Graph.async = previous
      end

      expected.zip(actual).each do |want, got|
        want.zip(got).each { |a, b| Conformance.assert_equal_arrays(a, b) }
      end
      Conformance.assert(plan.sync_stats[:synchronous].zero?,
                         "no node here should still wait, got #{plan.sync_stats.inspect}")
    end

    # #86 stages 2-3: several plans run inside one submit give the same values as
    # running them standalone. A plan installs its own execution context, so
    # without ownership linked to the outer Job, frees of buffers made in the
    # submit block would land outside the boundary.
    register(:graph_inside_submit_matches_the_standalone_plans) do
      first = Tilework::Graph.build do |g|
        g.output(Tilework::NN.silu(g.input(:x, [4, 6])) * 2.0)
      end.compile
      second = Tilework::Graph.build do |g|
        g.output(Tilework::NN.softmax(g.input(:y, [4, 6])) + 0.5)
      end.compile

      source = Tilework[Conformance.f32(Array.new(24) { |i| ((i % 6) - 2) * 0.5 })].reshape(4, 6)

      chain = lambda do
        scaled = source * 1.25
        out = second.run(y: first.run(x: scaled))
        scaled.free # deferred to the boundary inside submit
        out
      end

      expected = chain.call.to_a
      job = Tilework.submit { chain.call }
      actual = job.wait.to_a

      Conformance.assert_equal_arrays(expected, actual)
    end
  end
end
