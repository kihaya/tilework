# frozen_string_literal: true

require_relative "test_helper"

# Tests that need a real Metal device (Apple Silicon); run with rake test:metal.
# Backend-independent Ops/DSL/with_gpu checks belong to conformance_metal_test.rb
# (Tilework::Conformance), so this file covers what is specific to the Metal
# bridge: raw MSL, dialect strings, compile errors, resource health.
class MetalIntegrationTest < Minitest::Test
  VEC_ADD = <<~KERNEL
    GPU_KERNEL void vec_add(GPU_ARGS,
                            device const float* a,
                            device const float* b,
                            device float* c,
                            constant int& n) {
      int i = GLOBAL_ID_X;
      if (i < n) c[i] = a[i] + b[i];
    }
  KERNEL

  def setup
    skip "GPU tests disabled (set GPU_TESTS=1 or run rake test:metal)" unless gpu_tests_enabled?
    skip "not running on the Metal backend" unless gpu_backend == :metal
    Tilework.backend = :metal
    Tilework::Kernel.clear_cache!
  end

  def assert_float_arrays_equal(expected, actual)
    assert_equal expected.length, actual.length
    expected.zip(actual).each_with_index do |(e, a), i|
      assert_in_delta e, a, 1e-6, "element #{i}"
    end
  end

  # Reference Conv2D, checking MPSGraph's convolution2D independently.
  # NCHW / weights OIHW with zero padding (the conv.rb convention).
  def naive_conv2d(x, w, n:, cin:, h:, width:, cout:, kh:, kw:, pad_h:, pad_w:, stride_h:, stride_w:)
    oh = ((h + (2 * pad_h) - kh) / stride_h) + 1
    ow = ((width + (2 * pad_w) - kw) / stride_w) + 1
    out = Array.new(n * cout * oh * ow, 0.0)
    xi = ->(b, c, i, j) { (i.negative? || i >= h || j.negative? || j >= width) ? 0.0 : x[(((b * cin) + c) * h + i) * width + j] }
    wi = ->(o, c, i, j) { w[(((o * cin) + c) * kh + i) * kw + j] }

    n.times do |b|
      cout.times do |o|
        oh.times do |oy|
          ow.times do |ox|
            sum = 0.0
            cin.times do |c|
              kh.times do |ky|
                kw.times do |kx|
                  iy = (oy * stride_h) - pad_h + ky
                  ix = (ox * stride_w) - pad_w + kx
                  sum += xi.(b, c, iy, ix) * wi.(o, c, ky, kx)
                end
              end
            end
            out[(((b * cout) + o) * oh + oy) * ow + ox] = sum
          end
        end
      end
    end
    out
  end

  # Reference FFT (O(n^2)): a naive DFT to check MPSGraph's result independently.
  def naive_dft(signal)
    n = signal.length
    Array.new(n) do |k|
      re = 0.0
      im = 0.0
      signal.each_with_index do |x, t|
        angle = -2 * Math::PI * k * t / n
        re += x * Math.cos(angle)
        im += x * Math.sin(angle)
      end
      [re, im]
    end
  end

  def test_device_name_is_present
    refute_empty Tilework.device_name
  end

  def test_buffer_roundtrip_on_device
    values = Array.new(1000) { rand(-100.0..100.0) }
    expected = values.pack("f*").unpack("f*") # the reference is the value rounded to float32
    assert_float_arrays_equal expected, Tilework[values].to_a
  end

  def test_vec_add_matches_cpu_reference
    n = 4096
    a = Array.new(n) { rand }
    b = Array.new(n) { rand }
    da = Tilework[a]
    db = Tilework[b]
    dc = Tilework::Buffer.new(:float32, n)

    kernel = Tilework.compile(VEC_ADD, name: "vec_add")
    kernel.launch(grid: (n / 256.0).ceil, block: 256, args: [da, db, dc, n])

    expected = a.zip(b).map { |x, y| x + y }
    assert_float_arrays_equal expected, dc.to_a
  end

  def test_launch_async_and_to_a_waits_for_completion
    n = 4096
    a = Array.new(n) { rand }
    da = Tilework[a]
    db = Tilework[a]
    dc = Tilework::Buffer.new(:float32, n)

    kernel = Tilework.compile(VEC_ADD, name: "vec_add")
    kernel.launch_async(grid: (n / 256.0).ceil, block: 256, args: [da, db, dc, n])

    expected = a.map { |x| x * 2 }
    assert_float_arrays_equal expected, dc.to_a
  end

  def test_int32_kernel_arguments
    n = 256
    src = <<~KERNEL
      GPU_KERNEL void scale(GPU_ARGS, device int* a, constant int& factor, constant int& n) {
        int i = GLOBAL_ID_X;
        if (i < n) a[i] = a[i] * factor;
      }
    KERNEL
    buffer = Tilework[(1..n).to_a]
    Tilework.compile(src, name: "scale").launch(grid: 1, block: n, args: [buffer, 3, n])
    assert_equal (1..n).map { |v| v * 3 }, buffer.to_a
  end

  def test_compile_error_includes_metal_compiler_output
    error = assert_raises(Tilework::CompileError) do
      Tilework.compile("GPU_KERNEL void broken(GPU_ARGS) { undefined_fn(); }", name: "broken")
    end
    assert_match(/error/i, error.message)
  end

  def test_raw_msl_escape_hatch
    src = <<~MSL
      #include <metal_stdlib>
      using namespace metal;
      kernel void fill(device float* out [[buffer(0)]],
                       uint gid [[thread_position_in_grid]]) {
        out[gid] = 42.0;
      }
    MSL
    buffer = Tilework::Buffer.new(:float32, 64)
    Tilework.compile(src, name: "fill", raw: true).launch(grid: 1, block: 64, args: [buffer])
    assert buffer.to_a.all? { |v| (v - 42.0).abs < 1e-6 }
  end

  def test_block_limit_validation_raises_before_dispatch
    kernel = Tilework.compile(VEC_ADD, name: "vec_add")
    assert_raises(Tilework::LaunchError) do
      kernel.launch(grid: 1, block: 100_000, args: [])
    end
  end

  def test_gemm_available
    assert Tilework::MetalBridge.respond_to?(:gemm_available?)
    assert Tilework::MetalBridge.gemm_available?
    assert Tilework::MetalBridge.gemm_batched_available?
  end

  # conformance_metal_test.rb (Tilework::Conformance) already checks matmul
  # correctness for float32 through Ops/NN; fp16 input isn't exercised there,
  # and it's the path most likely to break from a dtype/rowBytes mixup in the
  # MPS descriptor (see bridge.swift's GEMM section — leftMatrix/rightMatrix
  # dataType: .float16, resultMatrix dataType: .float32).
  def test_gemm_float16_input_matches_float32_reference
    m, k, n = 5, 6, 4
    a = Array.new(m * k) { rand(-1.0..1.0) }
    b = Array.new(k * n) { rand(-1.0..1.0) }
    expected = Array.new(m) { |i|
      Array.new(n) { |j| (0...k).sum { |t| a[i * k + t] * b[t * n + j] } }
    }.flatten

    da = Tilework::Buffer.new(:float16, a.length, shape: [m, k])
    da.write(a)
    db = Tilework::Buffer.new(:float16, b.length, shape: [k, n])
    db.write(b)
    out = da.matmul(db)

    assert_equal :float32, out.dtype
    out.to_a.zip(expected).each_with_index do |(got, want), i|
      assert_in_delta want, got, 5e-2, "element #{i}"
    end
  end

  # Regression guard for the row-major (`gemm`) vs. column-major
  # (`gemm_strided_batched`) layout split in bridge.swift: a non-square,
  # transposed-A batched call exercises the lda/stride/trans_a bookkeeping
  # that NN.multi_head_attention relies on for its head-slicing views.
  def test_gemm_batched_matches_manual_reference_with_transpose_and_stride
    m, k, n, batch = 3, 5, 2, 4

    a_heads = Array.new(batch) { Array.new(k * m) { rand(-1.0..1.0) } } # stored (k, m), col-major, per head
    b_heads = Array.new(batch) { Array.new(k * n) { rand(-1.0..1.0) } } # stored (k, n), col-major, per head

    da = Tilework::Buffer.new(:float32, batch * k * m)
    da.write(a_heads.flatten)
    db = Tilework::Buffer.new(:float32, batch * k * n)
    db.write(b_heads.flatten)
    dc = Tilework::Buffer.new(:float32, batch * m * n, shape: [batch * m, n])

    Tilework::TensorCore.gemm_batched(da, db, dc,
                                      m: m, k: k, n: n,
                                      lda: k, ldb: k, ldc: m,
                                      stride_a: k * m, stride_b: k * n, stride_c: m * n,
                                      batch: batch, trans_a: true)

    got = dc.to_a
    batch.times do |h|
      a_col = a_heads[h] # (k, m) col-major
      b_col = b_heads[h] # (k, n) col-major
      expected_col = Array.new(m * n) do |idx|
        i = idx % m
        j = idx / m
        (0...k).sum { |t| a_col[t + i * k] * b_col[t + j * k] }
      end
      got_head = got[(h * m * n)...((h + 1) * m * n)]
      expected_col.zip(got_head).each_with_index do |(want, gotv), i|
        assert_in_delta want, gotv, 1e-4, "head #{h} element #{i}"
      end
    end
  end

  def test_random_available
    assert Tilework::MetalBridge.respond_to?(:random_available?)
    assert Tilework::MetalBridge.random_available?
    assert_equal :mps, Tilework::Random.library
  end

  # Random is not in Tilework::Conformance (only statistical properties can be
  # checked, which does not fit a framework built on bit identity with a CPU
  # oracle). This checks consistency with cuRAND's convention (uniform in
  # (0, 1], excluding 0 and including 1) and determinism under a seed.
  def test_random_uniform_matches_curand_convention_and_is_deterministic
    values = Tilework.random(200_000, seed: 42).to_a

    refute_includes values, 0.0
    assert values.all? { |v| v > 0.0 && v <= 1.0 }
    assert_in_delta 0.5, values.sum / values.length, 0.01

    assert_equal Tilework.random(500, seed: 7).to_a, Tilework.random(500, seed: 7).to_a
  end

  def test_random_normal_matches_requested_mean_and_stddev
    buffer = Tilework::Buffer.new(:float32, 200_000)
    Tilework::Random.normal!(buffer, mean: 5.0, stddev: 2.0, seed: 7)

    values = buffer.to_a
    mean = values.sum / values.length
    variance = values.sum { |v| (v - mean)**2 } / values.length

    assert_in_delta 5.0, mean, 0.05
    assert_in_delta 2.0, Math.sqrt(variance), 0.05
  end

  # MPSMatrixRandomPhilox emits four elements at a time, and requesting a
  # count that is not a multiple of four kills the process with an assert
  # (not an exception, so it cannot be rescued). The bridge rounds up, so this
  # checks that odd lengths simply go through.
  def test_random_handles_lengths_that_are_not_a_multiple_of_four
    [1, 2, 3, 5, 7, 21, 100_001].each do |n|
      values = Tilework.random(n, seed: 3).to_a
      assert_equal n, values.length
      assert values.all? { |v| v > 0.0 && v <= 1.0 }, "uniform out of range for n=#{n}"

      normals = Tilework.randn([n], seed: 3).to_a
      assert_equal n, normals.length
      assert normals.all? { |v| v.finite? }, "normal not finite for n=#{n}"
    end

    values = Tilework.random(100_001, seed: 11).to_a
    assert_in_delta 0.5, values.sum / values.length, 0.01
  end

  def test_fft_available
    assert Tilework::MetalBridge.respond_to?(:fft_available?)
    assert Tilework::MetalBridge.fft_available?
    assert_equal :mps, Tilework::FFT.library
  end

  # Like Random, FFT is not in Conformance. This checks R2C against a naive DFT
  # (MPSGraphFFTDescriptor's scalingMode/inverse are easy to get wrong) and
  # C2R/C2C by round trip.
  def test_fft_r2c_matches_naive_dft_reference
    n = 16
    signal = Array.new(n) { rand(-1.0..1.0) }
    expected = naive_dft(signal).first(n / 2 + 1)

    spec = Tilework::FFT.forward(Tilework[signal])
    got = spec.to_a
    (0...(n / 2 + 1)).each do |k|
      assert_in_delta expected[k][0], got[2 * k], 1e-3, "re[#{k}]"
      assert_in_delta expected[k][1], got[(2 * k) + 1], 1e-3, "im[#{k}]"
    end
  end

  def test_fft_c2r_roundtrips_r2c_for_even_and_odd_n
    [16, 15].each do |n|
      signal = Array.new(n) { rand(-1.0..1.0) }
      spec = Tilework::FFT.forward(Tilework[signal])
      recovered = Tilework::FFT.inverse(spec, n).to_a
      signal.each_with_index do |v, i|
        assert_in_delta v, recovered[i], 1e-3, "n=#{n} element #{i}"
      end
    end
  end

  def test_fft_c2c_forward_inverse_roundtrips
    n = 8
    signal = Array.new(n * 2) { rand(-1.0..1.0) }
    forward = Tilework::FFT.transform(Tilework[signal], direction: :forward)
    back = Tilework::FFT.transform(forward, direction: :inverse).to_a
    signal.each_with_index do |v, i|
      assert_in_delta v, back[i], 1e-3, "element #{i}"
    end
  end

  def test_fft_batched_matches_naive_dft_per_batch_element
    n, batch = 16, 3
    signals = Array.new(batch) { Array.new(n) { rand(-1.0..1.0) } }
    spec = Tilework::FFT.forward(Tilework[signals.flatten], batch: batch)
    got = spec.to_a
    per = 2 * ((n / 2) + 1)

    batch.times do |b|
      expected = naive_dft(signals[b]).first((n / 2) + 1)
      (0...(n / 2 + 1)).each do |k|
        assert_in_delta expected[k][0], got[(b * per) + (2 * k)], 1e-3, "batch #{b} re[#{k}]"
        assert_in_delta expected[k][1], got[(b * per) + (2 * k) + 1], 1e-3, "batch #{b} im[#{k}]"
      end
    end
  end

  def test_conv2d_available
    assert Tilework::MetalBridge.respond_to?(:conv2d_available?)
    assert Tilework::MetalBridge.conv2d_available?
    assert Tilework::MetalBridge.conv2d_backward_available?
    assert_equal :mps, Tilework::Conv.library
  end

  # Like Random/FFT, Conv is not in Conformance. This combines non-square
  # kernel/stride/padding with float16 input, where MPSGraphConvolution2D
  # OpDescriptor's NCHW/OIHW mix-ups and fp16 rounding are likely.
  def test_conv2d_forward_matches_naive_reference_with_asymmetric_params_and_fp16
    n, cin, h, width, cout, kh, kw = 2, 3, 9, 7, 4, 3, 5
    pad_h, pad_w, stride_h, stride_w = 1, 2, 2, 1

    x_data = Array.new(n * cin * h * width) { rand(-1.0..1.0) }
    w_data = Array.new(cout * cin * kh * kw) { rand(-1.0..1.0) }

    x = Tilework::Buffer.new(:float16, x_data.length, shape: [n, cin, h, width])
    x.write(x_data)
    weight = Tilework::Buffer.new(:float16, w_data.length, shape: [cout, cin, kh, kw])
    weight.write(w_data)

    y = Tilework::Conv.forward(x, weight, pad: [pad_h, pad_w], stride: [stride_h, stride_w])
    expected = naive_conv2d(x_data, w_data, n: n, cin: cin, h: h, width: width, cout: cout, kh: kh, kw: kw,
                            pad_h: pad_h, pad_w: pad_w, stride_h: stride_h, stride_w: stride_w)

    assert_equal :float16, y.dtype
    got = y.to_a
    expected.each_with_index do |v, i|
      assert_in_delta v, got[i], 5e-2, "element #{i}"
    end
  end

  # Backward has no analytic reference (naive_conv2d is not differentiable),
  # so dL/dx and dL/dw are checked independently by finite differences.
  def test_conv2d_backward_matches_finite_differences
    n, cin, h, width, cout, kh, kw = 1, 2, 6, 6, 3, 3, 3
    pad, stride = 1, 1
    dims = { n: n, cin: cin, h: h, width: width, cout: cout, kh: kh, kw: kw,
            pad_h: pad, pad_w: pad, stride_h: stride, stride_w: stride }

    x_data = Array.new(n * cin * h * width) { rand(-1.0..1.0) }
    w_data = Array.new(cout * cin * kh * kw) { rand(-1.0..1.0) }

    x = Tilework::Buffer.new(:float32, x_data.length, shape: [n, cin, h, width])
    x.write(x_data)
    weight = Tilework::Buffer.new(:float32, w_data.length, shape: [cout, cin, kh, kw])
    weight.write(w_data)

    y = Tilework::Conv.forward(x, weight, pad: pad, stride: stride)
    grad_output_data = Array.new(y.length) { rand(-1.0..1.0) }
    grad_output = Tilework::Buffer.new(:float32, grad_output_data.length, shape: y.shape)
    grad_output.write(grad_output_data)

    dx, dw = Tilework::Conv.backward(x, weight, grad_output, pad: pad, stride: stride)

    loss = lambda do |xd, wd|
      naive_conv2d(xd, wd, **dims).zip(grad_output_data).sum { |a, b| a * b }
    end

    eps = 1e-3
    dx_got = dx.to_a
    [0, 5, 10].each do |i|
      xp = x_data.dup.tap { |a| a[i] += eps }
      xm = x_data.dup.tap { |a| a[i] -= eps }
      grad_fd = (loss.(xp, w_data) - loss.(xm, w_data)) / (2 * eps)
      assert_in_delta grad_fd, dx_got[i], 1e-2, "dx[#{i}]"
    end

    dw_got = dw.to_a
    [0, 5, 10].each do |i|
      wp = w_data.dup.tap { |a| a[i] += eps }
      wm = w_data.dup.tap { |a| a[i] -= eps }
      grad_fd = (loss.(x_data, wp) - loss.(x_data, wm)) / (2 * eps)
      assert_in_delta grad_fd, dw_got[i], 1e-2, "dw[#{i}]"
    end
  end

  # Delegated convolutions also take their outputs from the pool.
  #
  # The plan **returns intermediates to the pool instead of freeing them**
  # (Plan#release), so an allocation bypassing the pool adds one slot per run.
  # Conv.forward / backward_data / backward_filter had to route through
  # Ops.allocate_output for that reason.
  #
  # MockBridge cannot reach this — the :custom backend does not carry
  # delegated_library_capabilities, so conv2d is decomposed into primitives at
  # record time. It only reproduces on hardware, hence its place here.
  def test_delegated_conv2d_output_comes_from_the_pool
    weight = Tilework::Buffer.new(:float32, 4 * 4 * 3, shape: [4, 4, 1, 3])
    weight.write(Array.new(4 * 4 * 3) { 0.01 })
    x = Tilework::Buffer.new(:float32, 4 * 64, shape: [1, 4, 1, 64])
    x.write(Array.new(4 * 64) { 0.01 })

    plan = Tilework::Graph.build do |g|
      t = g.input(:x, [1, 4, 1, 64])
      # Two stages: with one, the growth cancels against the single exit and is invisible.
      2.times do
        t = g.conv2d(t, g.param(weight), shape: [1, 4, 1, 62], pad: [0, 0], stride: [1, 1])
        t = g.pad(t.reshape(1, 4, 1, 62), before: [0, 0, 0, 1], after: [0, 0, 0, 1])
      end
      g.output(t)
    end.compile

    assert_includes plan.order.map(&:op), :conv2d, "展開されていたらこのテストは何も見ていない"

    3.times { plan.run(x: x).free }
    steady = plan.pool_stats[:pooled]
    3.times { plan.run(x: x).free }

    assert_equal steady, plan.pool_stats[:pooled],
                 "conv2d がプールを迂回して確保すると、実行のたびにプールが増える"
  end

  # MPSGraph's convolution silently breaks when cout > 65536 (rows past 65536
  # are never written and the first few rows differ). bridge.swift splits cout
  # to avoid it, so check that shapes crossing the boundary match the reference.
  #
  # Convolving everything in Ruby would not finish in reasonable time, so only
  # **the channels around the boundary** are compared (it breaks per channel,
  # so this catches it).
  def test_conv2d_forward_is_correct_beyond_65536_output_channels
    cin, h, width, kh, kw = 8, 1, 8, 1, 3
    cout = 70_000
    pad_w = 1

    x_data = Array.new(cin * h * width) { rand(-1.0..1.0) }
    w_data = Array.new(cout * cin * kh * kw) { rand(-1.0..1.0) }

    x = Tilework::Buffer.new(:float32, x_data.length, shape: [1, cin, h, width])
    x.write(x_data)
    weight = Tilework::Buffer.new(:float32, w_data.length, shape: [cout, cin, kh, kw])
    weight.write(w_data)

    got = Tilework::Conv.forward(x, weight, pad: [0, pad_w]).to_a

    [0, 65_535, 65_536, 65_537, cout - 1].each do |oc|
      width.times do |t|
        expected = 0.0
        cin.times do |c|
          kw.times do |j|
            s = t + j - pad_w
            next if s.negative? || s >= width

            expected += x_data[(c * width) + s] * w_data[(((oc * cin) + c) * kw) + j]
          end
        end
        assert_in_delta expected, got[(oc * width) + t], 1e-4, "channel #{oc}, column #{t}"
      end
    end
  end

  # Asynchronous launches are batched into one open command buffer, which is
  # not on the queue until it is committed. A delegated MPS / MPSGraph call
  # commits a command buffer of its own, so it must flush that batch first or
  # it runs before the kernels that produce its inputs (FFT returned all
  # zeros, a GEMM read its operand before it was written).
  def test_delegated_calls_run_behind_an_open_async_batch
    n = 64
    x = Tilework[Array.new(n * n) { rand(-1.0..1.0) }].reshape(n, n)
    w = Tilework[Array.new(n * n) { rand(-1.0..1.0) }].reshape(n, n)

    gemm = -> { ((x * 3.0) + 1.0).matmul(w) * 0.5 }
    assert_equal gemm.call.to_a, Tilework.submit(&gemm).wait.to_a, "GEMM"

    fft = -> { Tilework::FFT.forward(((x * 3.0) + 1.0).reshape(n * n)) }
    assert_equal fft.call.to_a, Tilework.submit(&fft).wait.to_a, "FFT"

    image = Tilework[Array.new(2 * 8 * 8) { rand(-1.0..1.0) }].reshape(1, 2, 8, 8)
    kernel = Tilework[Array.new(3 * 2 * 3 * 3) { rand(-1.0..1.0) }].reshape(3, 2, 3, 3)
    conv = -> { Tilework::Conv.forward((image * 2.0) + 0.5, kernel, pad: 1) }
    assert_equal conv.call.to_a, Tilework.submit(&conv).wait.to_a, "conv2d"
  end

  # MPS GEMM is committed onto the same queue as our kernels (`ordered_gemm?`),
  # so a graph does not fence it. Values must still match the serial plan with
  # pooled slots being reused across the chain.
  def test_graph_matmul_is_unfenced_and_matches_serial_execution
    n = 96
    x = Tilework[Array.new(n * n) { rand(-0.5..0.5) }].reshape(n, n)
    weights = Array.new(4) { Tilework[Array.new(n * n) { rand(-0.5..0.5) }].reshape(n, n) }
    build = lambda do
      Tilework::Graph.build do |g|
        h = g.input(:x, [n, n])
        weights.each do |weight|
          param = g.param(weight)
          h = (((h.matmul(param) * 0.05) + 0.01).matmul(param) * 0.05) - (h * 0.5)
        end
        g.output(h)
      end.compile
    end

    Tilework::Graph.async = false
    serial = build.call.run(x: x).to_a
    Tilework::Graph.async = true

    plan = build.call
    10.times { assert_equal serial, plan.run(x: x).to_a }
    assert_equal 0, plan.sync_stats[:fences]
    assert_equal 0, plan.sync_stats[:synchronous]
  ensure
    Tilework::Graph.async = true
  end

  def test_buffer_alloc_free_cycles_do_not_grow_rss
    # Warm up first. RSS must not grow monotonically over 2000 cycles of 1 KB.
    50.times { Tilework::Buffer.new(:float32, 256).free }
    GC.start
    before = rss_kb
    2000.times { Tilework::Buffer.new(:float32, 256).free }
    GC.start
    after = rss_kb
    assert_operator after - before, :<, 20_000, "RSS grew from #{before}KB to #{after}KB"
  end

  def test_host_written_buffers_release_their_shared_pages
    # Writes through `MTLBuffer.contents()` dirty shared system-memory pages.
    # Unless `#free` makes the resource purgeable again, RSS keeps as much as
    # was written (#52). Small allocation/free loops do not expose it, so this
    # writes 256 MB in total from the host.
    bytes = 16 * 1024 * 1024
    payload = "\0".b * bytes
    Tilework::Buffer.new(:float32, bytes / 4).write_bytes(payload).free # warm-up
    GC.start
    before = rss_kb
    16.times { Tilework::Buffer.new(:float32, bytes / 4).write_bytes(payload).free }
    GC.start
    after = rss_kb

    assert_operator after - before, :<, 96_000,
                    "host-written buffers retained #{after - before}KB of RSS"
  end

  def test_fft_cycles_do_not_grow_footprint
    # Temporary objects MPSGraph.run creates are autoreleased. Called from Ruby
    # there is no run loop draining the pool, so unless bridge.swift opens an
    # autoreleasepool per call, several MB pile up per call.
    #
    # **RSS does not catch it** — the leak is MPS temporary buffers
    # (wired/IOKit), which grow phys_footprint without showing in RSS; jetsam
    # watches the former.
    frames = 256
    n_fft = 1024
    framed = Tilework::Buffer.new(:float32, frames * n_fft, shape: [frames, n_fft])
    10.times { Tilework::FFT.forward(framed, batch: frames).free } # warm-up
    before = footprint_kb
    100.times { Tilework::FFT.forward(framed, batch: frames).free }
    after = footprint_kb
    framed.free
    assert_operator after - before, :<, 50_000,
                    "phys_footprint grew from #{before}KB to #{after}KB over 100 FFTs"
  end

  private

  def rss_kb
    `ps -o rss= -p #{Process.pid}`.to_i
  end

  # phys_footprint (KB). Metal device memory shows up here, not in RSS.
  def footprint_kb
    out = `footprint -p #{Process.pid} 2>/dev/null`
    out[/phys_footprint:\s+(\d+)\s*KB/, 1]&.to_i ||
      (out[/phys_footprint:\s+(\d+)\s*MB/, 1]&.to_i&.*(1024)) ||
      flunk("could not read phys_footprint")
  end
end
