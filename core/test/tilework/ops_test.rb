# frozen_string_literal: true

require_relative "../test_helper"

class OpsTest < Minitest::Test
  include BridgeTestSetup

  def last_kernel_src
    @bridge.compilations.last[:src]
  end

  def last_dispatch
    @bridge.dispatches.last
  end

  def test_buffer_plus_buffer_generates_elementwise_kernel
    a = Tilework[[1.0, 2.0]]
    b = Tilework[[3.0, 4.0]]
    c = a + b
    assert_instance_of Tilework::Buffer, c
    assert_equal :float32, c.dtype
    assert_equal 2, c.length
    assert_includes last_kernel_src, "c[i] = (float)a[i] + (float)b[i];"
  end

  def test_int_buffers_stay_int32
    a = Tilework[[1, 2]]
    b = Tilework[[3, 4]]
    assert_equal :int32, (a * b).dtype
    assert_includes last_kernel_src, "c[i] = (int)a[i] * (int)b[i];"
  end

  def test_mixed_dtypes_promote_to_float32
    a = Tilework[[1, 2]]
    b = Tilework[[1.0, 2.0]]
    assert_equal :float32, (a + b).dtype
  end

  def test_length_mismatch_raises
    a = Tilework[[1.0]]
    b = Tilework[[1.0, 2.0]]
    assert_raises(ArgumentError) { a + b }
  end

  def test_unsupported_operand_raises
    assert_raises(TypeError) { Tilework[[1.0]] + "x" }
  end

  def test_buffer_times_scalar_passes_scalar_as_bytes
    a = Tilework[[1.0, 2.0]]
    a * 2
    assert_includes last_kernel_src, "constant float& s"
    # The integer 2 is passed as a float because the output is float32
    assert_includes last_dispatch[:args], [2.0].pack("f")
  end

  def test_dispatch_failure_immediately_frees_the_allocated_output
    handles = []
    @bridge.define_singleton_method(:buffer_alloc) do |bytesize|
      MockBridge::Handle.new(("\x00" * bytesize).b, false, 0).tap { |handle| handles << handle }
    end
    @bridge.define_singleton_method(:dispatch) do |*_args|
      raise RuntimeError, "synthetic dispatch failure"
    end
    input = Tilework[[1.0, 2.0]]

    assert_raises(Tilework::LaunchError) { input + 1.0 }

    refute handles.first.freed, "the caller still owns the input"
    assert handles.last.freed, "the failed operation must not wait for GC to release its output"
  end

  def test_int_buffer_with_integer_scalar_stays_int32
    a = Tilework[[1, 2]]
    assert_equal :int32, (a + 3).dtype
    assert_includes last_dispatch[:args], [3].pack("l")
  end

  def test_coerce_enables_scalar_on_the_left
    a = Tilework[[1.0, 2.0]]
    c = 2.0 * a
    assert_instance_of Tilework::Buffer, c
    assert_includes last_kernel_src, "c[i] = (float)a[i] * s;"
  end

  def test_coerce_reverses_noncommutative_operations
    a = Tilework[[1.0, 2.0]]
    10.0 - a
    assert_includes last_kernel_src, "c[i] = s - (float)a[i];"
    10.0 / a
    assert_includes last_kernel_src, "c[i] = s / (float)a[i];"
  end

  def test_identical_op_reuses_cached_pipeline
    a = Tilework[[1.0, 2.0]]
    b = Tilework[[3.0, 4.0]]
    a + b
    a + b
    assert_equal 1, @bridge.compilations.size
  end

  def test_sum_generates_reduction_kernel_with_barrier
    Tilework[[1.0, 2.0, 3.0]].sum
    assert_includes last_kernel_src, "GPU_BARRIER;"
    assert_includes last_kernel_src, "GPU_SHARED float sh[256];"
  end

  def test_unary_math_promotes_int_to_float32
    a = Tilework[[1, 2]]
    assert_equal :float32, a.exp.dtype
    assert_includes last_kernel_src, "c[i] = exp((float)a[i]);"
  end

  def test_unary_math_functions_generate_expected_calls
    a = Tilework[[1.0, 2.0]]
    { log: "log", sqrt: "sqrt", tanh: "tanh", floor: "floor" }.each do |meth, fn|
      assert_equal :float32, a.public_send(meth).dtype
      assert_includes last_kernel_src, "c[i] = #{fn}((float)a[i]);"
    end
  end

  def test_abs_keeps_dtype_and_picks_fabs_or_abs
    Tilework[[-1.0, 2.0]].abs
    assert_includes last_kernel_src, "c[i] = fabs(a[i]);"
    assert_equal :int32, Tilework[[-1, 2]].abs.dtype
    assert_includes last_kernel_src, "c[i] = abs(a[i]);"
  end

  def test_comparison_returns_int32_mask
    a = Tilework[[1.0, 2.0]]
    b = Tilework[[2.0, 1.0]]
    mask = a > b
    assert_equal :int32, mask.dtype
    assert_includes last_kernel_src, "c[i] = (float)a[i] > (float)b[i] ? 1 : 0;"
  end

  def test_comparison_with_scalar_and_coerce
    a = Tilework[[1.0, 2.0]]
    assert_equal :int32, (a <= 1.5).dtype
    assert_includes last_kernel_src, "c[i] = (float)a[i] <= s ? 1 : 0;"
    # 1.5 > a is flipped into the equivalent a < 1.5
    assert_equal :int32, (1.5 > a).dtype
    assert_includes last_kernel_src, "c[i] = (float)a[i] < s ? 1 : 0;"
  end

  def test_eq_and_ne_masks
    a = Tilework[[1, 2]]
    assert_equal :int32, a.eq(2).dtype
    assert_includes last_kernel_src, "c[i] = (int)a[i] == s ? 1 : 0;"
    a.ne(Tilework[[1, 3]])
    assert_includes last_kernel_src, "c[i] = (int)a[i] != (int)b[i] ? 1 : 0;"
  end

  def test_maximum_minimum_generate_min_max_calls
    a = Tilework[[1.0, 2.0]]
    b = Tilework[[2.0, 1.0]]
    assert_equal :float32, a.maximum(b).dtype
    assert_includes last_kernel_src, "c[i] = max((float)a[i], (float)b[i]);"
    assert_equal :int32, Tilework[[1, 2]].minimum(0).dtype
    assert_includes last_kernel_src, "c[i] = min((int)a[i], s);"
  end

  def test_clamp_passes_bounds_as_scalars
    a = Tilework[[1.0, 2.0]]
    assert_equal :float32, a.clamp(0.0, 1.0).dtype
    assert_includes last_kernel_src, "c[i] = clamp((float)a[i], lo, hi);"
    assert_includes last_dispatch[:args], [1.0].pack("f")
    assert_equal :int32, Tilework[[1, 2]].clamp(0, 5).dtype
  end

  def test_clamp_rejects_non_numeric_bounds
    assert_raises(TypeError) { Tilework[[1.0]].clamp(0.0, "x") }
  end

  def test_min_max_reductions_use_matching_identity
    a = Tilework[[1.0, 2.0]]
    a.max
    assert_includes last_kernel_src, "sh[lid] = gid < n ? a[gid] : -INFINITY;"
    assert_includes last_kernel_src, "sh[lid] = max(sh[lid], sh[lid + s]);"
    Tilework[[1, 2]].min
    assert_includes last_kernel_src, "sh[lid] = gid < n ? a[gid] : 2147483647;"
  end

  def test_mean_returns_float
    # MockBridge returns zero-initialized buffers, so the partial sums are 0
    assert_equal 0.0, Tilework[[1, 2]].mean
  end

  def test_argmax_generates_pair_reduction
    Tilework[[1.0, 2.0]].argmax
    assert_includes last_kernel_src, "GPU_SHARED int si[256];"
    assert_includes last_kernel_src, "sv[lid + s] > sv[lid]"
    Tilework[[1.0, 2.0]].argmin
    assert_includes last_kernel_src, "sv[lid + s] < sv[lid]"
  end

  # --- Device-resident reductions (device: true) ---

  # The host-returning path still runs one pass and folds the partials in Ruby
  # (no extra launches).
  def test_host_reduction_still_takes_a_single_pass
    Tilework::Buffer.new(:float32, 100_000).sum
    assert_equal 1, @bridge.dispatches.length
  end

  # device: true folds until one partial is left. 100_000 elements take three
  # passes (391 → 2 → 1) and never reach the host.
  def test_device_reduction_folds_all_the_way_down
    result = Tilework::Buffer.new(:float32, 100_000).sum(device: true)

    assert_instance_of Tilework::Buffer, result
    assert_equal 1, result.length
    assert_equal :float32, result.dtype
    assert_equal [391, 2, 1], @bridge.dispatches.map { |d| d[:grid].first }
  end

  def test_device_reduction_keeps_the_accumulator_dtype
    assert_equal :int32, Tilework::Buffer.new(:int32, 1_000).sum(device: true).dtype
    assert_equal :int32, Tilework::Buffer.new(:int32, 1_000).max(device: true).dtype
    # mean divides by the count at the end, so it is float32 (like the host path's fdiv)
    assert_equal :float32, Tilework::Buffer.new(:int32, 1_000).mean(device: true).dtype
  end

  # From the second pass on it folds (value, index) pairs, so a different
  # kernel carries the indices.
  def test_device_arg_reduction_carries_indices_between_passes
    result = Tilework::Buffer.new(:float32, 100_000).argmax(device: true)

    assert_equal 1, result.length
    assert_equal :int32, result.dtype, "argmax が返すのは添字"
    assert_equal %w[arg_max arg_max_pairs], @bridge.compilations.map { |c| c[:name] }.uniq
    assert_includes last_kernel_src, "si[lid] = gid < n ? idx[gid] : 0;"
    # Pass 1 takes (a, pval, pidx, n); later passes take one more index buffer
    assert_equal [4, 5, 5], @bridge.dispatches.map { |d| d[:args].length }
  end

  def test_small_buffers_need_only_one_pass
    Tilework::Buffer.new(:float32, 100).sum(device: true)
    assert_equal 1, @bridge.dispatches.length
  end

  def test_matmul_generates_kernel_and_output_shape
    a = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # 2x3
    b = Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # 3x2
    c = a.matmul(b)
    assert_equal [2, 2], c.shape
    assert_includes last_kernel_src, "GPU_KERNEL void matmul(GPU_ARGS,"
    assert_includes last_kernel_src, "acc += (float)a[row * k + t] * (float)b[t * n + col];"
  end

  def test_matmul_int_stays_int32
    a = Tilework::Buffer.from_matrix([[1, 2], [3, 4]])
    b = Tilework::Buffer.from_matrix([[1, 0], [0, 1]])
    assert_equal :int32, a.matmul(b).dtype
  end

  def test_matmul_requires_2d_buffers
    a = Tilework[[1.0, 2.0, 3.0]]
    b = Tilework::Buffer.from_matrix([[1.0], [2.0], [3.0]])
    assert_raises(ArgumentError) { a.matmul(b) }
  end

  def test_matmul_rejects_incompatible_inner_dims
    a = Tilework::Buffer.from_matrix([[1.0, 2.0]]) # 1x2
    b = Tilework::Buffer.from_matrix([[1.0, 2.0]]) # 1x2, needs 2xN
    assert_raises(ArgumentError) { a.matmul(b) }
  end

  def test_matmul_rejects_non_buffer
    a = Tilework::Buffer.from_matrix([[1.0]])
    assert_raises(TypeError) { a.matmul(2) }
  end

  def test_elementwise_op_requires_matching_shape_when_2d
    a = Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0, 4.0]]) # 2x2
    b = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0, 4.0]])   # 1x4, same length
    assert_raises(ArgumentError) { a + b }
  end

  def test_elementwise_op_preserves_2d_shape
    a = Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0, 4.0]])
    b = Tilework::Buffer.from_matrix([[5.0, 6.0], [7.0, 8.0]])
    assert_equal [2, 2], (a + b).shape
  end

  def test_dialect_backend_is_used_by_default
    a = Tilework[[1.0, 2.0]]
    b = Tilework[[3.0, 4.0]]
    a + b
    assert_includes last_kernel_src, "GPU_KERNEL"
  end
end
