# frozen_string_literal: true

require_relative "../test_helper"

# For host-constant / unroll tests (resolved through the block's binding)
DSL_TEST_DIRS = [[0, 1], [1, 0], [1, -1]].freeze
DSL_TEST_SCALE = 7
DSL_TEST_GAIN = 2.5

class DSLTest < Minitest::Test
  include BridgeTestSetup

  VEC_ADD = Tilework.kernel("vec_add") do |a, b, c, n|
    i = global_id_x
    c[i] = a[i] + b[i] if i < n
  end

  def float_args(n = 4)
    [Tilework::Buffer.new(:float32, n), Tilework::Buffer.new(:float32, n), Tilework::Buffer.new(:float32, n), n]
  end

  def test_vec_add_source
    src = VEC_ADD.source_for(float_args)
    assert_includes src, "GPU_KERNEL void vec_add(GPU_ARGS,"
    assert_includes src, "device const float* a," # read-only input is const
    assert_includes src, "device const float* b,"
    assert_includes src, "device float* c,"        # output is non-const
    assert_includes src, "constant int& n)"
    assert_includes src, "int i = GLOBAL_ID_X;"
    assert_includes src, "if ((i < n)) {"
    assert_includes src, "c[i] = (a[i] + b[i]);"
  end

  def test_signature_specialization_generates_int_variant
    args = [Tilework::Buffer.new(:int32, 4), Tilework::Buffer.new(:int32, 4), Tilework::Buffer.new(:int32, 4), 4]
    assert_includes VEC_ADD.source_for(args), "device const int* a,"
  end

  def test_call_launches_with_auto_grid_block
    n = 1000
    VEC_ADD.(Tilework::Buffer.new(:float32, n), Tilework::Buffer.new(:float32, n), Tilework::Buffer.new(:float32, n), n)
    dispatch = @bridge.dispatches.last
    assert_equal [4, 1, 1], dispatch[:grid] # ceil(1000/256)
    assert_equal [256, 1, 1], dispatch[:block]
  end

  def test_call_memoizes_per_signature
    2.times { VEC_ADD.(*float_args) }
    compiled_before = @bridge.compilations.size
    VEC_ADD.(*float_args)
    assert_equal compiled_before, @bridge.compilations.size
  end

  def test_call_without_buffer_raises
    assert_raises(Tilework::LaunchError) { VEC_ADD.(1, 2, 3, 4) }
  end

  def test_explicit_launch_supports_2d_grid
    k = Tilework.kernel("fill2d") do |out, w|
      out[global_id_y * w + global_id_x] = 1.0
    end
    k.launch(grid: [2, 2], block: [16, 16], args: [Tilework::Buffer.new(:float32, 1024), 32])
    dispatch = @bridge.dispatches.last
    assert_equal [2, 2, 1], dispatch[:grid]
    assert_includes @bridge.compilations.last[:src], "out[((GLOBAL_ID_Y * w) + GLOBAL_ID_X)] = 1.0f;"
  end

  def test_times_loop_and_while_loop
    k = Tilework.kernel("loops") do |x, y, n|
      i = global_id_x
      if i < n
        acc = 0.0
        n.times { |j| acc = acc + x[j] }
        m = 0
        while m < 3
          acc = acc * 2.0
          m = m + 1
        end
        y[i] = acc
      end
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4])
    assert_includes src, "for (int j = 0; j < n; j++) {"
    assert_includes src, "while ((m < 3)) {"
    assert_includes src, "float acc = 0.0f;"
    assert_includes src, "int m = 0;"
  end

  def test_range_each_loop
    k = Tilework.kernel("ranges") do |x, n|
      (1...n).each { |j| x[j] = x[j - 1] }
      (0..2).each { |j| x[j] = 0.0 }
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), 4])
    assert_includes src, "for (int j = 1; j < n; j++) {"
    assert_includes src, "for (int j = 0; j <= 2; j++) {"
  end

  def test_math_functions_and_abs
    k = Tilework.kernel("mathy") do |x, y|
      i = global_id_x
      y[i] = Math.pow(Math.exp(x[i]), 2.0) + x[i].abs
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4)])
    assert_includes src, "pow((float)(exp((float)(x[i]))), (float)(2.0f))"
    assert_includes src, "fabs(x[i])"
  end

  def test_int_buffer_uses_int_abs
    k = Tilework.kernel("iabs") do |x|
      x[global_id_x] = x[global_id_x].abs
    end
    assert_includes k.source_for([Tilework::Buffer.new(:int32, 4)]), "abs(x[GLOBAL_ID_X])"
  end

  def test_type_promotion_and_mismatched_store_is_cast
    k = Tilework.kernel("promote") do |x, out|
      i = global_id_x
      v = x[i] * 2 # float * int -> float
      out[i] = v
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:int32, 4)])
    assert_includes src, "float v = (x[i] * 2);"
    assert_includes src, "out[i] = (int)(v);"
  end

  def test_unless_and_ternary_and_unary
    k = Tilework.kernel("misc") do |x, n|
      i = global_id_x
      unless i >= n
        x[i] = i % 2 == 0 ? -x[i] : x[i]
      end
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), 4])
    assert_includes src, "if (!((i >= n))) {"
    assert_includes src, "x[i] = (((i % 2) == 0) ? (-x[i]) : x[i]);"
  end

  def test_unsupported_syntax_raises_with_line_number
    k = Tilework.kernel("bad") do |x|
      puts x # around line 40: method calls are unsupported
    end
    error = assert_raises(Tilework::CompileError) { k.source_for([Tilework::Buffer.new(:float32, 4)]) }
    assert_match(/at line \d+/, error.message)
  end

  def test_unknown_identifier_raises
    k = Tilework.kernel("bad2") do |x|
      x[global_id_x] = mystery_value
    end
    error = assert_raises(Tilework::CompileError) { k.source_for([Tilework::Buffer.new(:float32, 4)]) }
    assert_includes error.message, "mystery_value"
  end

  def test_float_to_int_reassignment_raises
    k = Tilework.kernel("bad3") do |x|
      v = 1
      v = x[0]
      x[0] = v
    end
    error = assert_raises(Tilework::CompileError) { k.source_for([Tilework::Buffer.new(:float32, 4)]) }
    assert_includes error.message, "reassigning float value to int variable"
  end

  def test_wrong_arity_raises
    error = assert_raises(Tilework::CompileError) { VEC_ADD.source_for([Tilework::Buffer.new(:float32, 4)]) }
    assert_includes error.message, "takes 4 parameters"
  end

  def test_kernel_without_block_raises
    assert_raises(ArgumentError) { Tilework.kernel("nope") }
  end

  def test_out_param_is_auto_allocated_and_returned
    vec_add = Tilework.kernel("vec_add_out", out: 2) do |a, b, c, n|
      i = global_id_x
      c[i] = a[i] + b[i] if i < n
    end

    n = 100
    out = vec_add.(Tilework::Buffer.new(:float32, n), Tilework::Buffer.new(:float32, n), n)
    assert_instance_of Tilework::Buffer, out
    assert_equal n, out.length
    assert_equal :float32, out.dtype
    # The output buffer is inserted automatically at dispatch argument position 2
    assert_equal out.__handle__, @bridge.dispatches.last[:args][2]
  end

  def test_out_param_dtype_matches_int32_inputs
    vec_add = Tilework.kernel("vec_add_out_int", out: 2) do |a, b, c, n|
      i = global_id_x
      c[i] = a[i] + b[i] if i < n
    end
    out = vec_add.(Tilework::Buffer.new(:int32, 4), Tilework::Buffer.new(:int32, 4), 4)
    assert_equal :int32, out.dtype
  end

  def test_out_index_out_of_range_raises
    vec_add = Tilework.kernel("vec_add_out_range", out: 5) do |a, b, c, n|
      i = global_id_x
      c[i] = a[i] + b[i] if i < n
    end
    assert_raises(Tilework::LaunchError) { vec_add.(Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4) }
  end

  def test_out_dtype_overrides_inference
    # Inputs are all int32, but the output should be float32
    k = Tilework.kernel("to_float", out: 1, out_dtype: :float32) do |a, out, n|
      i = global_id_x
      out[i] = a[i] if i < n
    end
    out = k.(Tilework::Buffer.new(:int32, 4), 4)
    assert_equal :float32, out.dtype
  end

  def test_over_sets_launch_size_and_output_size
    # The longest Buffer is board (16), but we want cand (3) threads
    k = Tilework.kernel("pick", out: 2, out_dtype: :float32) do |board, cand, out, n|
      i = global_id_x
      out[i] = board[cand[i]] if i < n
    end
    board = Tilework::Buffer.new(:int32, 16)
    cand = Tilework::Buffer.new(:int32, 3)
    out = k.(board, cand, 3, over: 3)
    assert_equal 3, out.length # over:'s 3, not the longest Buffer's 16
    dispatch = @bridge.dispatches.last
    assert_equal [1, 1, 1], dispatch[:grid] # ceil(3/256) = 1
  end

  def test_over_without_buffers_still_launches
    k = Tilework.kernel("fill", out: 0, out_dtype: :int32) do |out, v, n|
      i = global_id_x
      out[i] = v if i < n
    end
    out = k.(7, 5, over: 5)
    assert_equal 5, out.length
    assert_equal :int32, out.dtype
  end

  def test_explicit_launch_still_requires_full_args_even_with_out
    vec_add = Tilework.kernel("vec_add_out_launch", out: 2) do |a, b, c, n|
      i = global_id_x
      c[i] = a[i] + b[i] if i < n
    end
    c = Tilework::Buffer.new(:float32, 4)
    vec_add.launch(grid: 1, block: 4, args: [Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), c, 4])
    assert_equal c.__handle__, @bridge.dispatches.last[:args][2]
  end

  def test_long_and_chain_keeps_all_terms
    # a && b && c && d puts every operand in one AND node; 3+ operands must not be dropped.
    k = Tilework.kernel("chain") do |x, n|
      i = global_id_x
      x[i] = 1.0 if i >= 0 && i < n && i != 3 && i < 100
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), 4])
    assert_includes src, "if (((i >= 0) && (i < n) && (i != 3) && (i < 100)))"
  end

  def test_unroll_numeric_array_inlines_compile_time_constants
    k = Tilework.kernel("scale_unroll") do |a, out, n|
      i = global_id_x
      if i < n
        unroll([1, 2, 4]) do |w|
          out[i] = a[i] * w
        end
      end
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4])
    assert_includes src, "out[i] = (a[i] * 1);"
    assert_includes src, "out[i] = (a[i] * 2);"
    assert_includes src, "out[i] = (a[i] * 4);"
    refute_includes src, "for (" # unroll emits no run-time loop
  end

  def test_unroll_destructures_tuples_into_constants
    k = Tilework.kernel("dirs") do |board, out, size|
      i = global_id_x
      unroll(DSL_TEST_DIRS) do |dr, dc|
        out[i] = board[(i + dr) * size + (i + dc)]
      end
    end
    src = k.source_for([Tilework::Buffer.new(:int32, 4), Tilework::Buffer.new(:int32, 4), 4])
    assert_includes src, "board[(((i + 0) * size) + (i + 1))]"
    assert_includes src, "board[(((i + 1) * size) + (i + 0))]"
    assert_includes src, "board[(((i + 1) * size) + (i + -1))]"
  end

  def test_unroll_symbol_aliases_params_and_locals
    k = Tilework.kernel("alias") do |a, out, n, lo, hi|
      i = global_id_x
      if i < n
        acc = 0
        unroll([[:lo, :acc], [:hi, :acc]]) do |bound, sink|
          sink = sink + (a[i] > bound ? 1 : 0)
        end
        out[i] = acc
      end
    end
    src = k.source_for([Tilework::Buffer.new(:int32, 4), Tilework::Buffer.new(:int32, 4), 4, 1, 9])
    # :lo / :hi expand as aliases of kernel arguments, :acc as an alias of a local variable
    assert_includes src, "acc = (acc + ((a[i] > lo) ? 1 : 0));"
    assert_includes src, "acc = (acc + ((a[i] > hi) ? 1 : 0));"
  end

  def test_host_constant_is_spliced_as_literal
    k = Tilework.kernel("consts") do |a, out, n|
      i = global_id_x
      out[i] = a[i] * DSL_TEST_SCALE + DSL_TEST_GAIN if i < n
    end
    src = k.source_for([Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4])
    assert_includes src, "(a[i] * 7)"
    assert_includes src, "2.5f"
  end

  def test_unroll_over_non_array_raises
    k = Tilework.kernel("bad_unroll") do |a, n|
      i = global_id_x
      unroll(DSL_TEST_SCALE) { |w| a[i] = w }
    end
    assert_raises(Tilework::CompileError) { k.source_for([Tilework::Buffer.new(:float32, 4), 4]) }
  end

  def test_unroll_symbol_not_naming_a_variable_raises
    k = Tilework.kernel("bad_alias") do |a, n|
      i = global_id_x
      unroll([:nonexistent]) { |x| a[i] = x }
    end
    error = assert_raises(Tilework::CompileError) { k.source_for([Tilework::Buffer.new(:float32, 4), 4]) }
    assert_includes error.message, "nonexistent"
  end

  def test_assigning_to_numeric_unrolled_constant_raises
    k = Tilework.kernel("bad_const_assign") do |a, n|
      i = global_id_x
      unroll([1, 2]) do |w|
        w = w + 1
        a[i] = w
      end
    end
    error = assert_raises(Tilework::CompileError) { k.source_for([Tilework::Buffer.new(:float32, 4), 4]) }
    assert_includes error.message, "unrolled constant"
  end

  def test_two_index_read_and_write_use_flat_row_major
    k = Tilework.kernel("mat2d", out: 1) do |mat, out|
      r = global_id_y
      c = global_id_x
      out[r, c] = mat[r, c] * 2.0
    end
    src = k.source_for([Tilework::Buffer.from_matrix([[1.0, 2.0], [3.0, 4.0]]),
                         Tilework::Buffer.from_matrix([[0.0, 0.0], [0.0, 0.0]])])
    assert_includes src, "out[((r) * 2) + (c)] = (mat[((r) * 2) + (c)] * 2.0f);"
  end

  # --- Automatic bounds guard for the launch domain ---
  #
  # A grid can only launch whole blocks (Metal's dispatchThreadgroups and
  # CUDA's cuLaunchKernel alike), so extra threads always run at the edge. The
  # generator inserts a guard so a user kernel cannot write out of range even
  # if it forgets one.

  def test_2d_domain_guard_is_emitted
    k = Tilework.kernel("guarded2d") { |mat, out| out[global_id_y, global_id_x] = mat[global_id_y, global_id_x] }
    m = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    src = k.source_for([m, m], over: [3, 2])
    assert_includes src, "if (GLOBAL_ID_X >= 3 || GLOBAL_ID_Y >= 2) return;"
  end

  def test_1d_domain_guard_omits_the_y_check
    k = Tilework.kernel("guarded1d") { |a, out| out[global_id_x] = a[global_id_x] }
    b = Tilework[[1.0, 2.0, 3.0, 4.0, 5.0]]
    src = k.source_for([b, b], over: 5)
    assert_includes src, "if (GLOBAL_ID_X >= 5) return;"
    refute_includes src, "GLOBAL_ID_Y"
  end

  # An explicit launch(grid:, block:) means the caller fully decides the launch
  # shape, so no domain is guessed and no guard is inserted.
  def test_explicit_launch_is_not_guarded
    k = Tilework.kernel("unguarded") { |a, out| out[global_id_x] = a[global_id_x] }
    b = Tilework[[1.0, 2.0]]
    src = k.source_for([b, b], over: nil)
    assert_includes src, "if (GLOBAL_ID_X >= 2) return;" # inferred when over: is omitted

    # Drives the Transpiler directly to check domain: nil
    sig = Tilework::DSL.signature_for([b, b])
    raw = Tilework::DSL::Transpiler.new(k.instance_variable_get(:@block), sig, "unguarded").transpile
    refute_includes raw, "return;"
  end

  # The domain is baked into the source as the guard's literal, so like shape
  # it must be part of the cache key.
  def test_cache_key_includes_the_domain
    k = Tilework.kernel("domkey", out: 1) { |a, out| out[global_id_x] = a[global_id_x] }
    k.(Tilework[[1.0, 2.0, 3.0, 4.0]], over: 4)
    k.(Tilework[[1.0, 2.0, 3.0, 4.0]], over: 2)
    assert_equal 2, k.instance_variable_get(:@compiled).size
  end

  def test_two_index_uses_each_buffers_own_cols
    # mat has 2 columns, out has 3. Each cols is baked in as a compile-time
    # constant from that buffer's own shape.
    k = Tilework.kernel("mixed_cols") do |mat, out|
      out[0, 0] = mat[0, 0]
    end
    src = k.source_for([Tilework::Buffer.from_matrix([[1.0, 2.0]]),
                         Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0]])])
    assert_includes src, "out[((0) * 3) + (0)] = mat[((0) * 2) + (0)];"
  end

  def test_two_index_on_1d_buffer_raises
    k = Tilework.kernel("bad2d") do |a|
      a[0, 0] = 1.0
    end
    error = assert_raises(Tilework::CompileError) { k.source_for([Tilework::Buffer.new(:float32, 4)]) }
    assert_includes error.message, "not a 2D buffer"
  end

  def test_three_or_more_indices_raises
    k = Tilework.kernel("bad3d") do |a|
      a[0, 0, 0] = 1.0
    end
    error = assert_raises(Tilework::CompileError) do
      k.source_for([Tilework::Buffer.from_matrix([[1.0, 2.0]])])
    end
    assert_includes error.message, "1 or 2 indices"
  end

  def test_rows_and_cols_sugar_bake_shape_as_constants
    k = Tilework.kernel("dims") do |mat, out|
      out[0] = mat.rows
      out[1] = mat.cols
    end
    src = k.source_for([Tilework::Buffer.from_matrix([[1, 2, 3], [4, 5, 6]]), Tilework::Buffer.new(:int32, 2)])
    assert_includes src, "out[0] = 2;"
    assert_includes src, "out[1] = 3;"
  end

  def test_rows_cols_on_1d_buffer_raises
    k = Tilework.kernel("bad_dims") do |a, out|
      out[0] = a.rows
    end
    error = assert_raises(Tilework::CompileError) do
      k.source_for([Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:int32, 1)])
    end
    assert_includes error.message, "not a 2D buffer"
  end

  def test_call_auto_infers_2d_domain_from_matrix_buffer_and_shapes_output
    k = Tilework.kernel("scale2d", out: 1) do |mat, out, factor|
      c = global_id_x
      r = global_id_y
      if c < mat.cols && r < mat.rows
        out[r, c] = mat[r, c] * factor
      end
    end
    mat = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) # shape [2, 3]
    out = k.(mat, 2.0)
    assert_equal [2, 3], out.shape
    dispatch = @bridge.dispatches.last
    assert_equal [1, 1, 1], dispatch[:grid]   # ceil(3/16), ceil(2/16)
    assert_equal [16, 16, 1], dispatch[:block]
  end

  def test_call_over_with_explicit_2d_dims
    k = Tilework.kernel("explicit2d", out: 0) do |out, w, h|
      c = global_id_x
      r = global_id_y
      out[r, c] = 1.0 if c < w && r < h
    end
    out = k.(3, 5, over: [3, 5]) # [cols, rows]
    assert_equal [5, 3], out.shape
  end

  def test_mismatched_2d_buffer_shapes_raise_without_over
    k = Tilework.kernel("ambiguous", out: 2) do |a, b, out|
      out[0, 0] = a[0, 0] + b[0, 0]
    end
    a = Tilework::Buffer.from_matrix([[1.0, 2.0]])       # shape [1, 2]
    b = Tilework::Buffer.from_matrix([[1.0], [2.0]])     # shape [2, 1]
    assert_raises(Tilework::LaunchError) { k.(a, b) }
  end

  def test_over_with_invalid_shape_raises
    k = Tilework.kernel("bad_over") do |n|
      i = global_id_x
    end
    assert_raises(Tilework::LaunchError) { k.(4, over: [1, 2, 3]) }
    assert_raises(Tilework::LaunchError) { k.(4, over: "nope") }
  end

  def test_dialect_backend_is_used_by_default
    k = Tilework.kernel("dialect_still_used") { |a, n| a[0] = 1.0 if 0 < n }
    k.(Tilework::Buffer.new(:float32, 4), 4)
    assert_includes @bridge.compilations.last[:src], "GPU_KERNEL"
  end
end
