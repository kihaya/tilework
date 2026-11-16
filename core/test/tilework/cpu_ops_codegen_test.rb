# frozen_string_literal: true

require_relative "../test_helper"
require "tilework/backend/cpu/ops_codegen"

# Unit tests for the :cpu backend's Ops code generation. ops_cpu_test.rb runs
# the ops on Tilework.backend = :cpu for correctness; this file checks the
# shape of the generated Ruby source and that the Proc it evals to works.
class CpuOpsCodegenTest < Minitest::Test
  def test_buffer_binary_arithmetic
    src = Tilework::CPU::OpsCodegen.buffer_binary("ew", a_dtype: :float32, b_dtype: :float32,
                                              out_dtype: :float32, compute_dtype: :float32,
                                              &Tilework::CPU::OpsCodegen.arithmetic_expr(:+))
    assert_includes src, "arg_kinds: [:float32_buffer, :float32_buffer, :float32_buffer, :int32_scalar]"
    pipeline = eval(src)
    bufs = [[1.0, 2.0], [10.0, 20.0], [0.0, 0.0]]
    scalars = [nil, nil, nil, 2]
    2.times { |i| pipeline[:proc].call(bufs, scalars, i, 0, 0, i, 0, 0, 0, 0, 0) }
    assert_equal [11.0, 22.0], bufs[2]
  end

  def test_compare_returns_int_mask
    src = Tilework::CPU::OpsCodegen.buffer_binary("ew", a_dtype: :float32, b_dtype: :float32,
                                              out_dtype: :int32, compute_dtype: :float32,
                                              &Tilework::CPU::OpsCodegen.compare_expr(:<))
    pipeline = eval(src)
    bufs = [[1.0, 5.0], [2.0, 3.0], [0, 0]]
    scalars = [nil, nil, nil, 2]
    2.times { |i| pipeline[:proc].call(bufs, scalars, i, 0, 0, i, 0, 0, 0, 0, 0) }
    assert_equal [1, 0], bufs[2]
  end

  def test_binary_fn_max_min
    max_src = Tilework::CPU::OpsCodegen.buffer_binary("ew", a_dtype: :float32, b_dtype: :float32,
                                                  out_dtype: :float32, compute_dtype: :float32,
                                                  &Tilework::CPU::OpsCodegen.binary_fn_expr("max"))
    pipeline = eval(max_src)
    bufs = [[1.0, 5.0], [2.0, 3.0], [0.0, 0.0]]
    scalars = [nil, nil, nil, 2]
    2.times { |i| pipeline[:proc].call(bufs, scalars, i, 0, 0, i, 0, 0, 0, 0, 0) }
    assert_equal [2.0, 5.0], bufs[2]
  end

  def test_matmul
    src = Tilework::CPU::OpsCodegen.matmul(a_dtype: :float32, b_dtype: :float32, dtype: :float32)
    pipeline = eval(src)
    # a: 1x2 [1,2], b: 2x1 [3,4] => c: 1x1 = 1*3 + 2*4 = 11
    bufs = [[1.0, 2.0], [3.0, 4.0], [0.0]]
    scalars = [nil, nil, nil, 1, 2, 1]
    pipeline[:proc].call(bufs, scalars, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert_equal [11.0], bufs[2]
  end

  def test_clamp
    src = Tilework::CPU::OpsCodegen.clamp(a_dtype: :float32, dtype: :float32)
    pipeline = eval(src)
    bufs = [[-1.0, 0.5, 5.0], nil, nil, [0.0, 0.0, 0.0]]
    scalars = [nil, 0.0, 2.0, nil, 3]
    3.times { |i| pipeline[:proc].call(bufs, scalars, i, 0, 0, i, 0, 0, 0, 0, 0) }
    assert_equal [0.0, 0.5, 2.0], bufs[3]
  end
end
