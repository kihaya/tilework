# frozen_string_literal: true

require_relative "../test_helper"
require "tilework/backend/cpu/dsl_codegen"

# Unit tests for the :cpu backend's DSL code generation. dsl_cpu_test.rb runs
# the kernels on Tilework.backend = :cpu for correctness; this file only checks
# the shape of the generated Ruby source.
class CpuDslCodegenTest < Minitest::Test
  include BridgeTestSetup

  def transpile(block, args, **kwargs)
    Tilework::CPU.transpile(block, args, **kwargs)
  end

  def test_vec_add_generates_arg_kinds_and_preamble
    vec_add = proc do |a, b, c, n|
      i = global_id_x
      c[i] = a[i] + b[i] if i < n
    end
    args = [Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4]
    src = transpile(vec_add, args, name: "vec_add")

    assert_includes src, "arg_kinds: [:float32_buffer, :float32_buffer, :float32_buffer, :int32_scalar]"
    assert_includes src, "a = bufs[0]"
    assert_includes src, "b = bufs[1]"
    assert_includes src, "c = bufs[2]"
    assert_includes src, "n = scalars[3]"
    assert_includes src, "i = gx"
    assert_includes src, "if (i < n)"
    assert_includes src, "c[i] = (a[i] + b[i])"

    pipeline = eval(src)
    assert_kind_of Proc, pipeline[:proc]
  end

  def test_int_buffer_uses_int32_arg_kind
    k = proc { |a, out, n| out[0] = a[0] if 0 < n }
    args = [Tilework::Buffer.new(:int32, 4), Tilework::Buffer.new(:int32, 4), 4]
    src = transpile(k, args, name: "k")
    assert_includes src, "arg_kinds: [:int32_buffer, :int32_buffer, :int32_scalar]"
  end

  def test_2d_indexing_bakes_cols_as_literal
    k = proc do |mat, out, factor|
      c = global_id_x
      r = global_id_y
      out[r, c] = mat[r, c] * factor if c < mat.cols && r < mat.rows
    end
    mat = Tilework::Buffer.from_matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = Tilework::Buffer.new(:float32, 6, shape: [2, 3])
    src = transpile(k, [mat, out, 2.0], name: "scale2d")
    assert_includes src, "out[((r) * 3) + (c)] = (mat[((r) * 3) + (c)] * factor)"
  end

  def test_unroll_expands_at_transpile_time
    k = proc do |a, out, n|
      i = global_id_x
      if i < n
        total = 0.0
        unroll([1.0, 2.0]) { |w| total = total + a[i] * w }
        out[i] = total
      end
    end
    args = [Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4]
    src = transpile(k, args, name: "weighted")
    assert_includes src, "total = (total + (a[i] * 1.0))"
    assert_includes src, "total = (total + (a[i] * 2.0))"
  end

  def test_math_pow_uses_ruby_exponent_operator
    k = proc { |a, out, n| out[0] = Math.pow(a[0], 2.0) if 0 < n }
    args = [Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4]
    src = transpile(k, args, name: "pow_k")
    assert_includes src, "**"
    refute_includes src, "Math.pow"
  end

  def test_requires_a_block_defined_in_a_readable_source_file
    inline_block = eval("proc { |a, out, n| out[0] = a[0] if 0 < n }") # rubocop:disable Security/Eval
    args = [Tilework::Buffer.new(:float32, 4), Tilework::Buffer.new(:float32, 4), 4]
    error = assert_raises(Tilework::CompileError) { transpile(inline_block, args, name: "k") }
    assert_includes error.message, "readable source file"
  end
end
