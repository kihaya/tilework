# frozen_string_literal: true

require_relative "../test_helper"

# The op table (Graph::Op) itself. The table exists so that forgetting a
# declaration when adding an op raises, so this checks that the gate is shut.
class GraphOpTest < Minitest::Test
  include BridgeTestSetup

  Op = Tilework::Graph::Op

  def test_every_op_declares_an_execution_method_that_plan_implements
    Op.names.each do |name|
      run = Op.fetch(name).run
      assert Tilework::Graph::Plan.private_method_defined?(run),
             "op #{name.inspect} names #{run.inspect}, which Plan does not implement"
      assert_equal 3, Tilework::Graph::Plan.instance_method(run).arity,
                   "#{run.inspect} must take (node, inputs, bindings)"
    end
  end

  def test_unknown_ops_are_rejected_when_fetched_and_when_recorded
    error = assert_raises(Tilework::Error) { Op.fetch(:no_such_op) }
    assert_includes error.message, "unknown graph op :no_such_op"

    Tilework::Graph.build do |g|
      x = g.input(:x, [4])
      error = assert_raises(Tilework::Error) do
        g.record(:no_such_op, inputs: [x.node], dtype: :float32, shape: [4])
      end
      assert_includes error.message, "unknown graph op :no_such_op"
    end
  end

  def test_defining_the_same_op_twice_is_an_error
    error = assert_raises(Tilework::Error) { Op.define(:binary, run: :run_binary) }
    assert_includes error.message, "already defined"
  end

  # Some properties depend on the node or the capabilities. Rows written as
  # booleans and rows written as Procs must read the same way to callers.
  def test_node_dependent_properties_are_asked_per_node
    Tilework::Graph.build do |g|
      a = g.input(:a, [2, 3, 4])
      b = g.input(:b, [2, 4, 3])
      plain = g.input(:m, [3, 4]).matmul(g.input(:n, [4, 3])).node
      batched = a.matmul(b).node

      assert plain.spec.placeable?(plain)
      # Batched matmul allocates differently with and without a delegate, so
      # its output cannot be placed.
      refute batched.spec.placeable?(batched)
      # Without gemm_library the context takes over freeing the per-batch
      # temporaries (#86), so no synchronous execution is needed.
      refute batched.spec.synchronous?(batched)
      refute plain.spec.synchronous?(plain)
    end
  end

  def test_ops_that_do_not_own_their_buffer_are_the_ones_the_graph_did_not_allocate
    refute Op.fetch(:input).owned?
    refute Op.fetch(:const).owned?
    %i[state_init state_write state_merge].each do |name|
      refute Op.fetch(name).owned?, "#{name} points at the state's own buffer"
    end
    assert Op.fetch(:binary).owned?
  end

  def test_labels_come_from_the_table
    Tilework::Graph.build do |g|
      x = g.input(:x, [4])
      assert_equal "input :x", x.node.label
      assert_equal "* 2.0", (x * 2.0).node.label
      assert_equal "exp", x.exp.node.label
      assert_equal "sum", x.sum.node.label
      # An op without a label in the table prints its op name.
      assert_equal "concat", g.concat(x, x).node.label
    end
  end
end
