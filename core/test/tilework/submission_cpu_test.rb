# frozen_string_literal: true

require_relative "../test_helper"

# Tilework.submit on a real backend (:cpu).
#
# Agreement with the synchronous eager path on every backend is checked by
# Tilework::Conformance's submit_* checks; this file holds the runtime
# behavior worth confirming on :cpu — deferred frees take effect and an
# exception unwinds cleanly. :cpu dispatches synchronously, so the number of
# waits does not drop here (measuring that is the device bench's job).
class SubmissionCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def teardown
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  def test_values_match_the_synchronous_eager_path
    a = Tilework::Buffer.from([1.0, 2.0, 3.0, 4.0])
    b = Tilework::Buffer.from([0.5, 0.5, 0.5, 0.5])

    expected = ((a + b) * 2.0).to_a
    actual = Tilework.submit { (a + b) * 2.0 }.wait.to_a

    assert_equal expected, actual
  end

  def test_nn_and_reductions_inside_a_submit_match_eager
    x = Tilework::Buffer.from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    expected_probs = Tilework::NN.softmax(x).to_a
    expected_sum = Tilework::Ops.reduce(:sum, x)

    probs, sum = Tilework.submit { [Tilework::NN.softmax(x), Tilework::Ops.reduce(:sum, x)] }.wait

    assert_equal expected_probs, probs.to_a
    assert_equal expected_sum, sum
  end

  def test_a_deferred_free_actually_releases_at_the_boundary
    a = Tilework::Buffer.from([1.0, 2.0])
    temp = nil

    job = Tilework.submit do
      temp = a * 2.0
      temp.free
      temp + 1.0
    end

    refute temp.freed?
    out = job.wait

    assert temp.freed?
    assert_equal [3.0, 5.0], out.to_a
  end

  def test_an_exception_leaves_no_submission_context_behind
    assert_raises(RuntimeError) { Tilework.submit { raise "boom" } }

    refute Tilework::Submission.active?
    assert_nil Tilework::Ops.context
  end

  def test_a_graph_plan_runs_unchanged_inside_a_submit
    a = Tilework::Buffer.from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = Tilework::Buffer.from([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [2, 3])
      g.output(x.matmul(g.param(w)) + 1.0)
    end.compile

    expected = plan.run(x: a).to_nested_a

    assert_equal expected, Tilework.submit { plan.run(x: a) }.wait.to_nested_a
  end
end
