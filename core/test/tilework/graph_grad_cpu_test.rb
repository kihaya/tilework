# frozen_string_literal: true

require_relative "../test_helper"

# g.grad (#22 Step 1) on :cpu. **Whether gradient values are right** is checked
# by Tilework::Conformance's graph_gradients_match_finite_differences and
# graph_a_training_step_reduces_the_loss. This file covers what those two
# cannot tell: non-differentiable ops fail with a reason, what comes back when
# no gradient flows, and that second derivatives work.
class GraphGradCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def nd(shape, values)
    Tilework::Buffer.new(:float32, values.length, shape: shape).tap { |b| b.write(values) }
  end

  def test_grad_requires_a_scalar_loss
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build(differentiable: true) do |g|
        x = g.input(:x, [2, 3])
        g.grad(x * 2.0, wrt: [x])
      end
    end
    assert_match(/scalar loss/, error.message)
  end

  # Gradients are taken with respect to the read value, not the state itself;
  # getting that wrong must say what to do next.
  def test_grad_points_at_read_when_given_a_state_ref
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build(differentiable: true) do |g|
        ref = g.state(nd([2], [1.0, 2.0]), name: :w)
        loss = g.read(ref).sum
        g.grad(loss, wrt: [ref])
      end
    end
    assert_match(/g\.read\(ref\)/, error.message)
  end

  def test_opaque_says_why_it_has_no_gradient
    opaque = assert_raises(Tilework::Error) do
      Tilework::Graph.build(differentiable: true) do |g|
        x = g.input(:x, [4])
        y = g.opaque(x, shape: [4], name: "mystery") { |b| b }
        g.grad(y.sum, wrt: [x])
      end
    end
    assert_match(/cannot see inside/, opaque.message)
  end

  # gather's vjp is scatter-add. The point is that **drawing the same row
  # twice doubles its gradient** (it accumulates rather than overwrites).
  def test_gather_accumulates_repeated_indices
    ids = Tilework::Buffer.new(:int32, 4, shape: [4])
    ids.write([0, 2, 2, 4])

    plan = Tilework::Graph.build(differentiable: true) do |g|
      table = g.input(:t, [5, 2])
      rows = g.gather(table, g.param(ids))
      g.output(g.grad((rows * rows).sum, wrt: [table])[0])
    end.compile

    values = (1..10).map(&:to_f)
    counts = [1, 0, 2, 0, 1] # row 2 is drawn twice
    want = values.each_with_index.map { |v, i| 2 * v * counts[i / 2] }
    assert_equal want, plan.run(t: nd([5, 2], values)).to_a
  end

  # A high-level op without a decomposition (gelu_exact: erf is not a
  # primitive) stays :nn even under differentiable: true. It must fail by name
  # rather than be silently swapped for another approximation.
  def test_a_high_level_op_without_a_decomposition_raises
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build(differentiable: true) do |g|
        x = g.input(:x, [2, 2])
        g.grad(Tilework::NN.gelu(x, exact: true).sum, wrt: [x])
      end
    end
    assert_match(/gelu_exact/, error.message)
  end

  # Differentiating a function with no hand-written vjp while differentiable is
  # false must say what to do (a :nn node cannot fall back to primitive vjps).
  def test_grad_on_a_plain_graph_names_the_flag
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build do |g|
        x = g.input(:x, [2, 4])
        g.grad(Tilework::NN.silu(x).sum, wrt: [x])
      end
    end
    assert_match(/differentiable: true/, error.message)
  end

  # Conversely, a function with a hand-written vjp is differentiable **without
  # differentiable: true** (it never needs decomposing).
  def test_a_hand_written_vjp_works_without_the_flag
    plan = Tilework::Graph.build do |g|
      x = g.input(:x, [1, 3])
      probs = Tilework::NN.softmax(x)
      g.output(g.grad((probs * probs).sum, wrt: [x])[0])
    end.compile

    assert_equal 1, plan.nodes.count { |n| n.op == :nn }
    # With softmax output p, d/dx Σp² = 2p(p - Σp²).
    values = plan.run(x: nd([1, 3], [0.0, 1.0, 2.0])).to_a
    p_row = Tilework::NN.softmax(nd([1, 3], [0.0, 1.0, 2.0])).to_a
    sq = p_row.sum { |v| v * v }
    p_row.each_with_index { |v, i| assert_in_delta 2 * v * (v - sq), values[i], 1e-6 }
  end

  # The gradient of a parameter the loss does not depend on is zero, not nil,
  # so optimizers need no "no gradient" branch.
  def test_an_unused_parameter_gets_a_zero_gradient
    plan = Tilework::Graph.build(differentiable: true) do |g|
      x = g.input(:x, [3])
      unused = g.input(:u, [2])
      dx, du = g.grad((x * 2.0).sum, wrt: [x, unused])
      g.output(dx)
      g.output(du)
    end.compile

    dx, du = plan.run(x: nd([3], [1.0, 2.0, 3.0]), u: nd([2], [5.0, 6.0]))
    assert_equal [2.0, 2.0, 2.0], dx.to_a
    assert_equal [0.0, 0.0], du.to_a
  end

  # Gradients are ordinary nodes written in primitives, so they differentiate
  # again: d/dx x^3 = 3x^2, d2/dx2 x^3 = 6x.
  def test_gradients_are_themselves_differentiable
    plan = Tilework::Graph.build(differentiable: true) do |g|
      x = g.input(:x, [3])
      first = g.grad((x * x * x).sum, wrt: [x])[0]
      g.output(first)
      g.output(g.grad(first.sum, wrt: [x])[0])
    end.compile

    first, second = plan.run(x: nd([3], [1.0, 2.0, 3.0])).map(&:to_a)
    assert_equal [3.0, 12.0, 27.0], first
    assert_equal [6.0, 12.0, 18.0], second
  end

  # Redrawing random numbers inside a checkpoint gives **the same sequence**.
  # The Philox key comes from (plan seed, the node's static index), and the
  # index carries over to the copy, so recomputation does not change values —
  # #19's choice not to make RNG mutable state is what makes checkpointing safe.
  def test_checkpoint_redraws_the_same_random_numbers
    build = lambda do |use_checkpoint|
      Tilework::Graph.build(differentiable: true) do |g|
        x = g.input(:x, [4])
        body = -> { (x * g.random([4])).exp }
        h = use_checkpoint ? g.checkpoint { body.call } : body.call
        g.output(g.grad(h.sum, wrt: [x])[0])
      end.compile
    end

    x = nd([4], [0.25, -0.5, 0.75, 1.0])
    assert_equal build.call(false).run(x: x, seed: 99).to_a,
                 build.call(true).run(x: x, seed: 99).to_a
  end

  # checkpoint only **removes backward from the readers** of values made inside
  # the block; values used from outside the region are untouched.
  def test_checkpoint_leaves_the_gradient_unchanged
    build = lambda do |use_checkpoint|
      Tilework::Graph.build(differentiable: true) do |g|
        x = g.input(:x, [2, 3])
        body = -> { Tilework::NN.gelu(x * 2.0).exp }
        h = use_checkpoint ? g.checkpoint { body.call } : body.call
        g.output(g.grad((h * h).sum, wrt: [x])[0])
      end.compile
    end

    x = nd([2, 3], [0.5, -1.0, 0.25, 0.75, -0.5, 1.5])
    assert_equal build.call(false).run(x: x).to_a, build.call(true).run(x: x).to_a
  end

  # An optimizer turns parameters into state with attach, then writes updates
  # with step. Calling them out of order must say what to do.
  def test_optimizer_requires_attach_before_step
    opt = Tilework::Graph::SGD.new({ w: nd([2], [1.0, 2.0]) }, lr: 0.1)
    error = assert_raises(Tilework::Error) do
      Tilework::Graph.build(differentiable: true) { |g| opt.step(g.input(:x, [1])) }
    end
    assert_match(/attach/, error.message)
  end

  # Adam's moments live across plan runs (they are mutable state). Later steps
  # moving by a different amount than the first shows the state takes effect.
  def test_adam_moments_persist_across_runs
    param = nd([2], [1.0, 1.0])
    opt = Tilework::Graph::Adam.new({ w: param }, lr: 0.1)
    plan = Tilework::Graph.build(differentiable: true) do |g|
      p = opt.attach(g)
      g.output((p[:w] * p[:w]).sum)
      opt.step((p[:w] * p[:w]).sum)
    end.compile

    plan.run
    after_first = param.to_a.dup
    plan.run
    after_second = param.to_a

    # The first step is almost exactly lr; the second differs because the correction changes.
    assert_in_delta 0.9, after_first[0], 1e-3
    refute_in_delta after_first[0] - after_second[0], 0.1, 1e-4
  ensure
    opt.free!
  end

  # A value with several consumers receives several cotangents. Forgetting to
  # sum them returns only one path's share, so check the sum directly.
  def test_cotangents_from_several_consumers_are_summed
    plan = Tilework::Graph.build(differentiable: true) do |g|
      x = g.input(:x, [2])
      g.output(g.grad(((x * 2.0) + (x * 3.0) + (x * x)).sum, wrt: [x])[0])
    end.compile

    assert_equal [7.0, 9.0], plan.run(x: nd([2], [1.0, 2.0])).to_a
  end
end
