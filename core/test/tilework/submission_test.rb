# frozen_string_literal: true

require "test_helper"

# Tilework.submit (#71) — an explicit eager asynchronous boundary.
#
# Each claim (launches do not wait, one synchronization, nothing changes
# outside the boundary) is visible through MockBridge's dispatches[:wait] and
# synchronize_count, the same style as kernel_test.rb / graph_test.rb.
class SubmissionTest < Minitest::Test
  include BridgeTestSetup

  def waits = @bridge.dispatches.map { |d| d[:wait] }

  # Buffer.from also synchronizes for its host transfer, so count only the
  # synchronizations submit itself causes.
  def syncs_since(baseline) = @bridge.synchronize_count - baseline

  def test_launches_inside_submit_do_not_wait
    a = Tilework::Buffer.from([1.0, 2.0, 3.0, 4.0])
    b = Tilework::Buffer.from([1.0, 1.0, 1.0, 1.0])
    baseline = @bridge.synchronize_count

    job = Tilework.submit { (a + b) * 2.0 }

    refute_empty waits
    assert_equal [false], waits.uniq
    assert_equal 0, syncs_since(baseline), "submit 自体は同期しない"
    assert_kind_of Tilework::Submission::Job, job
  end

  def test_wait_synchronizes_once_and_returns_the_block_value
    a = Tilework::Buffer.from([1.0, 2.0])
    baseline = @bridge.synchronize_count
    job = Tilework.submit { a + a }

    out = job.wait

    assert_kind_of Tilework::Buffer, out
    assert_equal 1, syncs_since(baseline)
    assert job.done?

    assert_same out, job.wait
    assert_equal 1, syncs_since(baseline), "2 回目の wait は同期しない"
  end

  def test_value_is_an_alias_for_wait
    a = Tilework::Buffer.from([1.0, 2.0])
    baseline = @bridge.synchronize_count
    job = Tilework.submit { a * 3.0 }

    assert_kind_of Tilework::Buffer, job.value
    assert_equal 1, syncs_since(baseline)
  end

  def test_outside_submit_nothing_changes
    a = Tilework::Buffer.from([1.0, 2.0])
    Tilework.submit { a + a }.wait
    @bridge.dispatches.clear
    before = @bridge.synchronize_count

    a + a

    assert_equal [true], waits.uniq, "境界の外は同期起動のまま"
    assert_equal before, @bridge.synchronize_count
  end

  def test_nested_submit_does_not_add_a_boundary
    a = Tilework::Buffer.from([1.0, 2.0])
    baseline = @bridge.synchronize_count

    outer = Tilework.submit do
      inner = Tilework.submit { a + a }
      assert inner.done?, "内側の Job は完了済みとして返る"
      assert_equal 0, syncs_since(baseline), "内側は同期しない"
      inner.wait * 2.0
    end

    assert_equal [false], waits.uniq
    assert_equal 0, syncs_since(baseline)
    outer.wait
    assert_equal 1, syncs_since(baseline)
  end

  def test_an_exception_drains_the_queue_before_propagating
    a = Tilework::Buffer.from([1.0, 2.0])
    boom = Class.new(StandardError)
    baseline = @bridge.synchronize_count

    error = assert_raises(boom) do
      Tilework.submit do
        a + a
        raise boom, "boom"
      end
    end

    assert_equal "boom", error.message
    assert_equal 1, syncs_since(baseline), "巻き戻す前にキューを drain する"
    assert_equal [false], waits.uniq
  end

  def test_free_inside_submit_is_deferred_until_the_job_completes
    a = Tilework::Buffer.from([1.0, 2.0])
    temp = nil

    job = Tilework.submit do
      temp = a + a
      temp.free
      refute temp.freed?, "実行中のカーネルが読んでいるメモリは手放さない"
      temp * 2.0
    end

    refute temp.freed?
    job.wait
    assert temp.freed?, "wait 後に実際に解放される"
  end

  def test_a_deferred_free_still_runs_when_the_block_raises
    a = Tilework::Buffer.from([1.0, 2.0])
    temp = nil

    assert_raises(RuntimeError) do
      Tilework.submit do
        temp = a + a
        begin
          raise "boom"
        ensure
          temp.free
        end
      end
    end

    assert temp.freed?, "例外経路でも drain 後に解放される"
  end

  def test_free_outside_submit_is_immediate
    buffer = Tilework::Buffer.new(:float32, 4)
    buffer.free
    assert buffer.freed?
  end

  def test_nn_kernels_are_enqueued_inside_submit
    x = Tilework::Buffer.new(:float32, 8, shape: [2, 4])

    Tilework.submit { Tilework::NN.softmax(x) }

    refute_empty waits
    assert_equal [false], waits.uniq
  end

  def test_reading_back_to_the_host_still_synchronizes
    a = Tilework::Buffer.from([1.0, 2.0])
    baseline = @bridge.synchronize_count

    Tilework.submit { (a + a).to_a }

    assert_operator syncs_since(baseline), :>=, 1, "host 転送は境界の中でも待つ"
  end

  def test_submission_active_only_inside_the_block
    refute Tilework::Submission.active?
    Tilework.submit { assert Tilework::Submission.active? }.wait
    refute Tilework::Submission.active?
  end

  # An asynchronous launch returns before it runs, so a failure has no caller
  # to return to. The bridge reports it from `synchronize`, and core raises it
  # as a Tilework error at that boundary rather than a bare RuntimeError.
  def test_a_failed_asynchronous_launch_surfaces_at_synchronize
    @bridge.define_singleton_method(:synchronize) { raise "command buffer failed" }

    error = assert_raises(Tilework::LaunchError) { Tilework.synchronize }
    assert_includes error.message, "command buffer failed"
  end
end
