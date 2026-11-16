# frozen_string_literal: true

require_relative "../test_helper"

class TelemetryTest < Minitest::Test
  include BridgeTestSetup

  SRC = "GPU_KERNEL void f(GPU_ARGS, device float* a) {}"

  def test_disabled_outside_profile_block
    refute Tilework::Telemetry.enabled?
    Tilework[[1.0, 2.0]].to_a
    refute Tilework::Telemetry.enabled?
  end

  def test_profile_records_compile_and_dispatch_events
    report = Tilework.profile do
      kernel = Tilework.compile(SRC, name: "f")
      kernel.launch(grid: 1, block: 8, args: [Tilework::Buffer.new(:float32, 8)])
    end

    assert_equal 1, report.summary[[:compile, "f"]][:count]
    assert_equal 1, report.summary[[:dispatch, "f"]][:count]
  end

  def test_profile_marks_cache_hit_on_second_compile
    report = Tilework.profile do
      Tilework.compile(SRC, name: "f")
      Tilework.compile(SRC, name: "f")
    end

    compiles = report.events.select { |e| e[:name] == :compile }
    assert_equal [false, true], compiles.map { |e| e[:cache_hit] }
  end

  def test_profile_records_buffer_transfer_bytes
    report = Tilework.profile do
      buf = Tilework[[1.0, 2.0]]
      buf.to_a
    end

    h2d = report.events.find { |e| e[:name] == :h2d }
    d2h = report.events.find { |e| e[:name] == :d2h }
    assert_equal 8, h2d[:bytes]
    assert_equal 8, d2h[:bytes]
    assert report.events.any? { |e| e[:name] == :synchronize }
  end

  def test_profile_records_dispatch_bytes_and_threads
    kernel = Tilework.compile(SRC, name: "f")
    buf = Tilework::Buffer.new(:float32, 8)

    report = Tilework.profile { kernel.launch(grid: 2, block: 4, args: [buf]) }

    event = report.events.find { |e| e[:name] == :dispatch }
    assert_equal "f", event[:label]
    assert_equal buf.bytesize, event[:bytes]
    assert_equal 8, event[:threads]
  end

  def test_profile_raises_on_nesting_without_clobbering_outer
    outer_report = Tilework.profile do
      assert_raises(Tilework::Error) { Tilework.profile {} }
      Tilework[[1.0]]
    end

    assert outer_report.events.any? { |e| e[:name] == :h2d }
    refute Tilework::Telemetry.enabled?
  end

  def test_profile_cleans_up_collector_after_exception
    assert_raises(RuntimeError) { Tilework.profile { raise "boom" } }
    refute Tilework::Telemetry.enabled?
  end

  def test_report_to_s_handles_empty_and_populated_reports
    empty = Tilework.profile {}
    assert_includes empty.to_s, "no events recorded"

    populated = Tilework.profile { Tilework[[1.0]] }
    assert_includes populated.to_s, "h2d"
  end

  def test_resources_snapshot_tracks_live_buffer_count_and_bytes
    before = Tilework::Telemetry::Resources.snapshot
    buf = Tilework::Buffer.new(:float32, 10)
    during = Tilework::Telemetry::Resources.snapshot
    assert_equal before[:live_buffers] + 1, during[:live_buffers]
    assert_equal before[:live_bytes] + 40, during[:live_bytes]

    buf.free
    after = Tilework::Telemetry::Resources.snapshot
    assert_equal before[:live_buffers], after[:live_buffers]
    assert_equal before[:live_bytes], after[:live_bytes]
  end

  def test_resources_snapshot_reflects_cached_pipeline_count
    assert_equal 0, Tilework::Telemetry::Resources.snapshot[:cached_pipelines]
    Tilework.compile(SRC, name: "f")
    assert_equal 1, Tilework::Telemetry::Resources.snapshot[:cached_pipelines]
  end

  def test_finalizer_untracks_live_buffer
    buf = Tilework::Buffer.new(:float32, 4)
    handle = buf.instance_variable_get(:@handle)
    before = Tilework::Telemetry::Resources.snapshot[:live_buffers]

    Tilework::Buffer.finalizer(handle, @bridge).call

    assert_equal before - 1, Tilework::Telemetry::Resources.snapshot[:live_buffers]
  end
end
