# frozen_string_literal: true

require_relative "../test_helper"
require "tilework/conformance"

# Tests Tilework::Conformance::Report and run!'s definition of "green" itself —
# not what each check does, but the mechanism that keeps **checks that did not
# run from being reported as passing**.
#
# The suite itself runs per backend in conformance_test.rb (:cpu) /
# conformance_metal_test.rb (:metal).
class ConformanceReportTest < Minitest::Test
  Result = Tilework::Conformance::Result
  Report = Tilework::Conformance::Report

  def passing = Result.new(:some_check, nil)
  def failing = Result.new(:other_check, RuntimeError.new("boom"))

  def test_all_checks_passing_is_green
    report = Report.new(:cpu, [passing, passing])
    assert report.success?
    assert_equal "2/2 conformance checks passed on :cpu", report.summary
  end

  def test_a_single_failure_is_not_green
    report = Report.new(:cpu, [passing, failing])
    refute report.success?
    assert_includes report.summary, "1/2 conformance checks passed"
    assert_includes report.summary, "other_check: boom"
  end

  # Never green with nothing checked. If this turned true, emptying the suite
  # would make every backend green.
  def test_zero_checks_is_not_green
    report = Report.new(:cpu, [])
    refute report.success?, "0 checks must never report success"
    assert_includes report.summary, "no conformance checks ran"
  end

  def test_unavailable_backend_is_not_green_and_says_why
    report = Report.new(:cuda, [], unavailable_reason: "failed to load CUDA bridge")
    assert report.unavailable?
    refute report.success?
    assert_includes report.summary, "no execution environment"
    assert_includes report.summary, "failed to load CUDA bridge"
  end

  def test_available_backend_is_not_marked_unavailable
    refute Report.new(:cpu, [passing]).unavailable?
  end

  # A missing runtime yields a non-green Report, not an exception, so the
  # `raise report.summary unless report.success?` a backend gem's CI writes
  # doubles as the report that the runtime is missing.
  def test_run_returns_a_non_green_report_when_the_backend_is_unavailable
    skip "the CUDA extension is available here" if cuda_available?

    report = Tilework::Conformance.run!(backend: :cuda)
    refute report.success?
    assert report.unavailable?
    assert_includes report.summary, "no execution environment"
  end

  # With the backend registry, backend= no longer distinguishes a typo from an
  # uninstalled gem (both are a failed require), so an unknown backend name
  # becomes a non-green Report just like an uninstalled :cuda.
  def test_run_returns_a_non_green_report_for_an_unknown_backend_name
    report = Tilework::Conformance.run!(backend: :vulkan)
    refute report.success?
    assert report.unavailable?
    assert_includes report.summary, "not installed"
  end

  private

  def cuda_available?
    Tilework.backend = :cuda
    true
  rescue Tilework::DeviceError
    false
  end
end
