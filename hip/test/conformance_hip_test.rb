# frozen_string_literal: true

require_relative "test_helper"
require "tilework/conformance"

class ConformanceHipTest < Minitest::Test
  def setup
    skip "GPU tests disabled (set GPU_TESTS=1 or run rake test:hip)" unless gpu_tests_enabled?
    skip "not running on the HIP backend" unless gpu_backend == :hip
    Tilework.backend = :hip
    Tilework::Kernel.clear_cache!
  end

  Tilework::Conformance.names.each do |name|
    define_method("test_#{name}") { Tilework::Conformance.run_check!(name) }
  end
end
