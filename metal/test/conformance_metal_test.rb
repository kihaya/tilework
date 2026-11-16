# frozen_string_literal: true

require_relative "test_helper"
require "tilework/conformance"

# Runs Tilework::Conformance on :metal (needs a real Metal device) via
# rake test:metal on Apple Silicon. The checks live only in
# lib/tilework/conformance.rb; this file repeats the shape of
# ConformanceCPUTest for :metal, which is itself the evidence that one suite
# passes on every backend.
class ConformanceMetalTest < Minitest::Test
  def setup
    skip "GPU tests disabled (set GPU_TESTS=1 or run rake test:metal)" unless gpu_tests_enabled?
    skip "not running on the Metal backend" unless gpu_backend == :metal
    Tilework.backend = :metal
    Tilework::Kernel.clear_cache!
  end

  Tilework::Conformance.names.each do |name|
    define_method("test_#{name}") do
      Tilework::Conformance.run_check!(name)
    end
  end
end
