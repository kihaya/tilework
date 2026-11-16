# frozen_string_literal: true

require_relative "test_helper"
require "tilework/conformance"

# Runs Tilework::Conformance on :cuda (needs an NVIDIA GPU + CUDA Toolkit) via
# rake test:cuda. The checks live only in lib/tilework/conformance.rb; this
# file repeats the shape of ConformanceCPUTest / ConformanceMetalTest for
# :cuda, which is itself the evidence that one suite passes on every backend.
class ConformanceCudaTest < Minitest::Test
  def setup
    skip "GPU tests disabled (set GPU_TESTS=1 or run rake test:cuda)" unless gpu_tests_enabled?
    skip "not running on the CUDA backend" unless gpu_backend == :cuda
    Tilework.backend = :cuda
    Tilework::Kernel.clear_cache!
  end

  Tilework::Conformance.names.each do |name|
    define_method("test_#{name}") do
      Tilework::Conformance.run_check!(name)
    end
  end
end
