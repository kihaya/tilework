# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../../core/lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "tilework"
require "minitest/autorun"

# Guard for real-device tests (rake test:cuda / GPU_TESTS=1).
def gpu_tests_enabled?
  ENV["GPU_TESTS"] == "1"
end

# Backend used by the real-device tests (:cuda under rake test:cuda).
# Backend-independent checks belong to Tilework::Conformance
# (conformance_cuda_test.rb).
def gpu_backend
  (ENV["GPU_BACKEND"] || "cuda").to_sym
end
