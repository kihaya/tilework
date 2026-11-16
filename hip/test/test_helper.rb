# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../../core/lib", __dir__)
$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "tilework"
require "minitest/autorun"

def gpu_tests_enabled?
  ENV["GPU_TESTS"] == "1"
end

def gpu_backend
  (ENV["GPU_BACKEND"] || "hip").to_sym
end
