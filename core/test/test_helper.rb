# frozen_string_literal: true

$LOAD_PATH.unshift File.expand_path("../lib", __dir__)
require "tilework"
require "minitest/autorun"

# Mock bridge for running unit tests without a GPU. Provides the same
# (mechanism-only) interface as the native bridges.
class MockBridge
  # offset is for views; always 0 on a real Handle.
  Handle = Struct.new(:data, :freed, :offset)

  attr_reader :compilations, :dispatches, :synchronize_count
  attr_accessor :compile_error

  def initialize
    @compilations = []
    @dispatches = []
    @synchronize_count = 0
    @compile_error = nil
  end

  def device_name = "Mock Tilework"
  def device_available? = true

  def buffer_alloc(bytesize)
    Handle.new(("\x00" * bytesize).b, false, 0)
  end

  # A view shares the same data with a shifted offset and frees nothing.
  def buffer_view(handle, byte_offset)
    Handle.new(handle.data, false, handle.offset + byte_offset)
  end

  def buffer_write(handle, bytes, offset)
    handle.data[handle.offset + offset, bytes.bytesize] = bytes.b
  end

  def buffer_read(handle, bytesize, offset)
    handle.data[handle.offset + offset, bytesize]
  end

  def buffer_free(handle)
    handle.freed = true
  end

  def pipeline_create(src, name)
    raise RuntimeError, @compile_error if @compile_error

    @compilations << { src: src, name: name }
    { src: src, name: name }
  end

  def pipeline_max_threads(_pipeline) = 1024

  def dispatch(pipeline, args, grid, block, wait)
    @dispatches << { pipeline: pipeline, args: args, grid: grid, block: block, wait: wait }
  end

  def synchronize
    @synchronize_count += 1
  end
end

module BridgeTestSetup
  def setup
    super
    @bridge = MockBridge.new
    Tilework.bridge = @bridge
    Tilework::Kernel.clear_cache!
  end
end

# Guard for real-device tests (GPU_TESTS=1).
def gpu_tests_enabled?
  ENV["GPU_TESTS"] == "1"
end

# Backend used by the real-device tests. Backend-independent integration
# tests (Ops / DSL / with_gpu) run identically on each, which is itself a
# check of the layering.
def gpu_backend
  (ENV["GPU_BACKEND"] || "metal").to_sym
end
