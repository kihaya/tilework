# frozen_string_literal: true

require_relative "test_helper"
require "rbconfig"

class HipBackendLoadTest < Minitest::Test
  def test_unbuilt_extension_has_an_actionable_error
    skip "native HIP bridge is built" if File.exist?(File.expand_path("../lib/tilework/hip/bridge.#{RbConfig::CONFIG["DLEXT"]}", __dir__))
    error = assert_raises(Tilework::DeviceError) { Tilework.backend = :hip }
    assert_match(/HIP bridge/, error.message)
    assert_match(/ROCM_HOME/, error.message)
  end
end
