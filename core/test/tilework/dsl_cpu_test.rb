# frozen_string_literal: true

require_relative "../test_helper"

# :cpu-specific behavior of Tilework.kernel {}. Cross-backend numerical
# agreement lives in Tilework::Conformance (lib/tilework/conformance.rb, run on
# :cpu by test/tilework/conformance_test.rb); what remains here are the
# constraints only :cpu has.
class DSLCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
    Tilework::Kernel.clear_cache!
  end

  def teardown
    Tilework.instance_variable_set(:@backend, nil)
    Tilework.instance_variable_set(:@bridge, nil)
  end

  def test_dialect_string_kernels_are_rejected
    error = assert_raises(Tilework::CompileError) { Tilework.compile("GPU_KERNEL void f(GPU_ARGS) {}", name: "f") }
    assert_includes error.message, "does not support dialect kernel strings"
  end
end
