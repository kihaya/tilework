# frozen_string_literal: true

require_relative "../test_helper"
require "tilework/conformance"

# Runs Tilework::Conformance on :cpu. Each test_ method here is generated from
# Tilework::Conformance::CHECKS; what a check verifies lives only in
# lib/tilework/conformance.rb.
#
# Backend gems do not need to copy this file: calling
# Tilework::Conformance.run!(backend: :metal) once runs the same checks.
class ConformanceCPUTest < Minitest::Test
  def setup
    Tilework.backend = :cpu
  end

  Tilework::Conformance.names.each do |name|
    define_method("test_#{name}") do
      Tilework::Conformance.run_check!(name)
    end
  end
end
