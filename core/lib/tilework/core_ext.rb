# frozen_string_literal: true

require_relative "../tilework"

# Opt-in core extension. Adds Array#to_gpu only when "tilework/core_ext" is
# required (a plain require "tilework" does not touch core classes).
class Array
  def to_gpu(dtype: nil)
    Tilework::Buffer.from(self, dtype: dtype)
  end
end

# Top-level with_gpu (an alias of Tilework.run). Defined private like other
# Kernel methods so it is not visible with an explicit receiver.
module Kernel
  private

  def with_gpu(*inputs, &block)
    Tilework.run(*inputs, &block)
  end
end
