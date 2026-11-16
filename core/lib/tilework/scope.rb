# frozen_string_literal: true

module Tilework
  # Resource scope for with_gpu (Tilework.run). Tracks Buffers allocated inside
  # the block on a thread-local stack and frees them together when it ends.
  module Scope
    KEY = :gpu_scope_stack

    module_function

    def stack
      Thread.current[KEY] ||= []
    end

    def open
      scope = []
      stack.push(scope)
      scope
    end

    def close(scope)
      stack.delete_if { |s| s.equal?(scope) }
      scope.each(&:free)
      nil
    end

    # Called from Buffer#initialize. Does nothing when no scope is active.
    def register(buffer)
      stack.last&.push(buffer)
      buffer
    end
  end
end
