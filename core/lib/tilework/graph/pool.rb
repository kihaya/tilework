# frozen_string_literal: true

module Tilework
  module Graph
    # A pool of intermediate buffers. Buffers coming back from liveness are kept
    # per `[dtype, shape]` and handed out again.
    #
    # Eager does a `Buffer.new` per op, so a five-step elementwise chain
    # allocates and drops four intermediates. The graph knows the execution
    # order and each value's last use, so **rotating slots of the same shape**
    # is enough. The pool belongs to the plan, so reruns do not reallocate
    # either (which pays off in decode loops rerunning one plan).
    #
    # Only exact shape matches are reused — the same length with a different
    # shape is a different slot, because reshape makes views and returning one
    # to the pool would lose track of the owning buffer.
    #
    # It is safe with asynchronous launches because **reuse is not freeing**.
    # The next kernel to use a slot is queued after the last kernel that read
    # it, so same-queue ordering holds. Calling `Buffer#free` here would release
    # memory a running kernel uses (undefined behavior on CUDA).
    class Pool
      attr_reader :allocated, :reused

      def initialize
        @slots = Hash.new { |hash, key| hash[key] = [] }
        @allocated = 0
        @reused = 0
      end

      def allocate(dtype, length, shape)
        slot = @slots[[dtype, shape]]
        while (buffer = slot.pop)
          # It may have been freed from outside the plan, e.g. by a Scope (a plan
          # run inside Tilework.run). Drop it rather than hold on.
          next if buffer.freed?

          @reused += 1
          buffer.instance_variable_set(:@graph_pool_released, false)
          return buffer
        end

        @allocated += 1
        Buffer.new(dtype, length, shape: shape).tap do |buffer|
          # Marks the buffer so Plan#release returns only what this pool lent.
          # A separate set of object_ids would either keep plan outputs that left
          # alive or misread reused ids, so the owner is recorded on the buffer.
          buffer.instance_variable_set(:@graph_pool_owner, self)
        end
      end

      def owns?(buffer) = buffer.instance_variable_get(:@graph_pool_owner).equal?(self)

      # Adopts a buffer allocated outside the pool (a temporary an op made with
      # `Buffer.new`, like `nn.rb`'s scores), so what arrives through deferred
      # frees is reused on the next run.
      def adopt(buffer)
        buffer.instance_variable_set(:@graph_pool_owner, self)
        buffer
      end

      # Returning is **idempotent**. `nn.rb`'s error path frees twice with
      # `free unless freed?`, and `freed?` stays false while a free is deferred,
      # so a naive push would lend the same buffer to two nodes as two slots.
      def release(buffer)
        return buffer if buffer.instance_variable_get(:@graph_pool_released)

        buffer.instance_variable_set(:@graph_pool_released, true)
        @slots[[buffer.dtype, buffer.shape]].push(buffer)
        buffer
      end

      def pooled = @slots.each_value.sum(&:length)

      def stats = { allocated: @allocated, reused: @reused, pooled: pooled }

      # Releases the device memory the pool holds. Running kernels may still read
      # the slots, so wait first.
      def free_all!
        Tilework.synchronize
        @slots.each_value { |buffers| buffers.each(&:free) }
        @slots.clear
        nil
      end
    end

    # The context plugged into Ops while running (see the Ops::CONTEXT_KEY
    # comment). Routes output allocation to the pool and makes launches async.
    class Context
      def initialize(pool: nil, async: true)
        @pool = pool
        @async = async
        @outer = nil
        @deferred = []
      end

      attr_reader :pool

      # When there is an enclosing `Tilework.submit` boundary, ownership can be
      # handed to it (#86 stage 2). The plan keeps its own pool and passes only
      # buffers from outside the pool to the Job.
      attr_accessor :outer

      # Lets ops that free buffers mid-call (`nn.rb`'s scores, `Ops.reduce`'s
      # partials) run **asynchronously**: memory a running kernel reads is never
      # released, so there is no reason left to run them synchronously (#86).
      #
      # With a pool the buffer is returned to it — Pool's safety argument ("the
      # next writer of a slot is queued after its last reader") applies as is.
      # Views own nothing, so they are not taken and are freed on the spot (the
      # same judgment as Plan#release).
      def defer_free(buffer)
        return false if buffer.view?

        if @pool
          @pool.adopt(buffer) unless @pool.owns?(buffer)
          return @pool.release(buffer)
        end
        return @outer.defer_free(buffer) if @outer

        @deferred << buffer
      end

      # What piled up with neither a pool nor an outer boundary
      # (`Graph.buffer_pool = false`). Running kernels may read it, so wait first.
      def drain_deferred!
        return if @deferred.empty?

        pending = @deferred
        @deferred = []
        Tilework.synchronize
        pending.each(&:free)
        nil
      end

      def async? = @async

      # Makes the next allocated output **this buffer** instead of a new one, so
      # the node producing a `write_into` value gets the state's view as its
      # output from the start (#19 §3) — removing one copy.
      #
      # It applies once, and only if the dtype and length match. Otherwise a
      # normal allocation happens and `consumed?` stays false, so the caller
      # knows it could not place and falls back to the copy.
      def place(buffer)
        @placement = buffer
        @placement_consumed = false
        yield
      ensure
        @placement = nil
      end

      def placement_consumed? = @placement_consumed

      def allocate(dtype, length, shape)
        if @placement && @placement.dtype == dtype && @placement.length == length
          placed = @placement
          @placement = nil
          @placement_consumed = true
          return placed
        end
        return Buffer.new(dtype, length, shape: shape) unless @pool

        @pool.allocate(dtype, length, shape)
      end

      # Nodes that may run outside our queue, like delegated libraries and user
      # blocks, run synchronously.
      def synchronous
        previous = @async
        @async = false
        yield
      ensure
        @async = previous
      end
    end
  end
end
