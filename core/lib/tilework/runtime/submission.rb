# frozen_string_literal: true

module Tilework
  # An explicit asynchronous submit boundary for the eager path (#71).
  #
  # `Kernel#launch` always uses `wait: true`, so the host waits on every
  # launch. On hardware that wait is an order of magnitude larger than a small
  # kernel (a no-op kernel on HIP: 162 µs/op synchronous vs 2.4 µs/op queued on
  # the same stream with one final synchronization). Chains of small launches,
  # like elementwise chains and decode loops, are dominated by synchronization
  # rather than arithmetic.
  #
  #   job = Tilework.submit { (a * 2 + b).relu }
  #   out = job.wait
  #
  # Only inside the block are launches enqueue-only on the same queue;
  # `Job#wait` synchronizes once and returns the block's value. **Semantics
  # outside the block do not change at all** — the point is an explicit
  # boundary, not implicitly making a synchronous API asynchronous.
  #
  # It rides on the same `Ops` execution-context contract as the graph's
  # `Plan#run` (see the `Ops::CONTEXT_KEY` comment), so going asynchronous
  # needs no new machinery. What differs is ownership, which is most of this
  # file:
  #
  # - **The Job holds every output allocated inside.** `Buffer`'s finalizer
  #   calls `buffer_free`, so an intermediate created during submit that lost
  #   its last reference could vanish under a kernel reading it.
  # - **`free` is deferred until completion**, rather than rejected, so eager
  #   code written with `ensure buf.free` still works wrapped in submit
  #   (releasing memory a running kernel uses is undefined behavior on CUDA/HIP).
  # - **No pool.** The graph's `Pool` is safe asynchronously because it knows
  #   the execution order and can say "the next writer of a slot is queued
  #   after its last reader". Eager has no such information.
  module Submission
    # The context plugged into Ops while running. Satisfies the same shape as
    # Graph::Context (`#allocate` / `#async?` / `#synchronous`).
    class Context
      def initialize(job)
        @job = job
        @async = true
      end

      attr_reader :job

      def async? = @async

      def allocate(dtype, length, shape)
        Buffer.new(dtype, length, shape: shape).tap { |buffer| @job.retain(buffer) }
      end

      # `Buffer#free` inside the block is deferred to the boundary (the Ops.defer_free contract).
      def defer_free(buffer) = @job.defer_free(buffer)

      # Delegated libraries run synchronously. Sharing the name with
      # Graph::Context is deliberate, so callers that wrap either context the
      # same way (`Plan#dispatch_node`) need no branch.
      def synchronous
        previous = @async
        @async = false
        yield
      ensure
        @async = previous
      end
    end

    # The return value of `Tilework.submit`.
    class Job
      def initialize
        @value = nil
        @done = false
        @retained = []
        @deferred_free = []
      end

      # The block's value, waiting first if `wait` has not been called.
      #
      # @return [Object]
      def wait
        return @value if @done

        Tilework.synchronize
        complete!
        @value
      end
      alias value wait

      def done? = @done

      # An output allocated during submit. While the Job holds it the finalizer
      # cannot run, so memory never vanishes under a running kernel.
      def retain(buffer)
        @retained << buffer
        buffer
      end

      # Defers a `Buffer#free` made during submit until completion.
      def defer_free(buffer)
        @deferred_free << buffer
        buffer
      end

      # How many frees are pending (for tests and debugging).
      def deferred_free_count = @deferred_free.length

      # @api private
      def fulfill(value)
        @value = value
      end

      # @api private -- an exception unwind also drains the queue before getting here.
      def complete!
        return if @done

        @done = true
        @retained.clear
        pending = @deferred_free
        @deferred_free = []
        pending.each(&:free)
        nil
      end
    end

    class << self
      # Whether this thread is inside a submit.
      def context
        context = Ops.context
        context if context.is_a?(Context)
      end

      def active? = !context.nil?

      # @api private -- the public API is Tilework.submit
      def run(&block)
        # A nested submit opens no new boundary. Synchronizing inside would also
        # wait for the outer submit's queued work, defeating the explicit
        # boundary. The inner Job is returned completed (its value is already final).
        return nested(&block) if active?

        job = Job.new
        begin
          job.fulfill(Ops.with_context(Context.new(job), &block))
        rescue StandardError, ScriptError
          # Queued kernels may still be running. Drain before freeing and
          # re-raising, so the unwind does not release memory under them.
          Tilework.synchronize
          job.complete!
          raise
        end
        job
      end

      private

      def nested
        job = Job.new
        job.fulfill(yield)
        job.complete!
        job
      end
    end
  end
end
