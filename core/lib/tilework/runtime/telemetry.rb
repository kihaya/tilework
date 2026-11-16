# frozen_string_literal: true

module Tilework
  # Profiling event collection and resource accounting.
  # Instrumentation is called from Ruby-side Kernel / Buffer /
  # Tilework.synchronize, so it works on every backend without bridge changes.
  #
  # Event collection (#instrument) is active only inside a Tilework.profile
  # block and is a pass-through otherwise (not even Process.clock_gettime runs
  # outside one). Counting live buffers/pipelines in Resources is always on,
  # since leak detection and observability need it outside profile blocks too.
  module Telemetry
    KEY = :gpu_telemetry_events

    module_function

    def collector
      Thread.current[KEY]
    end

    def enabled?
      !collector.nil?
    end

    # Times the block as an event tied to name/label. Outside Tilework.profile it
    # does not measure and just returns the block's value. extra (bytes:,
    # threads:, cache_hit:, ...) is per-event-kind detail.
    def instrument(name, label: nil, **extra)
      return yield unless enabled?

      start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
      result = yield
      duration = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
      collector << { name: name, label: label, duration: duration }.merge(extra)
      result
    end

    def start!
      raise Error, "Tilework.profile blocks cannot be nested" if enabled?

      Thread.current[KEY] = []
    end

    def finish!
      events = collector
      Thread.current[KEY] = nil
      events
    end

    # Always counts live buffers / bytes and cached pipelines. Handles are keyed
    # by object_id: every alloc returns a new handle object on any backend, so it
    # identifies them uniquely and Hash#delete is naturally idempotent against a
    # double free (a backend that reuses integer handles just re-enters the next
    # alloc as a fresh entry).
    # No Mutex: untrack_buffer is also called from GC finalizers (buffer.rb),
    # which run under trap-context-like restrictions where Mutex#synchronize
    # raises ThreadError.
    module Resources
      @buffers = {} # handle.object_id => bytesize

      class << self
        def track_buffer(handle, bytesize)
          @buffers[handle.object_id] = bytesize
        end

        def untrack_buffer(handle)
          @buffers.delete(handle.object_id)
        end

        def snapshot
          { live_buffers: @buffers.size, live_bytes: @buffers.each_value.sum, cached_pipelines: Kernel.cache_size }
        end
      end
    end

    # Aggregated events returned by {Tilework.profile}.
    class Report
      attr_reader :events

      def initialize(events)
        @events = events
      end

      # Groups events by name and label.
      # @return [Hash]
      def summary
        @summary ||= events.group_by { |e| [e[:name], e[:label]] }.transform_values do |group|
          {
            count: group.size,
            total_duration: group.sum { |e| e[:duration] },
            total_bytes: group.sum { |e| e[:bytes] || 0 },
            cache_hits: group.count { |e| e[:cache_hit] },
          }
        end
      end

      # Formats a compact profiling table.
      # @return [String]
      def to_s
        return "Tilework.profile: (no events recorded)" if events.empty?

        header = format("%-24s %8s %12s %14s", "event", "count", "ms", "bytes")
        rows = summary.map do |(name, label), s|
          event_label = [name, label].compact.join(":")
          bytes = s[:total_bytes].positive? ? s[:total_bytes].to_s : "-"
          format("%-24s %8d %12.3f %14s", event_label, s[:count], s[:total_duration] * 1000, bytes)
        end
        ([header] + rows).join("\n")
      end
    end
  end
end
