# frozen_string_literal: true

module Tilework
  # Expands the CUDA-flavored dialect into the backend's shader language.
  # The approach is to prepend the compat header (compat/<backend>.h) and let
  # each backend's preprocessor do the actual expansion.
  module Dialect
    HEADERS = {
      metal: File.expand_path("compat/metal.h", __dir__),
      cuda:  File.expand_path("compat/cuda.h",  __dir__),
      hip:   File.expand_path("compat/hip.h",   __dir__),
    }.freeze

    # The dialect's macro set follows the Metal way of writing, so exactly two
    # spots are not valid C++ (NVRTC) syntax. Header macros cannot absorb them,
    # so on the CUDA path only, the source itself is minimally rewritten.
    #
    # 1. `f(GPU_ARGS, device const float* a)` — in Metal this expands to three
    #    implicit arguments, which have no CUDA counterpart. GPU_ARGS can be an
    #    empty macro, but the comma written in the source would remain as
    #    `f( , ...)`, so the comma is dropped here too.
    # 2. `constant int& n` — removing `constant` leaves the reference `int& n`,
    #    but dispatch passes scalars as value bytes (kernel.rb#convert_args), so
    #    as cuLaunchKernel kernelParams it must be the by-value `const int n`. A
    #    reference still compiles and silently corrupts the value, so it is
    #    rewritten to by-value here.
    #
    # MSL address-space qualifiers (device / constant) are also dropped here.
    # cuda.h cannot `#define device`: CUDA's host_defines.h defines
    # `__device__` as `__location__(device)`, so an empty `device` macro breaks
    # __device__ and leaves every function without an execution-space
    # annotation. Rewriting only the dialect source avoids that side effect.
    CPP_DIALECT_REWRITES = [
      [/\bGPU_ARGS\s*,\s*/, ""].freeze,
      # `constant int& n` -> `const int n` (by value). Must run before the
      # bare `constant` removal.
      [/\bconstant\s+([A-Za-z_]\w*)\s*&/, 'const \1'].freeze,
      [/\b(?:device|constant)\s+/, ""].freeze,
    ].freeze

    module_function

    def expand(src)
      "#{compat_header}\n#line 1 \"kernel\"\n#{rewrite(src)}"
    end

    # Unknown backends (such as :custom in tests) are treated as the Metal dialect.
    def header_key
      HEADERS.key?(Tilework.backend) ? Tilework.backend : :metal
    end

    def compat_header
      key = header_key
      # Memoized per backend. With a single key, a header read once under
      # :metal would stay after switching to :cuda and silently generate the
      # wrong source.
      @compat_headers ||= {}
      @compat_headers[key] ||= File.read(HEADERS.fetch(key))
    end

    def rewrite(src)
      return src unless %i[cuda hip].include?(header_key)

      CPP_DIALECT_REWRITES.reduce(src) { |acc, (pattern, replacement)| acc.gsub(pattern, replacement) }
    end
  end
end
