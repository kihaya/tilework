# frozen_string_literal: true

# Activates the Metal backend when `Tilework.backend = :metal` is selected.
#
# @note Raises {Tilework::DeviceError} when the native extension has not been built.
#
# `Tilework.backend = :metal` requires this conventional path
# (tilework/backend/<name>). If this file itself is missing, the gem is not
# installed and tilework.rb turns the LoadError into a DeviceError. A failure
# past this point means the gem is installed but the extension was not built,
# a different situation, so it is raised as its own DeviceError rather than
# a bare LoadError.
begin
  require "tilework/metal/bridge"
rescue LoadError => e
  raise Tilework::DeviceError,
        "failed to load the Metal bridge (is the native extension built? run `rake compile` " \
        "in the tilework-metal gem): #{e.message}"
end

Tilework.register_backend(:metal, Tilework::MetalBridge)

module Tilework::MetalBridge
  # @api private
  # core's math/*.rb report delegate library names assuming CUDA
  # (:cublas/:curand/...). Metal implements the same ABI with MPS, so this
  # overrides them with the real names for what is implemented — resolved
  # here on the reporting side via respond_to?, without adding a backend-name
  # branch to core (see CLAUDE.md).
  def self.gemm_library_name = :mps
  # MPS GEMM command buffers are committed onto the same Metal queue after
  # flushing Tilework's open kernel batch, so Graph does not need host fences.
  # A failure of one surfaces at the next `synchronize`.
  def self.ordered_gemm? = true
  def self.random_library_name = :mps
  def self.fft_library_name = :mps
  def self.conv_library_name = :mps
  def self.gemm_supported_dtypes = %i[float32 float16 bfloat16]
  def self.conv_supported_dtypes = %i[float32 float16 bfloat16]
end
