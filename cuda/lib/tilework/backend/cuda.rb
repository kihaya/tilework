# frozen_string_literal: true

# Activates the CUDA backend when `Tilework.backend = :cuda` is selected.
#
# @note Raises {Tilework::DeviceError} when the native extension or CUDA Toolkit is unavailable.
#
# `Tilework.backend = :cuda` requires this conventional path
# (tilework/backend/<name>). If this file itself is missing, the gem is not
# installed and tilework.rb turns the LoadError into a DeviceError. A failure
# past this point means the gem is installed but the extension was not built,
# a different situation, so it is raised as its own DeviceError rather than
# a bare LoadError.
begin
  require "tilework/cuda/bridge"
rescue LoadError => e
  raise Tilework::DeviceError,
        "failed to load the CUDA bridge (is the native extension built? run `rake compile` " \
        "in the tilework-cuda gem; requires a CUDA Toolkit — see CUDA_HOME): #{e.message}"
end

Tilework.register_backend(:cuda, Tilework::CUDABridge)

module Tilework::CUDABridge
  def self.gemm_supported_dtypes
    dtypes = %i[float32 float16]
    probe = respond_to?(:device_capabilities) && device_capabilities
    dtypes << :bfloat16 if probe && probe[:compute_capability]&.first.to_i >= 8
    dtypes
  end

  def self.conv_supported_dtypes = gemm_supported_dtypes
end
