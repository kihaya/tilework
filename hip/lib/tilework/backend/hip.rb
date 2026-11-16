# frozen_string_literal: true

# Activates the HIP backend when `Tilework.backend = :hip` is selected.
begin
  require "tilework/hip/bridge"
rescue LoadError => e
  raise Tilework::DeviceError,
        "failed to load the HIP bridge (is the native extension built? run `rake compile` " \
        "in the tilework-hip gem; requires ROCm — set ROCM_HOME): #{e.message}"
end

Tilework.register_backend(:hip, Tilework::HIPBridge)

module Tilework::HIPBridge
  def self.gemm_library_name = :hipblas
  def self.random_library_name = :hiprand
  def self.fft_library_name = :hipfft
  def self.conv_library_name = :miopen
  def self.gemm_supported_dtypes = %i[float32 float16 bfloat16]
  def self.conv_supported_dtypes = %i[float32 float16 bfloat16]
end
