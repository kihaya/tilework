# frozen_string_literal: true

require_relative "../test_helper"

class GPUModuleTest < Minitest::Test
  include BridgeTestSetup

  # With the backend registry, core no longer lists possible backend names.
  # backend= just requires tilework/backend/<name>, so a typo and an
  # uninstalled backend gem raise the same DeviceError.
  def test_unknown_backend_raises_device_error
    error = assert_raises(Tilework::DeviceError) { Tilework.backend = :vulkan }
    assert_match(/backend :vulkan is not installed/, error.message)
    assert_match(/tilework-vulkan/, error.message)
  end

  def test_native_bridge_shims_remain_source_parity_copies
    root = File.expand_path("../../..", __dir__)
    shims = %w[cuda metal hip].to_h do |backend|
      path = File.join(root, backend, "ext/tilework/#{backend}/gpu_bridge.c")
      source = File.read(path)
      # The selected Ruby module is deliberately backend-local; all remaining
      # shim code is one shared ABI contract.
      [backend, source.gsub(/"(?:CUDA|Metal|HIP)Bridge"/, '"BackendBridge"')]
    end
    assert_equal shims.fetch("cuda"), shims.fetch("metal")
    assert_equal shims.fetch("cuda"), shims.fetch("hip")
  end

  # Two cases: in core-only CI the backend gem is not on the load path, and in
  # the monorepo the Ruby loader is visible but the native extension is not
  # built. The contract is only that neither leaks a LoadError; both become a
  # DeviceError that points the user at installing or building the backend.
  def test_unavailable_cuda_backend_raises_actionable_device_error
    error = assert_raises(Tilework::DeviceError) { Tilework.backend = :cuda }
    assert_match(/(?:tilework-cuda|CUDA bridge)/, error.message)
    assert_match(/(?:Gemfile|rake compile)/, error.message)
  end

  # --- Capability queries ---

  # The contract's key list lives only in lib/tilework/capabilities.rb (not duplicated here).
  def test_capabilities_always_answers_the_whole_contract
    # Unimplemented entries still appear as false / nil, so callers can branch
    # on the value rather than on key presence.
    assert_equal Tilework::Capabilities::NAMES.sort, Tilework.capabilities.to_h.keys.sort
  end

  def test_capabilities_reports_unimplemented_features_honestly
    caps = Tilework.capabilities
    refute caps[:offload_async], "the async transfer ABI is not implemented yet"
    assert_nil caps[:gemm_library], "no GEMM library is wired up yet"
    refute caps[:gemm_batched], "batched GEMM needs a GEMM library"
  end

  # A backend branch can be broken without anyone noticing until it is actually
  # built, so build each one. This checks that it answers with every key and
  # without raising, not the values.
  def test_every_backend_branch_assembles_its_capabilities
    %i[cuda metal hip cpu custom].each do |name|
      Tilework.instance_variable_set(:@backend, name)
      caps = Tilework.capabilities
      assert_equal Tilework::Capabilities::NAMES.sort, caps.to_h.keys.sort,
                   "the #{name.inspect} branch must answer the whole contract"
    end
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  def test_every_builtin_backend_declares_bfloat16_storage
    %i[cuda metal hip cpu].each do |name|
      Tilework.instance_variable_set(:@backend, name)
      caps = Tilework.capabilities
      assert caps.bf16_storage, "#{name} must compute with bfloat16 storage"
      assert_includes caps.dtypes, :bfloat16
    end
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  # The :metal branch declares float16 storage (easy to lose silently in a regression).
  def test_metal_branch_declares_its_device_specific_facts
    Tilework.instance_variable_set(:@backend, :metal)
    caps = Tilework.capabilities
    assert_equal 32, caps[:warp_size]
    assert caps[:unified_memory]
    assert_includes caps[:dtypes], :float16
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  # When the bridge lacks the optional query (as MockBridge does), fall back to
  # the known constants.
  def test_cuda_capabilities_fall_back_without_the_optional_probe
    refute Tilework.bridge.respond_to?(:device_capabilities)
    Tilework.instance_variable_set(:@backend, :cuda)
    caps = Tilework.capabilities
    assert_equal 32, caps[:warp_size] # CUDA's warpSize is always 32
    refute caps[:tensor_core], "must not claim tensor cores without probing the device"
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  def test_hip_never_interprets_gfx_as_cuda_capability
    Tilework.instance_variable_set(:@backend, :hip)
    caps = Tilework.capabilities
    assert_nil caps[:compute_capability]
    refute caps[:tf32]
    refute caps[:tensor_core]
    refute caps[:fp16]
    assert_includes caps[:dtypes], :float16
  ensure
    Tilework.instance_variable_set(:@backend, :custom)
  end

  def test_device_name_delegates_to_bridge
    assert_equal "Mock Tilework", Tilework.device_name
  end

  def test_bracket_shorthand_with_splat
    buf = Tilework[1.0, 2.0, 3.0]
    assert_instance_of Tilework::Buffer, buf
    assert_equal :float32, buf.dtype
    assert_equal [1.0, 2.0, 3.0], buf.to_a
  end

  def test_bracket_shorthand_with_array_and_dtype
    assert_equal :int32, Tilework[[1, 2, 3]].dtype
    assert_equal :float32, Tilework[[1, 2], dtype: :float32].dtype
    assert_equal :float32, Tilework[1, 2, dtype: :float32].dtype
  end

  def test_bracket_shorthand_with_nested_array_builds_matrix
    buf = Tilework[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
    assert_equal [2, 3], buf.shape
    assert buf.matrix?
  end

  def test_core_ext_adds_array_to_gpu
    require "tilework/core_ext"
    buf = [1.0, 2.0].to_gpu
    assert_instance_of Tilework::Buffer, buf
    assert_equal :int32, [1, 2].to_gpu.dtype
    assert_equal :float32, [1, 2].to_gpu(dtype: :float32).dtype
  end

  def test_error_hierarchy
    assert_operator Tilework::DeviceError, :<, Tilework::Error
    assert_operator Tilework::CompileError, :<, Tilework::Error
    assert_operator Tilework::LaunchError, :<, Tilework::Error
    assert_operator Tilework::Error, :<, StandardError
  end
end
