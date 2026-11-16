# frozen_string_literal: true

require_relative "../core/lib/tilework/version"
require "rbconfig"

Gem::Specification.new do |spec|
  spec.name = "tilework-hip"
  spec.version = Tilework::VERSION
  spec.authors = ["Kihaya Sugiura"]
  spec.email = ["kihayas@gmail.com"]
  spec.summary = "HIP backend for tilework"
  spec.description = "The HIP backend gem for tilework: HIP Runtime + HIPRTC native bridge."
  spec.homepage = "https://github.com/kihaya/tilework"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1"
  # The registry gets the source-only gem and compiles this extension at
  # install time. The release workflow sets TILEWORK_GEM_PLATFORM only after
  # compiling, so its platform gem additionally carries the ready-to-load
  # native bridge.
  spec.files = Dir["ext/**/*.{rb,c,cpp}", "lib/**/*.rb", "README.md", "LICENSE*"]
  if ENV["TILEWORK_GEM_PLATFORM"]
    spec.files.concat Dir["lib/tilework/hip/bridge.#{RbConfig::CONFIG.fetch("DLEXT")}"]
  end
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/tilework/hip/extconf.rb"]
  spec.platform = ENV["TILEWORK_GEM_PLATFORM"] if ENV["TILEWORK_GEM_PLATFORM"]
  spec.add_dependency "tilework", "~> #{Tilework::VERSION[/\A\d+\.\d+/]}.0"
  spec.metadata["rubygems_mfa_required"] = "true"
end
