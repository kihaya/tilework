# frozen_string_literal: true

require_relative "lib/tilework/version"

Gem::Specification.new do |spec|
  spec.name = "tilework"
  spec.version = Tilework::VERSION
  spec.authors = ["Kihaya Sugiura"]
  spec.email = ["kihayas@gmail.com"]

  spec.summary = "Backend-agnostic core (Buffer / Kernel / DSL / Ops) plus the :cpu reference backend"
  spec.description = "The core of tilework: pure Ruby, " \
                     "no native extension, ships the :cpu backend as the conformance oracle for " \
                     "the tilework-metal / tilework-cuda / tilework-hip backend gems."
  spec.homepage = "https://github.com/kihaya/tilework"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1"

  spec.files = Dir["lib/**/*.{rb,h}", "README.md", "LICENSE*"]
  spec.require_paths = ["lib"]

  spec.metadata["rubygems_mfa_required"] = "true"
end
