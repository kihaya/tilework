# frozen_string_literal: true

require_relative "../core/lib/tilework/version"

Gem::Specification.new do |spec|
  spec.name = "tilework-cuda"
  spec.version = Tilework::VERSION
  spec.authors = ["Kihaya Sugiura"]
  spec.email = ["kihayas@gmail.com"]

  spec.summary = "CUDA backend for tilework"
  spec.description = "The CUDA backend gem for tilework: the native " \
                     "extension (ext/tilework/cuda, CUDA Driver API + NVRTC host C++). Depends on " \
                     "the tilework core gem for Buffer/Kernel/DSL/Ops and the compat/cuda.h dialect " \
                     "header."
  spec.homepage = "https://github.com/kihaya/tilework"
  spec.license = "MIT"
  spec.required_ruby_version = ">= 3.1"

  spec.files = Dir["ext/**/*.{rb,c,cpp}", "lib/**/*.rb", "README.md", "LICENSE*"]
  spec.require_paths = ["lib"]
  spec.extensions = ["ext/tilework/cuda/extconf.rb"]

  # CI sets this to tag the built gem with its target platform (e.g.
  # x86_64-linux, aarch64-linux) before `gem push`; local builds stay
  # platform-independent.
  spec.platform = ENV["TILEWORK_GEM_PLATFORM"] if ENV["TILEWORK_GEM_PLATFORM"]

  # Lockstep versioning: a contract-breaking change always bumps the minor
  # version, so the dependency is declared as ~> X.Y.0.
  spec.add_dependency "tilework", "~> #{Tilework::VERSION[/\A\d+\.\d+/]}.0"

  spec.metadata["rubygems_mfa_required"] = "true"
end
