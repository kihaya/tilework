# frozen_string_literal: true

# Runs each self-contained gem directory's own `rake` from the repo root
# (see CLAUDE.md "Testing").

GEM_DIRS = %w[core metal cuda hip].freeze

def run_in(dir, *task_args)
  Dir.chdir(dir) { sh "bundle", "exec", "rake", *task_args }
end

namespace :core do
  desc "Run core's default task (unit tests + :cpu integration)"
  task :test do
    run_in("core", "test")
  end

  desc "Run core's :cpu backend as the real backend under test"
  task "test:cpu" do
    run_in("core", "test:cpu")
  end

  desc "Generate the core API reference"
  task :yard do
    run_in("core", "yard")
  end
end

namespace :metal do
  desc "Build the Metal native extension"
  task :compile do
    run_in("metal", "compile")
  end

  desc "Remove Metal build artifacts"
  task :clean do
    run_in("metal", "clean")
  end

  desc "Run the Metal backend as the real backend under test (Apple Silicon required)"
  task :test do
    run_in("metal", "test:metal")
  end

  desc "Generate the Metal backend API reference"
  task :yard do
    run_in("metal", "yard")
  end
end

namespace :cuda do
  desc "Build the CUDA native extension"
  task :compile do
    run_in("cuda", "compile")
  end

  desc "Remove CUDA build artifacts"
  task :clean do
    run_in("cuda", "clean")
  end

  desc "Run the CUDA backend as the real backend under test (NVIDIA GPU + CUDA Toolkit required)"
  task :test do
    run_in("cuda", "test:cuda")
  end

  desc "Generate the CUDA backend API reference"
  task :yard do
    run_in("cuda", "yard")
  end
end

namespace :hip do
  task :compile do
    run_in("hip", "compile")
  end

  task :clean do
    run_in("hip", "clean")
  end

  desc "Run the HIP backend as the real backend under test (ROCm GPU required)"
  task :test do
    run_in("hip", "test:hip")
  end
end

desc "Run core's test suite (the only one that runs without real GPU hardware)"
task test: "core:test"

desc "Compile all native extensions (no-op where the toolchain is missing)"
task compile: ["metal:compile", "cuda:compile", "hip:compile"]

desc "Generate API references for every gem"
task yard: ["core:yard", "metal:yard", "cuda:yard"]

desc "Remove all build artifacts"
task clean: ["metal:clean", "cuda:clean", "hip:clean"]

task default: :test
