# frozen_string_literal: true

# mkmf does not support nvcc, so the Makefile is written directly.
#
#   1. CUDA Toolkit present -> link bridge.cpp with c++. cuBLAS / cuRAND /
#      cuFFT / cuDNN are each optional: a library is linked (and
#      TILEWORK_HAS_<LIB> defined) only when both its header and its
#      shared library are found, so a Toolkit without cuDNN — the common
#      case, since cuDNN is a separate download — still builds, and Ruby
#      falls back for whatever is missing.
#   2. absent               -> a no-op Makefile (so `bundle install` does not
#      fail without CUDA; core's backend= turns the missing extension into a
#      Tilework::DeviceError)
#
# The product goes to lib/tilework/cuda/bridge.<dlext> so it cannot collide
# with other backends.

require "rbconfig"

hdrdir     = RbConfig::CONFIG["rubyhdrdir"]
archhdrdir = RbConfig::CONFIG["rubyarchhdrdir"]
dlext      = RbConfig::CONFIG["DLEXT"] # linux: "so"
cc         = RbConfig::CONFIG["CC"]
cxx        = RbConfig::CONFIG["CXX"] || "c++"
srcdir     = File.expand_path(__dir__)

cuda_home = ENV["CUDA_HOME"] || "/usr/local/cuda"
nvcc      = "#{cuda_home}/bin/nvcc"

if File.exist?(nvcc)
  # libcuda.so (Driver API) ships with the driver, not the CUDA Toolkit, so it
  # may be missing from $CUDA_HOME/lib64 — typically on Jetson (aarch64), where
  # it lives in /usr/lib/aarch64-linux-gnu. Only existing dirs are added to -L.
  # /usr/lib64 is where Fedora/RHEL packages (cuDNN included) install.
  lib_dirs = [
    "#{cuda_home}/lib64",
    "#{cuda_home}/lib64/stubs",
    "/usr/lib/#{RbConfig::CONFIG["host_cpu"]}-linux-gnu",
    "/usr/lib64",
  ].select { |d| Dir.exist?(d) }
  ldflags = lib_dirs.map { |d| "-L#{d}" }.join(" ")
  include_dirs = ["#{cuda_home}/include", "/usr/include"]

  # header, -l name, macro
  optional_libs = {
    "cuBLAS" => ["cublas_v2.h", "cublas", "TILEWORK_HAS_CUBLAS"],
    "cuRAND" => ["curand.h", "curand", "TILEWORK_HAS_CURAND"],
    "cuFFT" => ["cufft.h", "cufft", "TILEWORK_HAS_CUFFT"],
    "cuDNN" => ["cudnn.h", "cudnn", "TILEWORK_HAS_CUDNN"],
  }
  found = optional_libs.select do |_, (header, lib, _)|
    include_dirs.any? { |d| File.file?("#{d}/#{header}") } &&
      lib_dirs.any? { |d| File.exist?("#{d}/lib#{lib}.so") }
  end
  lib_cxxflags = found.values.map { |(_, _, macro)| " -D#{macro}" }.join
  lib_ldlibs = found.values.map { |(_, lib, _)| " -l#{lib}" }.join

  # bridge.cpp is pure host code with no __global__ (NVRTC compiles kernels at
  # run time), so it builds with the host C++ compiler rather than nvcc. nvcc
  # would only inject fatbin registration code and a libcudart dependency.
  # nvcc's presence is used only to detect whether the CUDA Toolkit is installed.
  makefile = <<~MAKEFILE
    TARGET = bridge.#{dlext}
    CC = #{cc}
    CXX = #{cxx}
    CFLAGS   = -O2 -Wall -fPIC -DTILEWORK_BRIDGE_MODULE=\\\"CUDABridge\\\" -I#{hdrdir} -I#{archhdrdir}
    CXXFLAGS = -O2 -Wall -fPIC -pthread#{lib_cxxflags} -I#{cuda_home}/include -I/usr/include
    LDLIBS   = #{ldflags} -lcuda -lnvrtc#{lib_ldlibs} -pthread

    SRCDIR = #{srcdir}

    all: $(TARGET)

    gpu_bridge.o: $(SRCDIR)/gpu_bridge.c
    \t$(CC) $(CFLAGS) -c $(SRCDIR)/gpu_bridge.c -o $@

    bridge_cuda.o: $(SRCDIR)/bridge.cpp
    \t$(CXX) $(CXXFLAGS) -c $(SRCDIR)/bridge.cpp -o $@

    # Ruby symbols resolve when the extension loads, so they may stay undefined
    # here (Linux shared objects allow that by default).
    $(TARGET): gpu_bridge.o bridge_cuda.o
    \t$(CXX) -shared gpu_bridge.o bridge_cuda.o $(LDLIBS) -o $@

    clean:
    \trm -f $(TARGET) gpu_bridge.o bridge_cuda.o

    # gem install calls install with sitearchdir overridden. Install to the
    # namespaced path rather than a flat one so require
    # "tilework/cuda/bridge" resolves.
    sitearchdir = .
    install: $(TARGET)
    \tmkdir -p $(sitearchdir)/tilework/cuda
    \tcp $(TARGET) $(sitearchdir)/tilework/cuda/$(TARGET)

    .PHONY: all clean install
  MAKEFILE

  File.write("Makefile", makefile)
  status = optional_libs.keys.map { |name| "#{name} #{found.key?(name) ? "enabled" : "not found"}" }
  puts "created Makefile (CUDA backend, #{cuda_home}; #{status.join(", ")})"
else
  File.write("Makefile", "all:\ninstall:\nclean:\n.PHONY: all install clean\n")
  puts "skipping tilework/cuda build (no CUDA toolchain found; set CUDA_HOME if it's installed elsewhere)"
end
