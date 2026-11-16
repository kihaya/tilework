# frozen_string_literal: true

# HIPRTC is a separate ROCm library.  Test the headers and both shared
# libraries, rather than treating the presence of hipcc as sufficient.
require "rbconfig"

hdrdir = RbConfig::CONFIG["rubyhdrdir"]
archhdrdir = RbConfig::CONFIG["rubyarchhdrdir"]
dlext = RbConfig::CONFIG["DLEXT"]
cc = RbConfig::CONFIG["CC"]
cxx = RbConfig::CONFIG["CXX"] || "c++"
srcdir = File.expand_path(__dir__)
rocm_home = ENV["ROCM_HOME"] || "/opt/rocm"
include_dir = "#{rocm_home}/include"
lib_dirs = ["#{rocm_home}/lib", "#{rocm_home}/lib64"].select { |d| Dir.exist?(d) }
have_headers = File.file?("#{include_dir}/hip/hip_runtime_api.h") && File.file?("#{include_dir}/hip/hiprtc.h")
have_runtime = lib_dirs.any? { |d| File.exist?("#{d}/libamdhip64.so") }
have_hiprtc = lib_dirs.any? { |d| File.exist?("#{d}/libhiprtc.so") }
have_hipblas = File.file?("#{include_dir}/hipblas/hipblas.h") &&
               lib_dirs.any? { |d| File.exist?("#{d}/libhipblas.so") }
have_hiprand = File.file?("#{include_dir}/hiprand/hiprand.h") &&
               lib_dirs.any? { |d| File.exist?("#{d}/libhiprand.so") }
have_hipfft = File.file?("#{include_dir}/hipfft/hipfft.h") &&
              lib_dirs.any? { |d| File.exist?("#{d}/libhipfft.so") }
have_miopen = File.file?("#{include_dir}/miopen/miopen.h") &&
              lib_dirs.any? { |d| File.exist?("#{d}/libMIOpen.so") }
have_ruby_headers = File.file?("#{hdrdir}/ruby.h") && File.file?("#{archhdrdir}/ruby/config.h")

if have_headers && have_runtime && have_hiprtc && have_ruby_headers
  ldflags = lib_dirs.map { |d| "-L#{d}" }.join(" ")
  hipblas_cxxflags = have_hipblas ? " -DTILEWORK_HAS_HIPBLAS" : ""
  hipblas_ldlibs = have_hipblas ? " -lhipblas" : ""
  hiprand_cxxflags = have_hiprand ? " -DTILEWORK_HAS_HIPRAND" : ""
  hiprand_ldlibs = have_hiprand ? " -lhiprand" : ""
  hipfft_cxxflags = have_hipfft ? " -DTILEWORK_HAS_HIPFFT" : ""
  hipfft_ldlibs = have_hipfft ? " -lhipfft" : ""
  miopen_cxxflags = have_miopen ? " -DTILEWORK_HAS_MIOPEN" : ""
  miopen_ldlibs = have_miopen ? " -lMIOpen" : ""
  File.write("Makefile", <<~MAKEFILE)
    TARGET = bridge.#{dlext}
    CC = #{cc}
    CXX = #{cxx}
    CFLAGS = -O2 -Wall -fPIC -DTILEWORK_BRIDGE_MODULE=\\\"HIPBridge\\\" -I#{hdrdir} -I#{archhdrdir}
    CXXFLAGS = -O2 -Wall -Wextra -fPIC -std=c++17 -D__HIP_PLATFORM_AMD__#{hipblas_cxxflags}#{hiprand_cxxflags}#{hipfft_cxxflags}#{miopen_cxxflags} -I#{include_dir}
    LDLIBS = #{ldflags} -lamdhip64 -lhiprtc#{hipblas_ldlibs}#{hiprand_ldlibs}#{hipfft_ldlibs}#{miopen_ldlibs} -pthread
    SRCDIR = #{srcdir}
    all: $(TARGET)
    gpu_bridge.o: $(SRCDIR)/gpu_bridge.c
    \t$(CC) $(CFLAGS) -c $(SRCDIR)/gpu_bridge.c -o $@
    bridge_hip.o: $(SRCDIR)/bridge.cpp
    \t$(CXX) $(CXXFLAGS) -c $(SRCDIR)/bridge.cpp -o $@
    $(TARGET): gpu_bridge.o bridge_hip.o
    \t$(CXX) -shared gpu_bridge.o bridge_hip.o $(LDLIBS) -o $@
    clean:
    \trm -f $(TARGET) gpu_bridge.o bridge_hip.o
    sitearchdir = .
    install: $(TARGET)
    \tmkdir -p $(sitearchdir)/tilework/hip
    \tcp $(TARGET) $(sitearchdir)/tilework/hip/$(TARGET)
    .PHONY: all clean install
  MAKEFILE
  puts "created Makefile (HIP backend, #{rocm_home}; hipBLAS #{have_hipblas ? "enabled" : "not found"}, hipRAND #{have_hiprand ? "enabled" : "not found"}, hipFFT #{have_hipfft ? "enabled" : "not found"}, MIOpen #{have_miopen ? "enabled" : "not found"})"
else
  File.write("Makefile", "all:\ninstall:\nclean:\n.PHONY: all install clean\n")
  puts "skipping tilework/hip build (missing Ruby development headers or HIP Runtime/HIPRTC under #{rocm_home}; set ROCM_HOME)"
end
