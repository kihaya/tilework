# frozen_string_literal: true

# mkmf does not support Swift, so the Makefile is written directly.
#
#   1. macOS     -> link bridge.swift with swiftc
#   2. elsewhere -> a no-op Makefile (so `bundle install` does not fail
#      without Metal; core's backend= turns the missing extension into a
#      Tilework::DeviceError)
#
# The product goes to lib/tilework/metal/bridge.<dlext> so it cannot collide
# with other backends.

require "rbconfig"

hdrdir     = RbConfig::CONFIG["rubyhdrdir"]
archhdrdir = RbConfig::CONFIG["rubyarchhdrdir"]
dlext      = RbConfig::CONFIG["DLEXT"] # darwin: "bundle"
cc         = RbConfig::CONFIG["CC"]
srcdir     = File.expand_path(__dir__)

if RbConfig::CONFIG["host_os"] =~ /darwin/
  # The C side (gpu_bridge.c) is compiled with clang and linked together with
  # the Swift side (bridge.swift) into a .bundle, using swiftc as the linker
  # driver. Ruby symbols resolve via -undefined dynamic_lookup, as is usual
  # for C extensions.
  makefile = <<~MAKEFILE
    TARGET = bridge.#{dlext}
    CC = #{cc}
    SWIFTC = swiftc
    CFLAGS = -O2 -Wall -fPIC -DTILEWORK_BRIDGE_MODULE=\\\"MetalBridge\\\" -I#{hdrdir} -I#{archhdrdir}
    SWIFTFLAGS = -O -parse-as-library
    LDFLAGS = -Xlinker -undefined -Xlinker dynamic_lookup -framework Metal -framework MetalPerformanceShadersGraph
    # MPSGraphFFTDescriptor / realToHermiteanFFT etc. need macOS 14 or later.
    DEPLOY = MACOSX_DEPLOYMENT_TARGET=14.0

    SRCDIR = #{srcdir}

    all: $(TARGET)

    gpu_bridge.o: $(SRCDIR)/gpu_bridge.c
    \t$(DEPLOY) $(CC) $(CFLAGS) -c $(SRCDIR)/gpu_bridge.c -o $@

    $(TARGET): gpu_bridge.o $(SRCDIR)/bridge.swift
    \t$(DEPLOY) $(SWIFTC) $(SWIFTFLAGS) -emit-library $(SRCDIR)/bridge.swift gpu_bridge.o $(LDFLAGS) -o $@

    clean:
    \trm -f $(TARGET) gpu_bridge.o

    # gem install calls install with sitearchdir overridden. Install to the
    # namespaced path rather than a flat one so require
    # "tilework/metal/bridge" resolves.
    sitearchdir = .
    install: $(TARGET)
    \tmkdir -p $(sitearchdir)/tilework/metal
    \tcp $(TARGET) $(sitearchdir)/tilework/metal/$(TARGET)

    .PHONY: all clean install
  MAKEFILE

  File.write("Makefile", makefile)
  puts "created Makefile (Metal backend)"
else
  File.write("Makefile", "all:\ninstall:\nclean:\n.PHONY: all install clean\n")
  puts "skipping tilework/metal build (not on macOS)"
end
