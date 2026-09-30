# Phase 2, built natively on the Nano with clang.
# Host code is C11. The kernel is compiled to PTX and loaded at runtime through the CUDA
# driver API, so the binary has no nvcc, no CUDA runtime and no C++ in it.
#
#   make            build build/phase2, build/threshold.ptx and the kbench tool
#   make tools      just build/kbench and build/probe.ptx (see tools/kbench.md)
#   make CLANG=...  use a different clang (default clang-12)

CLANG     ?= clang-12
CUDA      ?= /usr/local/cuda
MMAPI     ?= /usr/src/jetson_multimedia_api
TEGRA_LIB ?= /usr/lib/aarch64-linux-gnu/tegra
BUILD     := build

CC      := $(CLANG)
PKGS    := gstreamer-1.0 gstreamer-app-1.0
CFLAGS  := -std=c11 -D_GNU_SOURCE -O2 -g -Wall -Wextra -Wno-unused-parameter \
           $(shell pkg-config --cflags $(PKGS)) -I$(CUDA)/include -I$(MMAPI)/include
LDFLAGS := -L$(TEGRA_LIB) -Wl,-rpath,$(TEGRA_LIB)
LDLIBS  := $(shell pkg-config --libs $(PKGS)) -lnvbuf_utils -lcuda -lEGL

# sm_53 = the Nano's Maxwell GPU. --cuda-device-only -S stops at PTX (no host side, no fatbinary).
# clang-12 doesn't know CUDA 10.2 by name and treats it as 10.1; that's fine for this code.
CUFLAGS := -x cuda --cuda-gpu-arch=sm_53 --cuda-device-only --cuda-path=$(CUDA) -O3 -S \
           -Wno-unknown-cuda-version

# kbench needs only the CUDA driver API, no GStreamer or EGL.
KBENCH_CFLAGS := -std=c11 -D_GNU_SOURCE -O2 -g -Wall -Wextra -I$(CUDA)/include

all: $(BUILD)/phase2 $(BUILD)/threshold.ptx tools

tools: $(BUILD)/kbench $(BUILD)/probe.ptx $(BUILD)/threshold.ptx

$(BUILD)/phase2: $(BUILD)/main.o $(BUILD)/gpu.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: src/%.c src/gpu.h src/ink_stats.h | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD)/threshold.ptx: kernels/threshold.cu src/ink_stats.h | $(BUILD)
	$(CLANG) $(CUFLAGS) -o $@ $<

$(BUILD)/kbench: tools/kbench.c src/ink_stats.h | $(BUILD)
	$(CC) $(KBENCH_CFLAGS) $(LDFLAGS) -o $@ $< -lcuda

$(BUILD)/probe.ptx: tools/probe.cu | $(BUILD)
	$(CLANG) $(CUFLAGS) -o $@ $<

$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean tools
