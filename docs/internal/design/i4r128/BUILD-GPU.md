# Building the I4R128 GPU path on ROCm 10.1.0 / gfx1151

How to compile the I4R128 HIP kernels and their tests against the pinned
toolchain. The design contract is in [RFC.md](RFC.md); this page is only the
build.

## Toolchain

| Item | Value |
| --- | --- |
| ROCm | 10.1.0 (TheRock `therock-dist-linux-gfx1151-10.1.0`, sha `50deb993…`) |
| hipcc | AMD clang 24.0.0git, HIP 7.16.26385 |
| Offload arch | `gfx1151` |
| Matches | Gufo's pinned ROCm 10 / HIP Clang 24 |

Environment for a standalone (non-CMake) build:

```sh
export ROCM_PATH=/opt/rocm-10.1.0
export PATH=/opt/rocm-10.1.0/bin:$PATH
```

**Container prerequisite.** A freshly rebuilt box has no C++ standard library
headers, so hipcc's device pass fails on `<cstdlib>`. Install them first
(sandbox-local only, no host or toolchain substitution):

```sh
sudo apt-get update && sudo apt-get install -y g++
```

## Standalone compile (what was verified)

Kernels compile clean for gfx1151:

```sh
hipcc --offload-arch=gfx1151 -I. -c src/core/quant/i4r_kernels.hip.cpp -o kernels.o
```

GPU test links against the kernels plus the CPU oracle objects
(`i4r_rotation.cpp`, `ggml_dequant.cpp`), skipping with code 77 when no GPU
is present:

```sh
g++  -std=c++20 -I. -c src/core/quant/i4r_rotation.cpp -o rot.o
g++  -std=c++20 -I. -c src/core/quant/ggml_dequant.cpp -o dq.o
hipcc --offload-arch=gfx1151 -I. -std=c++20 \
  tests/models/qwen38_flash_next/i4r_hip_test.hip -x none kernels.o rot.o dq.o \
  -o i4r_gpu_test
./i4r_gpu_test   # SKIP (77) with no device; runs the checks on gfx1151
```

The `-x none` before the object files stops hipcc's `-x hip` from treating the
`.o` files as source.

The host-emulation test needs no HIP at all — plain C++ against the oracle:

```sh
g++ -std=c++20 -I. tests/core/i4r_hip_math_host_test.cpp \
  src/core/quant/i4r_rotation.cpp src/core/quant/ggml_dequant.cpp -o i4r_host_test
./i4r_host_test   # 9 PASS / 0 FAIL, no GPU
```

`i4r_hip_math.hpp` neutralises `__device__`/`__host__` when they are not
already defined, so the same header compiles under hipcc and under plain g++.

## CMake integration

Already wired in `CMakeLists.txt`:

- `src/core/quant/i4r_kernels.hip.cpp` is added to `gufo_core` under
  `if (ENGINE_ENABLE_HIP)`, so the `i4r_launch_*` externs link with the
  library.
- `i4r_hip_math_host_test` is a CPU test (`LABELS "cpu;core;quant"`), built
  whenever `BUILD_TESTING` is on.
- `i4r_hip_test` is a GPU test (`LABELS "gpu;hip;gfx1151"`,
  `SKIP_RETURN_CODE 77`) in the HIP test section.

Configure and build as the repo prescribes:

```sh
cmake --preset gpu-test
cmake --build --preset gpu-test --target i4r_hip_test i4r_hip_math_host_test
ctest --preset gpu-full -R 'i4r_hip' --output-on-failure
```

## Known gap (runtime, not compile)

The plain TheRock tarball does not include Gufo's `rocr-fixed-cwsr-rocm10`
runtime patch. It affects kernel execution on device, not compilation. The
kernels compile and the host-emulation test passes without it; the GPU test's
device run needs the patched runtime, which is the build host's responsibility.
