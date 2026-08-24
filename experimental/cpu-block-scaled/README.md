# CPU block-scaled matmul benchmark

Compile the fixed `64x64x256` unsigned-i8 by signed-i8 benchmark with:

```shell
build/tools/iree-compile experimental/cpu-block-scaled/scaled_matmul_i8.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --iree-llvmcpu-target-cpu-features=host \
  --iree-dispatch-creation-data-tiling \
  --iree-llvmcpu-enable-inner-tiled \
  -o /tmp/scaled_matmul_i8.vmfb
```

Benchmark it with generated inputs:

```shell
build/tools/iree-benchmark-module \
  --module=/tmp/scaled_matmul_i8.vmfb \
  --device=local-task \
  --function=scaled_matmul_64x64x256 \
  --input=64x64x4xi8 --input=64x64x4xi8 \
  --input=64x64xf32 --input=64x64xf32
```

Use `--device=local-sync` to measure the generated kernel and packing on one
thread without local-task scheduling overhead. On the initial Ryzen 9 9950X
development host, four-row M unrolling reduced this measurement from about
20 us to 16 us (roughly 22%). `local-task` remains around 134 us for this small
shape because dispatch scheduling and four input-packing dispatches dominate.

For a simple correctness check, all-one values and scales must produce 256 in
every output element:

```shell
build/tools/iree-run-module \
  --module=/tmp/scaled_matmul_i8.vmfb --device=local-sync \
  --function=scaled_matmul_64x64x256 \
  --input=64x64x4xi8=1 --input=64x64x4xi8=1 \
  --input=64x64xf32=1 --input=64x64xf32=1
```

Use `--iree-hal-dump-executable-files-to=/tmp/scaled-dump` during compilation
to retain LLVM IR, object files, and disassembly inputs for inspection.

## 1024x1024x1024 kernel comparison

`scaled_matmul_1024.mlir` uses 32 scale blocks of 32 K elements. On the Ryzen 9
9950X development host, dispatch-only measurements gave:

| Kernel | local-sync | Effective GOPS | local-task |
| --- | ---: | ---: | ---: |
| Scaled UI8 x I8 -> FP32 | 2.447 ms | 878 | 0.616 ms |
| IREE s8 x s8 -> i32 mmt4d ukernel | 3.517 ms | 611 | 0.690 ms |

These are not equal signedness instructions. The existing mmt4d ukernel uses
`VPDPWSSD` after widening signed i8 values and processes two K elements per dot
instruction. The scaled kernel uses `VPDPBUSD` directly and processes four.
Consequently, the scaled prototype is already faster than that existing IREE
i8 mmt4d baseline despite conversion and scaling. An ideal unscaled UI8 x I8
`VPDPBUSD` microkernel would be the stricter reference and is expected to be
faster than the scaled kernel.

### ONNX Runtime MatMulNBits reference

`benchmark_ort_matmul_nbits.py` creates an ONNX Runtime `MatMulNBits` model with
the same 1024x1024x1024 dimensions and block size 32. On ONNX Runtime 1.29.0:

| ORT weight format | Accuracy | 1 thread | 30 threads |
| --- | ---: | ---: | ---: |
| 8-bit symmetric | 4 (internal int8 A) | 5.623 ms / 382 GOPS | 0.921 ms / 2330 GOPS |
| 4-bit symmetric | 4 (internal int8 A) | 6.434 ms / 334 GOPS | 0.899 ms / 2390 GOPS |

Accuracy level 4 permits ORT to quantize floating-point A to int8 internally,
making this much closer to our integer-dot computation. It is still not an
equal-boundary comparison: ORT receives FP32 A and performs its activation
quantization inside the operator, while our kernel receives UI8 A and its FP32
block scales directly and the table uses dispatch-only IREE timings.

### FP32-input comparison including activation quantization

`scaled_matmul_1024_with_lhs_quant.mlir` provides the closer public boundary.
It accepts FP32 A, computes one activation scale per 32-element K block, and
quantizes A with `linalg.generic` before running the scaled integer matmul. IREE
currently creates separate scale-reduction, quantization, and matmul dispatches.

On the same host, the complete operation measured 3.81 ms with `local-sync` and
0.539 ms median with `local-task`. Compared with the 8-bit ORT row above, these
measurements are about 1.48x and 1.71x faster, respectively.

This remains a prototype comparison. The IREE quantizer uses unsigned [0, 255]
quantization and therefore only models nonnegative activations. General signed
activations require zero-point correction or a signed-A intrinsic path. The
tools also came from a debug build, and the multithreaded measurements should
be repeated with pinned cores and matched thread-pool settings before drawing
strong performance conclusions.
