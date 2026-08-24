# CPU block-scaled matmul with `inner_tiled`

## Status

The initial x86 prototype is implemented. A canonical unsigned-i8 by signed-i8
scaled contraction with FP32 scales is recognized, packed as a
`1x16x1x32` M/N/Ko/K0 tile, and materialized as `iree_codegen.inner_tiled`
using AVX-512 VNNI `VPDPBUSD`. Eight native four-byte dot products accumulate
in i32 before the scale for the 32-element block is applied once.

## Target computation

For int8 operands with FP32 block scales, each K block computes an integer dot
product, converts that block's accumulator to FP32, applies the two scales, and
then accumulates into the FP32 output:

```text
for m_tile, n_tile:
  c_f32 = load C tile
  for k_block:
    dot_i32 = 0
    for k0 in k_block:
      dot_i32 = int8_dot(A, B, dot_i32)
    c_f32 += convert_f32(dot_i32) * A_scale * B_scale
  store c_f32
```

Scaling cannot be postponed until after the complete K reduction because every
K block may have different scales.

The intended packed layouts are:

```text
A values: [M_outer, K_outer, M_inner, K_block]
B values: [N_outer, K_outer, N_inner, K_block]
A scales: [M_outer, K_outer, M_inner]
B scales: [N_outer, K_outer, N_inner]
C:        [M_outer, N_outer, M_inner, N_inner]
```

For an N-vectorized CPU implementation, B values, B scales, and C are loaded as
vectors. A values and A scales are loaded and broadcast. The complete physical
inner tile is selected from the target dot-product intrinsic and a
register-budgeted composition of that intrinsic.

## Initial x86 target

The prototype targets AVX-512 VNNI. Native `VPDPBUSD` consumes unsigned i8 by
signed i8. Signed-by-signed models therefore require one of:

* an unsigned activation representation plus zero-point correction;
* a signed i8 to i16 widening path and signed word dot products; or
* a target-specific fallback sequence.

Signedness is part of intrinsic selection and must not be silently changed.

## Compiler work

1. Add a CPU inner-tile descriptor accepting five shaped operands in the order
   `(lhs, rhs, lhs_scale, rhs_scale, accumulator)`.
2. Describe value, scale, and accumulator tile types and verify their indexing
   maps.
3. Lower one scaled inner tile to integer MMA operations, i32-to-f32 conversion,
   FP32 scale multiplication, and FP32 accumulation.
4. Extend the CPU intrinsic/unrolling cost model to include temporary i32
   accumulators and scale registers.
5. Materialize scaled-contraction encodings into compatible packed layouts and
   create the scaled `iree_codegen.inner_tiled` operation.
6. Support tails, dynamic shapes, and profitable scale/value packing.
7. Optionally add an LLVM-bitcode ukernel after generated intrinsic lowering is
   correct and benchmarked.

## Prototype stages

The first stage is intentionally independent of encoding materialization: a
manually constructed scaled `inner_tiled` operation must lower to the expected
vector/intrinsic IR. This isolates arithmetic and register-layout correctness.
The second stage connects scaled contraction encoding materialization to that
operation. End-to-end performance tuning follows only after both stages have
correctness tests.

## Current limitations to remove

* Only the fixed `1x16x1x32` unsigned-i8 by signed-i8 AVX-512 VNNI tile is
  selected. Other signedness combinations and targets fall outside the
  prototype.
* CPU scaled contraction-to-`inner_tiled` rejects batch dimensions and requires
  one each of M, N, Ko, and K0.
* The lowering currently uses a fixed four-row M unroll. General M/N unrolling
  selection and its scale/register cost model are not yet implemented.
* Dynamic shapes, tails, and full CPU pipeline performance tuning still need
  end-to-end coverage.
