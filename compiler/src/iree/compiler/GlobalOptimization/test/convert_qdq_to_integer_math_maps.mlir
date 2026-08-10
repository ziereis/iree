// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(iree-global-opt-convert-qdq-to-integer-math))" %s | FileCheck %s

// Tests for the indexing maps, iterator types and result shapes the rewrite
// builds. The arithmetic is covered separately in
// convert_qdq_to_integer_math_algebra.mlir, and nothing here asserts on op
// bodies: a change to which correction terms are emitted, or to how operands are
// widened, must not affect this file.
//
// The emitted ops always appear in the order: integer contraction, lhs sum, rhs
// sum, epilogue. Each is identified below by that order and by its `outs` type.

//===----------------------------------------------------------------------===//
// Pullback
//
// The contraction's operand map is carried through the dequantize's own maps to
// give a map from the iteration space to the quantized input. The three ops the
// rewrite builds then all read that composed map rather than the original.
//===----------------------------------------------------------------------===//

// Baseline: plain matmul, everything an identity or a projected permutation.
// The integer contraction inherits the contraction's maps and iterator types
// unchanged, and the rhs sum drops M.
func.func @matmul_baseline(%aq: tensor<4x8xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<8x16xi8>, %sb: tensor<16xf32>) -> tensor<4x16xf32> {
  %ainit = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa, %za : tensor<4x8xi8>, tensor<f32>, tensor<i8>)
      outs(%ainit : tensor<4x8xf32>) -> tensor<4x8xf32>
  %binit = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb : tensor<8x16xi8>, tensor<16xf32>)
      outs(%binit : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @matmul_baseline(
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d2, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
// The rhs sum is stated over the two dims the rhs depends on, N and K, reducing
// only K. M never enters it.
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d0)>]
//  CHECK-SAME:     iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)

// -----

// matmul_transpose_b: the same per-channel rhs scale map as above, but the
// operand map places it on N instead of K. Composing the two is what decides
// legality, and the contraction's rhs map keeps the transposed form so no
// transpose is materialised.
func.func @matmul_transpose_b(%aq: tensor<4x8xi8>, %sa: tensor<f32>,
    %bq: tensor<16x8xi8>, %sb: tensor<16xf32>) -> tensor<4x16xf32> {
  %ainit = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa : tensor<4x8xi8>, tensor<f32>)
      outs(%ainit : tensor<4x8xf32>) -> tensor<4x8xf32>
  %binit = tensor.empty() : tensor<16x8xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb : tensor<16x8xi8>, tensor<16xf32>)
      outs(%binit : tensor<16x8xf32>) -> tensor<16x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d1, d2)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf32>, tensor<16x8xf32>)
      outs(%fill : tensor<4x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %m = arith.mulf %in, %in_0 : f32
    %s = arith.addf %out, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @matmul_transpose_b(
//       CHECK:   linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<4x8xi8>, tensor<16x8xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)

// -----

// A transpose folded into the dequantize: its output map is a non-identity
// permutation, so inverting it is what carries the contraction's operand map
// onto the 16x8 quantized data. The contraction reads the untransposed tensor
// through a permuted map, and no transpose op survives.
func.func @transposing_dequantize(%aq: tensor<4x8xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<16x8xi8>, %sb: tensor<16xf32>) -> tensor<4x16xf32> {
  %ainit = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa, %za : tensor<4x8xi8>, tensor<f32>, tensor<i8>)
      outs(%ainit : tensor<4x8xf32>) -> tensor<4x8xf32>
  %binit = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d1, d0)>]}
      ins(%bq, %sb : tensor<16x8xi8>, tensor<16xf32>)
      outs(%binit : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @transposing_dequantize(
//   CHECK-NOT:   linalg.transpose
// The contraction reads the 16x8 tensor directly, with the permutation absorbed
// into its rhs map.
//       CHECK:   linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2) -> (d1, d2)>
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<4x8xi8>, tensor<16x8xi8>)
// The sum sees the same untransposed layout, so N is its leading dim.
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d0)>]
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)

// -----

// A batch dim is a parallel dim like any other: the sum keeps it because the rhs
// depends on it, and drops only M.
func.func @batch_matmul(%aq: tensor<2x4x8xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<2x8x16xi8>, %sb: tensor<f32>) -> tensor<2x4x16xf32> {
  %ainit = tensor.empty() : tensor<2x4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %sa, %za : tensor<2x4x8xi8>, tensor<f32>, tensor<i8>)
      outs(%ainit : tensor<2x4x8xf32>) -> tensor<2x4x8xf32>
  %binit = tensor.empty() : tensor<2x8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %sb : tensor<2x8x16xi8>, tensor<f32>)
      outs(%binit : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x4x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  %c = linalg.batch_matmul ins(%a, %b : tensor<2x4x8xf32>, tensor<2x8x16xf32>)
      outs(%fill : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  return %c : tensor<2x4x16xf32>
}
// CHECK-LABEL: func.func @batch_matmul(
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<2x4x16xi32>)
// The sum keeps the batch dim and N, reducing only K.
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<2x16xi32>)

// -----

// Several M, N and K dims at once. The rewrite never names M, N or K, it only
// partitions reduction dims, so a multi-dim contraction needs no special
// handling: the sum keeps both N dims and reduces both K dims.
func.func @contract_multi_dim(%aq: tensor<2x4x3x8xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<3x8x5x16xi8>, %sb: tensor<5x16xf32>) -> tensor<2x4x5x16xf32> {
  %ainit = tensor.empty() : tensor<2x4x3x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%aq, %sa, %za : tensor<2x4x3x8xi8>, tensor<f32>, tensor<i8>)
      outs(%ainit : tensor<2x4x3x8xf32>) -> tensor<2x4x3x8xf32>
  %binit = tensor.empty() : tensor<3x8x5x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%bq, %sb : tensor<3x8x5x16xi8>, tensor<5x16xf32>)
      outs(%binit : tensor<3x8x5x16xf32>) -> tensor<3x8x5x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<2x4x5x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<2x4x5x16xf32>) -> tensor<2x4x5x16xf32>
  %c = linalg.contract
      indexing_maps = [affine_map<(m0, m1, n0, n1, k0, k1) -> (m0, m1, k0, k1)>,
                       affine_map<(m0, m1, n0, n1, k0, k1) -> (k0, k1, n0, n1)>,
                       affine_map<(m0, m1, n0, n1, k0, k1) -> (m0, m1, n0, n1)>]
      ins(%a, %b : tensor<2x4x3x8xf32>, tensor<3x8x5x16xf32>)
      outs(%fill : tensor<2x4x5x16xf32>) -> tensor<2x4x5x16xf32>
  return %c : tensor<2x4x5x16xf32>
}
// CHECK-LABEL: func.func @contract_multi_dim(
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<2x4x5x16xi32>)
// Both N dims survive into the sum, both K dims are reduced, and both M dims are
// dropped.
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d2, d3, d0, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<5x16xi32>)

//===----------------------------------------------------------------------===//
// Quantization parameter maps in the epilogue
//
// The map that decided legality is reused to index the scale and zero point,
// after being renumbered into the epilogue's iteration space.
//===----------------------------------------------------------------------===//

// -----

// Per-channel on both sides, along different result dims. Each parameter lands
// on its own epilogue dim, so a lhs parameter indexes M and a rhs parameter
// indexes N. Swapping them would still typecheck.
func.func @per_channel_different_dims(%aq: tensor<4x8xi8>, %sa: tensor<4xf32>, %za: tensor<4xi8>,
    %bq: tensor<8x16xi8>, %sb: tensor<16xf32>, %zb: tensor<16xi8>) -> tensor<4x16xf32> {
  %ainit = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa, %za : tensor<4x8xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%ainit : tensor<4x8xf32>) -> tensor<4x8xf32>
  %binit = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb, %zb : tensor<8x16xi8>, tensor<16xf32>, tensor<16xi8>)
      outs(%binit : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @per_channel_different_dims(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[SA:[a-zA-Z0-9_]+]]: tensor<4xf32>, %[[ZA:[a-zA-Z0-9_]+]]: tensor<4xi8>,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[SB:[a-zA-Z0-9_]+]]: tensor<16xf32>, %[[ZB:[a-zA-Z0-9_]+]]: tensor<16xi8>
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
// The lhs sum keeps M, the rhs sum keeps N.
//       CHECK:   %[[RA:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[AQ]] : tensor<4x8xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4xi32>)
//       CHECK:   %[[RB:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[BQ]] : tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
// The epilogue's maps are listed in operand order, so the two lists below read
// together: the contraction over both dims, then the lhs sum over M, the rhs
// zero point over N, the rhs sum over N, and the lhs zero point and lhs scale
// over M. Checking either list alone would not say which map belongs to which
// operand.
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d0)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d0)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d0)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel"]
//  CHECK-SAME:     ins(%[[D]], %[[RA]], %[[ZB]], %[[RB]], %[[ZA]], %[[SA]], %[[SB]] :
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)

//===----------------------------------------------------------------------===//
// Sum construction
//
// A sum is stated over the dims its operand depends on, compressed to their own
// numbering. Which dims survive, and in what order, is what the epilogue then
// has to match.
//===----------------------------------------------------------------------===//

// -----

// A GEMV, where the sum reduces as many elements as the contraction multiplies.
// It is still a single pass over the operand, and over constant weights it folds
// entirely, so it is built rather than declined.
func.func @gemv(%aq: tensor<1x128xi8>, %sa: f32, %za: i64,
    %bq: tensor<10x128xi8>, %sb: tensor<10xf32>) -> tensor<1x10xf32> {
  %ainit = tensor.empty() : tensor<1x128xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa, %za : tensor<1x128xi8>, f32, i64)
      outs(%ainit : tensor<1x128xf32>) -> tensor<1x128xf32>
  %binit = tensor.empty() : tensor<10x128xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb : tensor<10x128xi8>, tensor<10xf32>)
      outs(%binit : tensor<10x128xf32>) -> tensor<10x128xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x10xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x10xf32>) -> tensor<1x10xf32>
  %c = linalg.matmul
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                       affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1)>]
      ins(%a, %b : tensor<1x128xf32>, tensor<10x128xf32>)
      outs(%fill : tensor<1x10xf32>) -> tensor<1x10xf32>
  return %c : tensor<1x10xf32>
}
// CHECK-LABEL: func.func @gemv(
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x10xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<10xi32>)

//===----------------------------------------------------------------------===//
// Shape-only witness operand
//
// linalg recovers loop extents by inverting the concatenated operand maps, which
// only recovers a dim appearing as a bare result. A convolution's window dims
// appear only inside `oh * stride + kh * dilation`, so a shape-only operand
// naming them is appended and ignored by the body.
//===----------------------------------------------------------------------===//

// -----

// Convolution with asymmetric weights, which is the case needing the image sum.
// The image does not depend on F, so the sum drops it entirely and reduces C as
// well, leaving (N, OH, OW). That is what makes it 1/F the size of the
// convolution.
func.func @conv_2d_nhwc_hwcf_image_sum(%aq: tensor<1x8x8x4xi8>, %sa: tensor<f32>,
    %bq: tensor<3x3x4x16xi8>, %sb: tensor<16xf32>, %zb: tensor<16xi8>) -> tensor<1x6x6x16xf32> {
  %ai = tensor.empty() : tensor<1x8x8x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%aq, %sa : tensor<1x8x8x4xi8>, tensor<f32>)
      outs(%ai : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
  %bi = tensor.empty() : tensor<3x3x4x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%bq, %sb, %zb : tensor<3x3x4x16xi8>, tensor<16xf32>, tensor<16xi8>)
      outs(%bi : tensor<3x3x4x16xf32>) -> tensor<3x3x4x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x6x6x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x8x8x4xf32>, tensor<3x3x4x16xf32>)
      outs(%fill : tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  return %c : tensor<1x6x6x16xf32>
}
// CHECK-LABEL: func.func @conv_2d_nhwc_hwcf_image_sum(
//       CHECK:   linalg.generic
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x6x6x16xi32>)
// The window dims are named by a shape-only operand, sized from the filter.
//       CHECK:   %[[W:.+]] = tensor.empty() : tensor<3x3xi8>
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d3, d2 + d4, d5)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]
//  CHECK-SAME:     ins(%{{.+}}, %[[W]] : tensor<1x8x8x4xi8>, tensor<3x3xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x6x6xi32>)

// -----

// Depthwise: there is no F to drop, so the image sum keeps C and comes out the
// full output shape. Every dim of the convolution survives into it, which is the
// worst case for the rewrite and still worth building.
func.func @depthwise_conv_image_sum(%aq: tensor<1x8x8x4xi8>, %sa: tensor<f32>,
    %bq: tensor<3x3x4xi8>, %sb: tensor<4xf32>, %zb: tensor<4xi8>) -> tensor<1x6x6x4xf32> {
  %ainit = tensor.empty() : tensor<1x8x8x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%aq, %sa : tensor<1x8x8x4xi8>, tensor<f32>)
      outs(%ainit : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
  %binit = tensor.empty() : tensor<3x3x4xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %sb, %zb : tensor<3x3x4xi8>, tensor<4xf32>, tensor<4xi8>)
      outs(%binit : tensor<3x3x4xf32>) -> tensor<3x3x4xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x6x6x4xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>
  %c = linalg.depthwise_conv_2d_nhwc_hwc {dilations = dense<1> : tensor<2xi64>,
                                          strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x8x8x4xf32>, tensor<3x3x4xf32>)
      outs(%fill : tensor<1x6x6x4xf32>) -> tensor<1x6x6x4xf32>
  return %c : tensor<1x6x6x4xf32>
}
// CHECK-LABEL: func.func @depthwise_conv_image_sum(
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5, d3)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x6x6x4xi32>)
// The image sum has the same iteration space and the same result shape as the
// contraction above it, because C is kept rather than reduced: the image map and
// the result map below are the same two maps again, with only the filter operand
// replaced by the window witness.
//       CHECK:   %[[W:.+]] = tensor.empty() : tensor<3x3xi8>
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 + d4, d2 + d5, d3)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]
//  CHECK-SAME:     ins(%{{.+}}, %[[W]] : tensor<1x8x8x4xi8>, tensor<3x3xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x6x6x4xi32>)

// -----

// Strides and dilations survive into the image sum's map, so it reads the same
// window the convolution does. Getting either factor wrong here would sum the
// wrong elements.
func.func @conv_strided_dilated_image_sum(%aq: tensor<1x7x7x2xi8>, %sa: tensor<f32>,
    %bq: tensor<2x2x2x2xi8>, %sb: tensor<2xf32>, %zb: tensor<2xi8>) -> tensor<1x3x3x2xf32> {
  %ai = tensor.empty() : tensor<1x7x7x2xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%aq, %sa : tensor<1x7x7x2xi8>, tensor<f32>)
      outs(%ai : tensor<1x7x7x2xf32>) -> tensor<1x7x7x2xf32>
  %bi = tensor.empty() : tensor<2x2x2x2xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%bq, %sb, %zb : tensor<2x2x2x2xi8>, tensor<2xf32>, tensor<2xi8>)
      outs(%bi : tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x3x3x2xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<2> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>}
      ins(%a, %b : tensor<1x7x7x2xf32>, tensor<2x2x2x2xf32>)
      outs(%fill : tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32>
  return %c : tensor<1x3x3x2xf32>
}
// CHECK-LABEL: func.func @conv_strided_dilated_image_sum(
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x3x3x2xi32>)
//       CHECK:   %[[W:.+]] = tensor.empty() : tensor<2x2xi8>
//       CHECK:   linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1 * 2 + d3 * 2, d2 * 2 + d4 * 2, d5)>
//  CHECK-SAME:     ins(%{{.+}}, %[[W]] : tensor<1x7x7x2xi8>, tensor<2x2xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x3x3xi32>)

// -----

// A convolution whose weights are symmetric needs only the filter sum, and the
// filter map is a projected permutation, so every dim is already named as a bare
// coordinate. No witness operand is emitted.
func.func @no_witness_when_map_is_projected_permutation(%aq: tensor<1x8x8x4xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<3x3x4x16xi8>, %sb: tensor<16xf32>) -> tensor<1x6x6x16xf32> {
  %ai = tensor.empty() : tensor<1x8x8x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%aq, %sa, %za : tensor<1x8x8x4xi8>, tensor<f32>, tensor<i8>)
      outs(%ai : tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
  %bi = tensor.empty() : tensor<3x3x4x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%bq, %sb : tensor<3x3x4x16xi8>, tensor<16xf32>)
      outs(%bi : tensor<3x3x4x16xf32>) -> tensor<3x3x4x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x6x6x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  %c = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x8x8x4xf32>, tensor<3x3x4x16xf32>)
      outs(%fill : tensor<1x6x6x16xf32>) -> tensor<1x6x6x16xf32>
  return %c : tensor<1x6x6x16xf32>
}
// CHECK-LABEL: func.func @no_witness_when_map_is_projected_permutation(
//   CHECK-NOT:   tensor.empty() : tensor<3x3xi8>
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x6x6x16xi32>)
// The filter sum takes one input and no witness, keeping only F.
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d1, d2, d3, d0)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3) -> (d0)>]
//  CHECK-SAME:     iterator_types = ["parallel", "reduction", "reduction", "reduction"]
//  CHECK-SAME:     ins(%{{.+}} : tensor<3x3x4x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)

// -----

// The channel-first layout, to show the construction does not depend on where
// the spatial dims sit. The image map carries the window in its last two
// results rather than its middle two, and the filter sum keeps F.
func.func @conv_2d_nchw_fchw(%aq: tensor<1x4x8x8xi8>, %sa: f32, %za: i64,
    %bq: tensor<16x4x3x3xi8>, %sb: tensor<16xf32>) -> tensor<1x16x6x6xf32> {
  %ai = tensor.empty() : tensor<1x4x8x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned,
       indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> ()>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%aq, %sa, %za : tensor<1x4x8x8xi8>, f32, i64)
      outs(%ai : tensor<1x4x8x8xf32>) -> tensor<1x4x8x8xf32>
  %bi = tensor.empty() : tensor<16x4x3x3xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                        affine_map<(d0, d1, d2, d3) -> (d0)>,
                        affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]}
      ins(%bq, %sb : tensor<16x4x3x3xi8>, tensor<16xf32>)
      outs(%bi : tensor<16x4x3x3xf32>) -> tensor<16x4x3x3xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<1x16x6x6xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<1x16x6x6xf32>) -> tensor<1x16x6x6xf32>
  %c = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
      ins(%a, %b : tensor<1x4x8x8xf32>, tensor<16x4x3x3xf32>)
      outs(%fill : tensor<1x16x6x6xf32>) -> tensor<1x16x6x6xf32>
  return %c : tensor<1x16x6x6xf32>
}
// CHECK-LABEL: func.func @conv_2d_nchw_fchw(
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>]
//  CHECK-SAME:     outs(%{{.+}} : tensor<1x16x6x6xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2, d3) -> (d0)>]
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)

//===----------------------------------------------------------------------===//
// Promotion of the dims the parameters vary along
//
// In the partial form those reduction dims become parallel in the integer
// contraction, are appended to its result, and reappear as the epilogue's
// trailing reduction.
//===----------------------------------------------------------------------===//

// -----

// Blockwise with an asymmetric lhs. K is expanded to (G, L), the lhs parameters
// vary along G, so G stays parallel in the integer contraction and the epilogue
// reduces across it.
func.func @blockwise_promotes_block_dim(%aq: tensor<4x2x8xi8>, %sa: tensor<4x2xf32>, %za: tensor<4x2xi8>,
    %bq: tensor<2x8x16xi8>, %sb: f32) -> tensor<4x16xf32> {
  %ainit = tensor.empty() : tensor<4x2x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %sa, %za : tensor<4x2x8xi8>, tensor<4x2xf32>, tensor<4x2xi8>)
      outs(%ainit : tensor<4x2x8xf32>) -> tensor<4x2x8xf32>
  %binit = tensor.empty() : tensor<2x8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %sb : tensor<2x8x16xi8>, f32)
      outs(%binit : tensor<2x8x16xf32>) -> tensor<2x8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.contract
      indexing_maps = [affine_map<(m, n, g, l) -> (m, g, l)>,
                       affine_map<(m, n, g, l) -> (g, l, n)>,
                       affine_map<(m, n, g, l) -> (m, n)>]
      ins(%a, %b : tensor<4x2x8xf32>, tensor<2x8x16xf32>)
      outs(%fill : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @blockwise_promotes_block_dim(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x2x8xi8>, %[[SA:[a-zA-Z0-9_]+]]: tensor<4x2xf32>, %[[ZA:[a-zA-Z0-9_]+]]: tensor<4x2xi8>,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<2x8x16xi8>, %[[SB:[a-zA-Z0-9_]+]]: f32
// G is parallel here, and appended to the result, so the contraction yields one
// partial per block.
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[AQ]], %[[BQ]] : tensor<4x2x8xi8>, tensor<2x8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16x2xi32>)
// The rhs sum keeps N and G, reducing only L. The kept dims come out in
// iteration order, so it is (N, G) rather than (G, N), which is the trailing
// pair of the contraction's (M, N, G) and needs no transpose below.
//       CHECK:   %[[RB:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2, d0)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[BQ]] : tensor<2x8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<16x2xi32>)
// The epilogue's iteration space is the result dims followed by G, and it
// reduces across G. Its result map is the same (M, N) map the sum above writes.
// Reading the two lists in step: the contraction over (M, N, G), the rhs sum
// over (N, G), then the lhs zero point and lhs scale over (M, G) because they
// are per block, and the rhs scale over the whole tensor.
//       CHECK:   linalg.generic
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d1, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> ()>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>]
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
//  CHECK-SAME:     ins(%[[D]], %[[RB]], %[[ZA]], %[[SA]], %[[SB]] :
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)

//===----------------------------------------------------------------------===//
// Dynamic shapes
//
// Extents are read from the quantized inputs rather than from the dequantized
// values, and result dims are taken from the original init so the replacement
// carries the original result type.
//===----------------------------------------------------------------------===//

// -----

// A dynamic reduction extent. The N*zA*zB term needs it as a value, so it is
// materialised from the operand rather than folded to a constant.
func.func @dynamic_reduction(%aq: tensor<4x?xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<?x16xi8>, %sb: tensor<f32>, %zb: tensor<i8>) -> tensor<4x16xf32> {
  %c1 = arith.constant 1 : index
  %k = tensor.dim %aq, %c1 : tensor<4x?xi8>
  %ainit = tensor.empty(%k) : tensor<4x?xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa, %za : tensor<4x?xi8>, tensor<f32>, tensor<i8>)
      outs(%ainit : tensor<4x?xf32>) -> tensor<4x?xf32>
  %binit = tensor.empty(%k) : tensor<?x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb, %zb : tensor<?x16xi8>, tensor<f32>, tensor<i8>)
      outs(%binit : tensor<?x16xf32>) -> tensor<?x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<4x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x?xf32>, tensor<?x16xf32>)
      outs(%fill : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @dynamic_reduction(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x?xi8>
// The extent is read from the quantized input, never from a dequantize result.
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   tensor.dim %[[AQ]], %[[C1]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
//       CHECK:   arith.index_cast

// -----

// A dynamic parallel dim propagates into the contraction, the lhs sum and the
// epilogue, all of which carry it as a dynamic extent.
func.func @dynamic_parallel_dim(%aq: tensor<?x8xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<8x16xi8>, %sb: tensor<f32>, %zb: tensor<i8>) -> tensor<?x16xf32> {
  %c0 = arith.constant 0 : index
  %m = tensor.dim %aq, %c0 : tensor<?x8xi8>
  %ainit = tensor.empty(%m) : tensor<?x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa, %za : tensor<?x8xi8>, tensor<f32>, tensor<i8>)
      outs(%ainit : tensor<?x8xf32>) -> tensor<?x8xf32>
  %binit = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb, %zb : tensor<8x16xi8>, tensor<f32>, tensor<i8>)
      outs(%binit : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty(%m) : tensor<?x16xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x16xf32>) -> tensor<?x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<?x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<?x16xf32>) -> tensor<?x16xf32>
  return %c : tensor<?x16xf32>
}
// CHECK-LABEL: func.func @dynamic_parallel_dim(
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<?x16xi32>)
// The lhs sum inherits the dynamic dim; the rhs sum does not depend on it.
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<?xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<?x16xf32>)

// -----

// The inputs know M statically but the result type does not. Sizes for result
// dims are taken from the init rather than from whichever operand mentions them
// first, so the replacement keeps the original dynamic result type.
func.func @static_input_dim_dynamic_result(%aq: tensor<4x8xi8>, %sa: tensor<f32>, %za: tensor<i8>,
    %bq: tensor<8x16xi8>, %sb: tensor<f32>, %init: tensor<?x16xf32>) -> tensor<?x16xf32> {
  %ainit = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %sa, %za : tensor<4x8xi8>, tensor<f32>, tensor<i8>)
      outs(%ainit : tensor<4x8xf32>) -> tensor<4x8xf32>
  %binit = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %sb : tensor<8x16xi8>, tensor<f32>)
      outs(%binit : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<?x16xf32>) -> tensor<?x16xf32>
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d2, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%fill : tensor<?x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %m = arith.mulf %in, %in_0 : f32
    %s = arith.addf %out, %m : f32
    linalg.yield %s : f32
  } -> tensor<?x16xf32>
  return %c : tensor<?x16xf32>
}
// CHECK-LABEL: func.func @static_input_dim_dynamic_result(
// Both the integer contraction and the epilogue keep the dynamic result type,
// even though the lhs knows M is 4.
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<4x8xi8>, tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<?x16xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<?x16xf32>)
