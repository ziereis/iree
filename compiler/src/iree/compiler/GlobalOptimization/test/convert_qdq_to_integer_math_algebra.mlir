// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-global-opt-convert-qdq-to-integer-math))" %s | FileCheck %s

// Tests for the arithmetic the rewrite emits: which correction terms exist, how
// operands are widened, and when the rewrite declines. The indexing maps and
// iterator types are covered separately in
// convert_qdq_to_integer_math_maps.mlir

// Both symmetric: nothing to correct, so the epilogue only applies the scales.
func.func @zp_neither(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_neither(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
// The integer contraction reads both quantized inputs and accumulates the
// product into the init.
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[AQ]], %[[BQ]] : tensor<4x8xi8>, tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
//       CHECK:   ^bb0(%[[L:[a-zA-Z0-9_]+]]: i8, %[[R:[a-zA-Z0-9_]+]]: i8, %[[ACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[LE:.+]] = arith.extsi %[[L]]
//       CHECK:     %[[RE:.+]] = arith.extsi %[[R]]
//       CHECK:     %[[P:.+]] = arith.muli %[[LE]], %[[RE]]
//       CHECK:     arith.addi %[[ACC]], %[[P]]
// The epilogue takes only the contraction and the two scales: with no zero point
// on either side there is no sum to correct with.
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[D]], %[[AS]], %[[BS]] :
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)
//       CHECK:   ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[ESA:[a-zA-Z0-9_]+]]: f32, %[[ESB:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32):
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[ED]]
//       CHECK:     %[[S:.+]] = arith.mulf %[[ESA]], %[[ESB]]
//       CHECK:     arith.mulf %[[REAL]], %[[S]]
//   CHECK-NOT:     arith.subi

// -----

// A zero point on the lhs produces the *rhs* sum and the `- zA*RB` term.
func.func @zp_lhs_only(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_lhs_only(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32, %[[AZ:[a-zA-Z0-9_]+]]: i8,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
// One sum, and it reduces the *rhs*, which is what pairs with the lhs zero
// point. Reducing the lhs here would be the wrong term.
//       CHECK:   %[[RB:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[BQ]] : tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
// It is a plain accumulation of the widened operand, nothing more.
//       CHECK:   ^bb0(%[[SV:[a-zA-Z0-9_]+]]: i8, %[[SACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[SVE:.+]] = arith.extsi %[[SV]]
//       CHECK:     arith.addi %[[SACC]], %[[SVE]]
// The epilogue pairs the lhs zero point with that rhs sum, and subtracts the
// product *from* the contraction rather than the other way round.
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[D]], %[[RB]], %[[AZ]], %[[AS]], %[[BS]] :
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)
//       CHECK:   ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[ERB:[a-zA-Z0-9_]+]]: i32, %[[EZA:[a-zA-Z0-9_]+]]: i8, %[[ESA:[a-zA-Z0-9_]+]]: f32, %[[ESB:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32):
//       CHECK:     %[[ZAE:.+]] = arith.extsi %[[EZA]]
//       CHECK:     %[[T:.+]] = arith.muli %[[ZAE]], %[[ERB]]
//       CHECK:     %[[C:.+]] = arith.subi %[[ED]], %[[T]]
//   CHECK-NOT:     arith.subi
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[C]]
//       CHECK:     %[[S:.+]] = arith.mulf %[[ESA]], %[[ESB]]
//       CHECK:     arith.mulf %[[REAL]], %[[S]]

// -----

// The mirror image: a zero point on the rhs produces the lhs sum.
func.func @zp_rhs_only(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_rhs_only(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32, %[[BZ:[a-zA-Z0-9_]+]]: i8
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
// The one sum reduces the lhs, and keeps M rather than N.
//       CHECK:   %[[RA:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[AQ]] : tensor<4x8xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4xi32>)
//       CHECK:   ^bb0(%[[SV:[a-zA-Z0-9_]+]]: i8, %[[SACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[SVE:.+]] = arith.extsi %[[SV]]
//       CHECK:     arith.addi %[[SACC]], %[[SVE]]
// The mirror pairing: the rhs zero point multiplies the lhs sum.
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[D]], %[[RA]], %[[BZ]], %[[AS]], %[[BS]] :
//       CHECK:   ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[ERA:[a-zA-Z0-9_]+]]: i32, %[[EZB:[a-zA-Z0-9_]+]]: i8, %{{.+}}: f32, %{{.+}}: f32, %{{.+}}: f32):
//       CHECK:     %[[ZBE:.+]] = arith.extsi %[[EZB]]
//       CHECK:     %[[T:.+]] = arith.muli %[[ZBE]], %[[ERA]]
//       CHECK:     %[[C:.+]] = arith.subi %[[ED]], %[[T]]
//   CHECK-NOT:     arith.subi
//       CHECK:     arith.sitofp %[[C]]

// -----

// Both asymmetric: all four terms, including the N*zA*zB constant that only
// exists when both sides carry a zero point. N is the reduction extent, 8 here.
func.func @zp_both(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_both(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x8xi8>, %[[AS:[a-zA-Z0-9_]+]]: f32, %[[AZ:[a-zA-Z0-9_]+]]: i8,
//  CHECK-SAME:   %[[BQ:[a-zA-Z0-9_]+]]: tensor<8x16xi8>, %[[BS:[a-zA-Z0-9_]+]]: f32, %[[BZ:[a-zA-Z0-9_]+]]: i8
//   CHECK-DAG:   %[[N:.+]] = arith.constant 8 : i32
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
// Both sums exist: the lhs sum keeps M, the rhs sum keeps N. Each is a plain
// accumulation of its own widened operand.
//       CHECK:   %[[RA:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[AQ]] : tensor<4x8xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4xi32>)
//       CHECK:   ^bb0(%[[AV:[a-zA-Z0-9_]+]]: i8, %[[AACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[AVE:.+]] = arith.extsi %[[AV]]
//       CHECK:     arith.addi %[[AACC]], %[[AVE]]
//       CHECK:   %[[RB:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     ins(%[[BQ]] : tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
//       CHECK:   ^bb0(%[[BV:[a-zA-Z0-9_]+]]: i8, %[[BACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[BVE:.+]] = arith.extsi %[[BV]]
//       CHECK:     arith.addi %[[BACC]], %[[BVE]]
// The operand list is where the crossing is visible: the lhs sum sits next to
// the *rhs* zero point, and the rhs sum next to the lhs one.
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[D]], %[[RA]], %[[BZ]], %[[RB]], %[[AZ]], %[[AS]], %[[BS]] :
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)
//       CHECK:   ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[ERA:[a-zA-Z0-9_]+]]: i32, %[[EZB:[a-zA-Z0-9_]+]]: i8, %[[ERB:[a-zA-Z0-9_]+]]: i32, %[[EZA:[a-zA-Z0-9_]+]]: i8, %[[ESA:[a-zA-Z0-9_]+]]: f32, %[[ESB:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f32):
//       CHECK:     %[[ZAE:.+]] = arith.extsi %[[EZA]]
//       CHECK:     %[[ZBE:.+]] = arith.extsi %[[EZB]]
// - zB*RA, subtracted from the contraction.
//       CHECK:     %[[T1:.+]] = arith.muli %[[ZBE]], %[[ERA]]
//       CHECK:     %[[C1:.+]] = arith.subi %[[ED]], %[[T1]]
// - zA*RB, subtracted from the running value.
//       CHECK:     %[[T2:.+]] = arith.muli %[[ZAE]], %[[ERB]]
//       CHECK:     %[[C2:.+]] = arith.subi %[[C1]], %[[T2]]
// + N*zA*zB, added back.
//       CHECK:     %[[ZZ:.+]] = arith.muli %[[ZAE]], %[[ZBE]]
//       CHECK:     %[[T3:.+]] = arith.muli %[[ZZ]], %[[N]]
//       CHECK:     %[[C3:.+]] = arith.addi %[[C2]], %[[T3]]
// Only now is the accumulator converted, and both scales applied to it.
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[C3]]
//       CHECK:     %[[S:.+]] = arith.mulf %[[ESA]], %[[ESB]]
//       CHECK:     arith.mulf %[[REAL]], %[[S]]

// -----

// The same four terms, but with a dynamic reduction extent. N is no longer a
// folded constant: it is read from the reduction dim of the quantized lhs and
// converted. Everything a lit test can establish here is that the value fed to
// the N*zA*zB term is derived from the right dim; that it is numerically right
// at runtime is not checked anywhere.
func.func @zp_both_dynamic_reduction(%aq: tensor<4x?xi8>, %a_s: f32, %a_z: i8,
    %bq: tensor<?x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %c1 = arith.constant 1 : index
  %k = tensor.dim %aq, %c1 : tensor<4x?xi8>
  %a_i = tensor.empty(%k) : tensor<4x?xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x?xi8>, f32, i8)
      outs(%a_i : tensor<4x?xf32>) -> tensor<4x?xf32>
  %b_i = tensor.empty(%k) : tensor<?x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<?x16xi8>, f32, i8)
      outs(%b_i : tensor<?x16xf32>) -> tensor<?x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x?xf32>, tensor<?x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_both_dynamic_reduction(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x?xi8>
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-NOT:   arith.constant {{.*}} : i32
// The extent comes from the quantized lhs, not from a dequantize result.
//       CHECK:   %[[K:.+]] = tensor.dim %[[AQ]], %[[C1]]
//       CHECK:   %[[N:.+]] = arith.index_cast %[[K]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)
//       CHECK:   ^bb0(%{{.+}}: i32, %{{.+}}: i32, %[[EZB:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %[[EZA:[a-zA-Z0-9_]+]]: i8, %{{.+}}: f32, %{{.+}}: f32, %{{.+}}: f32):
//       CHECK:     %[[ZAE:.+]] = arith.extsi %[[EZA]]
//       CHECK:     %[[ZBE:.+]] = arith.extsi %[[EZB]]
//       CHECK:     %[[ZZ:.+]] = arith.muli %[[ZAE]], %[[ZBE]]
//       CHECK:     arith.muli %[[ZZ]], %[[N]]

// -----

// Two reduction dims, both dynamic, so N is the product of the two extents.
// This is the only test covering that accumulation over `intReductionDims`;
// with a single reduction dim the loop degenerates to one term.
func.func @zp_both_dynamic_multi_dim_reduction(%aq: tensor<4x?x?xi8>, %a_s: f32, %a_z: i8,
    %bq: tensor<?x?x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %k0 = tensor.dim %aq, %c1 : tensor<4x?x?xi8>
  %k1 = tensor.dim %aq, %c2 : tensor<4x?x?xi8>
  %a_i = tensor.empty(%k0, %k1) : tensor<4x?x?xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %a_s, %a_z : tensor<4x?x?xi8>, f32, i8)
      outs(%a_i : tensor<4x?x?xf32>) -> tensor<4x?x?xf32>
  %b_i = tensor.empty(%k0, %k1) : tensor<?x?x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %b_s, %b_z : tensor<?x?x16xi8>, f32, i8)
      outs(%b_i : tensor<?x?x16xf32>) -> tensor<?x?x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.contract
      indexing_maps = [affine_map<(m, n, k0, k1) -> (m, k0, k1)>,
                       affine_map<(m, n, k0, k1) -> (k0, k1, n)>,
                       affine_map<(m, n, k0, k1) -> (m, n)>]
      ins(%a, %b : tensor<4x?x?xf32>, tensor<?x?x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_both_dynamic_multi_dim_reduction(
//  CHECK-SAME:   %[[AQ:[a-zA-Z0-9_]+]]: tensor<4x?x?xi8>
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// N is the product of both reduction extents, taken in index arithmetic and
// converted once.
//       CHECK:   %[[K0:.+]] = tensor.dim %[[AQ]], %[[C1]]
//       CHECK:   %[[K1:.+]] = tensor.dim %[[AQ]], %[[C2]]
//       CHECK:   %[[PROD:.+]] = arith.muli %[[K0]], %[[K1]]
//       CHECK:   %[[N:.+]] = arith.index_cast %[[PROD]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)
//       CHECK:   ^bb0(%{{.+}}: i32, %{{.+}}: i32, %[[EZB:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %[[EZA:[a-zA-Z0-9_]+]]: i8, %{{.+}}: f32, %{{.+}}: f32, %{{.+}}: f32):
//       CHECK:     %[[ZAE:.+]] = arith.extsi %[[EZA]]
//       CHECK:     %[[ZBE:.+]] = arith.extsi %[[EZB]]
//       CHECK:     %[[ZZ:.+]] = arith.muli %[[ZAE]], %[[ZBE]]
//       CHECK:     arith.muli %[[ZZ]], %[[N]]

//===----------------------------------------------------------------------===//
// Signedness
//
// MLIR integers are signless, so how a value widens is carried by the
// dequantize's attributes rather than by its type. `input_unsigned` governs the
// quantized data and `zp_unsigned` the zero point, and they are independent.
//===----------------------------------------------------------------------===//

// -----

// Unsigned storage and unsigned zero points: every widening is a zero extension.
// The final conversion stays signed, because the accumulator is signed by
// construction whatever the inputs were.
func.func @both_unsigned(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @both_unsigned(
//   CHECK-NOT:   arith.extsi
//       CHECK:   linalg.generic
//       CHECK:     arith.extui
//       CHECK:     arith.extui
// Both sums zero extend as well.
//       CHECK:   linalg.generic
//       CHECK:     arith.extui
//       CHECK:   linalg.generic
//       CHECK:     arith.extui
// The zero points zero extend in the epilogue, and the accumulator still
// converts as signed.
//       CHECK:   linalg.generic
//       CHECK:     arith.extui
//       CHECK:     arith.extui
//       CHECK:     arith.sitofp

// -----

// The combination PT2E emits on x86: unsigned activations against signed
// weights. The two operands must widen differently within the same op.
func.func @lhs_unsigned_rhs_signed(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i8, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned, zp_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i8)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @lhs_unsigned_rhs_signed(
// The contraction widens its two operands differently, and each extension is
// tied to the operand it belongs to rather than merely appearing in order.
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
//       CHECK:   ^bb0(%[[L:[a-zA-Z0-9_]+]]: i8, %[[R:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32):
//       CHECK:     %[[LE:.+]] = arith.extui %[[L]]
//       CHECK:     %[[RE:.+]] = arith.extsi %[[R]]
//       CHECK:     arith.muli %[[LE]], %[[RE]]
// Each sum follows the signedness of the operand it reduces: the lhs sum zero
// extends, the rhs sum sign extends.
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4xi32>)
//       CHECK:   ^bb0(%[[AV:[a-zA-Z0-9_]+]]: i8, %[[AACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[AVE:.+]] = arith.extui %[[AV]]
//       CHECK:     arith.addi %[[AACC]], %[[AVE]]
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
//       CHECK:   ^bb0(%[[BV:[a-zA-Z0-9_]+]]: i8, %[[BACC:[a-zA-Z0-9_]+]]: i32):
//       CHECK:     %[[BVE:.+]] = arith.extsi %[[BV]]
//       CHECK:     arith.addi %[[BACC]], %[[BVE]]
// The zero points follow their own attribute, not the storage one: the lhs zero
// point is declared unsigned and the rhs signed.
//       CHECK:   ^bb0(%{{.+}}: i32, %{{.+}}: i32, %[[EZB:[a-zA-Z0-9_]+]]: i8, %{{.+}}: i32, %[[EZA:[a-zA-Z0-9_]+]]: i8, %{{.+}}: f32, %{{.+}}: f32, %{{.+}}: f32):
//       CHECK:     arith.extui %[[EZA]]
//       CHECK:     arith.extsi %[[EZB]]

// -----

// `input_unsigned` and `zp_unsigned` are independent: an unsigned tensor may
// carry a signed zero point of a wider type, which is what ONNX and TFLite
// importers produce. The data zero extends while the zero point sign extends.
func.func @zp_signedness_differs_from_storage(%aq: tensor<4x8xi8>, %a_s: f32, %a_z: i32, %bq: tensor<8x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {input_unsigned,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi8>, f32, i32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<8x16xi8>, f32, i8)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @zp_signedness_differs_from_storage(
// The lhs data is unsigned, so the contraction zero extends it.
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
//       CHECK:     arith.extui
//       CHECK:     arith.extsi
// The lhs sum reads the same unsigned data.
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4xi32>)
//       CHECK:     arith.extui
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<16xi32>)
//       CHECK:     arith.extsi
// The lhs zero point is already i32 and signed, so it needs no extension at all,
// while the rhs one sign extends from i8.
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)
//   CHECK-NOT:     arith.extui
//       CHECK:     arith.sitofp

//===----------------------------------------------------------------------===//
// Form
//
// Reduction dims split by whether a quantization parameter varies along them.
// Only the arithmetic consequence is checked here; the resulting maps and
// iterator types belong to the maps file.
//===----------------------------------------------------------------------===//

// -----

// Blockwise: K is expanded to (G, L) and the lhs parameters vary along G, so the
// contraction is integer within a block and the residual sum across blocks
// happens in floating point. The observable is the trailing addf in the
// epilogue, which no other form produces.
func.func @form_partial_reduces_in_float(%aq: tensor<4x2x4xi8>, %a_s: tensor<4x2xf32>, %a_z: tensor<4x2xi8>,
    %bq: tensor<2x4x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x2x4xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %a_s, %a_z : tensor<4x2x4xi8>, tensor<4x2xf32>, tensor<4x2xi8>)
      outs(%a_i : tensor<4x2x4xf32>) -> tensor<4x2x4xf32>
  %b_i = tensor.empty() : tensor<2x4x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %b_s : tensor<2x4x16xi8>, f32)
      outs(%b_i : tensor<2x4x16xf32>) -> tensor<2x4x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.contract
      indexing_maps = [affine_map<(m, n, g, l) -> (m, g, l)>,
                       affine_map<(m, n, g, l) -> (g, l, n)>,
                       affine_map<(m, n, g, l) -> (m, n)>]
      ins(%a, %b : tensor<4x2x4xf32>, tensor<2x4x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @form_partial_reduces_in_float(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.generic
//       CHECK:   linalg.generic
// The epilogue applies the scales per block and then accumulates across blocks
// in floating point. The addf takes the running accumulator as its first
// operand, so it really is accumulating rather than overwriting.
//       CHECK:   linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf32>)
//       CHECK:   ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[ERB:[a-zA-Z0-9_]+]]: i32, %[[EZA:[a-zA-Z0-9_]+]]: i8, %[[ESA:[a-zA-Z0-9_]+]]: f32, %[[ESB:[a-zA-Z0-9_]+]]: f32, %[[ACC:[a-zA-Z0-9_]+]]: f32):
//       CHECK:     %[[ZAE:.+]] = arith.extsi %[[EZA]]
//       CHECK:     %[[T:.+]] = arith.muli %[[ZAE]], %[[ERB]]
//       CHECK:     %[[C:.+]] = arith.subi %[[ED]], %[[T]]
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[C]]
//       CHECK:     %[[S:.+]] = arith.mulf %[[ESA]], %[[ESB]]
//       CHECK:     %[[SCALED:.+]] = arith.mulf %[[REAL]], %[[S]]
//       CHECK:     arith.addf %[[ACC]], %[[SCALED]]

// -----

// Per-channel along K, the only reduction dim, so no sub-reduction has constant
// parameters and there is no integer contraction to form. Left alone.
func.func @form_none_is_declined(%aq: tensor<4x8xi8>, %a_s: tensor<8xf32>,
    %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, tensor<8xf32>)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @form_none_is_declined(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

//===----------------------------------------------------------------------===//
// Accumulator bound
//
// The rewrite declines when the products and corrections could overflow the i32
// accumulator. The bound counts the terms that actually survive rather than
// assuming the worst, so the same shape can be admissible symmetric and
// inadmissible asymmetric.
//===----------------------------------------------------------------------===//

// -----

// i8 by i8 with K = 65536. Symmetric, so a single term, and 7+7+16+0+1 = 31 bits
// fit.
func.func @acc_symmetric_fits_deep(%aq: tensor<4x65536xi8>, %a_s: f32, %bq: tensor<65536x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x65536xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x65536xi8>, f32)
      outs(%a_i : tensor<4x65536xf32>) -> tensor<4x65536xf32>
  %b_i = tensor.empty() : tensor<65536x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<65536x16xi8>, f32)
      outs(%b_i : tensor<65536x16xf32>) -> tensor<65536x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x65536xf32>, tensor<65536x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @acc_symmetric_fits_deep(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.generic

// -----

// The same shape and types, but both sides asymmetric. That is four terms rather
// than one, costing two more bits: 7+7+16+2+1 = 33, so it no longer fits and the
// rewrite declines. This pair is what pins the term counting.
func.func @acc_asymmetric_too_deep(%aq: tensor<4x65536xi8>, %a_s: f32, %a_z: i8, %bq: tensor<65536x16xi8>, %b_s: f32, %b_z: i8) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x65536xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x65536xi8>, f32, i8)
      outs(%a_i : tensor<4x65536xf32>) -> tensor<4x65536xf32>
  %b_i = tensor.empty() : tensor<65536x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s, %b_z : tensor<65536x16xi8>, f32, i8)
      outs(%b_i : tensor<65536x16xf32>) -> tensor<65536x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x65536xf32>, tensor<65536x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @acc_asymmetric_too_deep(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// Mixed storage widths are bounded per operand rather than by the wider of the
// two: 15+7 magnitude bits plus a shallow reduction still fits.
func.func @acc_mixed_width_fits(%aq: tensor<4x8xi16>, %a_s: f32, %a_z: i16, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s, %a_z : tensor<4x8xi16>, f32, i16)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @acc_mixed_width_fits(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%{{.+}}, %{{.+}} : tensor<4x8xi16>, tensor<8x16xi8>)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)

// -----

// Both operands i16: 15+15 magnitude bits leaves no room for even a shallow
// reduction.
func.func @acc_wide_storage_declined(%aq: tensor<4x8xi16>, %a_s: f32, %bq: tensor<8x16xi16>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi16>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi16>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @acc_wide_storage_declined(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

//===----------------------------------------------------------------------===//
// Structural preconditions
//
// Cases where the rewrite does not apply at all. Each keeps the floating point
// path, which stays correct.
//===----------------------------------------------------------------------===//

// -----

// Only one operand is quantized, so there is no integer contraction to form: the
// other operand is real valued.
func.func @single_quantized_operand_declined(%aq: tensor<4x8xi8>, %a_s: f32, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @single_quantized_operand_declined(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// The integer contraction starts from an integer zero, so the original
// accumulator has to be a zero fill for the epilogue not to have to carry it.
// A bias initialised accumulator is not handled.
func.func @nonzero_init_declined(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32, %bias: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%bias : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @nonzero_init_declined(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// The dequantize has to be the contraction's immediate producer. Anything in
// between hides it, which is why the pass runs after the passes that move
// reshapes and transposes out from between the two.
func.func @op_between_dequantize_and_contraction_declined(%aq: tensor<4x6xi8>, %a_s: f32,
    %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x6xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x6xi8>, f32)
      outs(%a_i : tensor<4x6xf32>) -> tensor<4x6xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %pad_cst = arith.constant 0.000000e+00 : f32
  %a_pad = tensor.pad %a low[0, 0] high[0, 2] {
  ^bb0(%i: index, %j: index):
    tensor.yield %pad_cst : f32
  } : tensor<4x6xf32> to tensor<4x8xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.matmul ins(%a_pad, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @op_between_dequantize_and_contraction_declined(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.matmul

// -----

// The epilogue multiplies the scales into the result directly, so the
// dequantized element type has to match the contraction's. Here the dequantizes
// produce f16 that the contraction extends to f32, which is not handled.
func.func @element_type_mismatch_declined(%aq: tensor<4x8xi8>, %a_s: f16, %bq: tensor<8x16xi8>, %b_s: f16) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f16)
      outs(%a_i : tensor<4x8xf16>) -> tensor<4x8xf16>
  %b_i = tensor.empty() : tensor<8x16xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f16)
      outs(%b_i : tensor<8x16xf16>) -> tensor<8x16xf16>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d2, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%f : tensor<4x16xf32>) {
  ^bb0(%in: f16, %in_0: f16, %out: f32):
    %le = arith.extf %in : f16 to f32
    %re = arith.extf %in_0 : f16 to f32
    %m = arith.mulf %le, %re : f32
    %s = arith.addf %out, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @element_type_mismatch_declined(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   linalg.generic

// -----

// The maps and iterator types alone do not make an op a contraction. This has
// the shape of one but maximises instead of accumulating, so the expansion the
// rewrite performs would be meaningless.
func.func @non_mul_add_body_declined(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>, %b_s: f32) -> tensor<4x16xf32> {
  %a_i = tensor.empty() : tensor<4x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf32>) -> tensor<4x8xf32>
  %b_i = tensor.empty() : tensor<8x16xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf32>) -> tensor<8x16xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %e = tensor.empty() : tensor<4x16xf32>
  %f = linalg.fill ins(%cst : f32) outs(%e : tensor<4x16xf32>) -> tensor<4x16xf32>
  %c = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                        affine_map<(d0, d1, d2) -> (d2, d1)>,
                                        affine_map<(d0, d1, d2) -> (d0, d1)>],
                       iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<4x8xf32>, tensor<8x16xf32>)
      outs(%f : tensor<4x16xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %m = arith.mulf %in, %in_0 : f32
    %s = arith.maximumf %out, %m : f32
    linalg.yield %s : f32
  } -> tensor<4x16xf32>
  return %c : tensor<4x16xf32>
}
// CHECK-LABEL: func.func @non_mul_add_body_declined(
//       CHECK:   iree_linalg_ext.dequantize_affine
//       CHECK:   arith.maximumf

// -----

// An f16 contraction with f32 scales, as a half precision PT2E model produces.
// The scales are narrowed to the result type before they are multiplied
// together.
func.func @wider_scales(%aq: tensor<4x8xi8>, %a_s: f32, %bq: tensor<8x16xi8>,
    %b_s: f32) -> tensor<4x16xf16> {
  %a_i = tensor.empty() : tensor<4x8xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%aq, %a_s : tensor<4x8xi8>, f32)
      outs(%a_i : tensor<4x8xf16>) -> tensor<4x8xf16>
  %b_i = tensor.empty() : tensor<8x16xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> ()>,
                        affine_map<(d0, d1) -> (d0, d1)>]}
      ins(%bq, %b_s : tensor<8x16xi8>, f32)
      outs(%b_i : tensor<8x16xf16>) -> tensor<8x16xf16>
  %cst = arith.constant 0.000000e+00 : f16
  %e = tensor.empty() : tensor<4x16xf16>
  %f = linalg.fill ins(%cst : f16) outs(%e : tensor<4x16xf16>) -> tensor<4x16xf16>
  %c = linalg.matmul ins(%a, %b : tensor<4x8xf16>, tensor<8x16xf16>)
      outs(%f : tensor<4x16xf16>) -> tensor<4x16xf16>
  return %c : tensor<4x16xf16>
}
// CHECK-LABEL: func.func @wider_scales(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   %[[D:[a-zA-Z0-9_]+]] = linalg.generic
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xi32>)
//       CHECK:   linalg.generic
//  CHECK-SAME:     ins(%[[D]], %{{.+}}, %{{.+}} : tensor<4x16xi32>, f32, f32)
//  CHECK-SAME:     outs(%{{.+}} : tensor<4x16xf16>)
//       CHECK:   ^bb0(%[[ED:[a-zA-Z0-9_]+]]: i32, %[[ESA:[a-zA-Z0-9_]+]]: f32, %[[ESB:[a-zA-Z0-9_]+]]: f32, %{{.+}}: f16):
//       CHECK:     %[[REAL:.+]] = arith.sitofp %[[ED]] : i32 to f16
//       CHECK:     %[[SA:.+]] = arith.truncf %[[ESA]] : f32 to f16
//       CHECK:     %[[SB:.+]] = arith.truncf %[[ESB]] : f32 to f16
//       CHECK:     %[[S:.+]] = arith.mulf %[[SA]], %[[SB]] : f16
//       CHECK:     arith.mulf %[[REAL]], %[[S]] : f16
