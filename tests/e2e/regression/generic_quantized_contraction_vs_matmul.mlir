// This test verifies the correctness of the generic quantized contraction
// decomposition pass (iree-global-opt-quantized-matmul-to-matmul) for
// linalg.generic ops with various zero-point granularities and extension kinds.
//
// Each test function computes the quantized contraction as a single fused
// linalg.generic (which the pass will decompose) and compares the result
// against a reference that computes the same math using separate extend,
// subtract, and matmul steps (which the pass will NOT match).
//
// Reference: Section 2.3 of https://arxiv.org/abs/1712.05877.

// ==========================================================================
// Reference implementations: extend+subtract then i32 matmul
// These use separate ops so the quantized contraction pass doesn't match them.
// ==========================================================================

// Reference for per-channel asymmetric (both sides, both extsi)
// M=3, K=4, N=5. LHS zp: per-M (3xi32), RHS zp: per-N (5xi32).
func.func private @ref_per_channel_both_asymmetric_extsi(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: tensor<3xi32>, %rhs_zp: tensor<5xi32>) -> tensor<3x5xi32> {
  // Extend LHS and subtract per-channel zero points.
  %init_lhs = tensor.empty() : tensor<3x4xi32>
  %lhs_dequant = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %lhs_zp : tensor<3x4xi8>, tensor<3xi32>)
    outs(%init_lhs : tensor<3x4xi32>) {
    ^bb0(%a: i8, %zp: i32, %out: i32):
      %ext = arith.extsi %a : i8 to i32
      %sub = arith.subi %ext, %zp : i32
      linalg.yield %sub : i32
  } -> tensor<3x4xi32>

  // Extend RHS and subtract per-channel zero points.
  %init_rhs = tensor.empty() : tensor<4x5xi32>
  %rhs_dequant = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%rhs, %rhs_zp : tensor<4x5xi8>, tensor<5xi32>)
    outs(%init_rhs : tensor<4x5xi32>) {
    ^bb0(%b: i8, %zp: i32, %out: i32):
      %ext = arith.extsi %b : i8 to i32
      %sub = arith.subi %ext, %zp : i32
      linalg.yield %sub : i32
  } -> tensor<4x5xi32>

  // i32 matmul
  %c0 = arith.constant 0 : i32
  %init_result = tensor.empty() : tensor<3x5xi32>
  %zero_acc = linalg.fill ins(%c0 : i32) outs(%init_result : tensor<3x5xi32>) -> tensor<3x5xi32>
  %result = linalg.matmul ins(%lhs_dequant, %rhs_dequant : tensor<3x4xi32>, tensor<4x5xi32>)
    outs(%zero_acc : tensor<3x5xi32>) -> tensor<3x5xi32>
  return %result : tensor<3x5xi32>
}

// Reference for mixed extension: LHS extui, RHS extsi, both per-channel zps.
func.func private @ref_mixed_ext_per_channel(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: tensor<3xi32>, %rhs_zp: tensor<5xi32>) -> tensor<3x5xi32> {
  %init_lhs = tensor.empty() : tensor<3x4xi32>
  %lhs_dequant = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %lhs_zp : tensor<3x4xi8>, tensor<3xi32>)
    outs(%init_lhs : tensor<3x4xi32>) {
    ^bb0(%a: i8, %zp: i32, %out: i32):
      %ext = arith.extui %a : i8 to i32
      %sub = arith.subi %ext, %zp : i32
      linalg.yield %sub : i32
  } -> tensor<3x4xi32>

  %init_rhs = tensor.empty() : tensor<4x5xi32>
  %rhs_dequant = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%rhs, %rhs_zp : tensor<4x5xi8>, tensor<5xi32>)
    outs(%init_rhs : tensor<4x5xi32>) {
    ^bb0(%b: i8, %zp: i32, %out: i32):
      %ext = arith.extsi %b : i8 to i32
      %sub = arith.subi %ext, %zp : i32
      linalg.yield %sub : i32
  } -> tensor<4x5xi32>

  %c0 = arith.constant 0 : i32
  %init_result = tensor.empty() : tensor<3x5xi32>
  %zero_acc = linalg.fill ins(%c0 : i32) outs(%init_result : tensor<3x5xi32>) -> tensor<3x5xi32>
  %result = linalg.matmul ins(%lhs_dequant, %rhs_dequant : tensor<3x4xi32>, tensor<4x5xi32>)
    outs(%zero_acc : tensor<3x5xi32>) -> tensor<3x5xi32>
  return %result : tensor<3x5xi32>
}

// Reference for mixed granularity: scalar LHS zp (i32), per-channel RHS zp (5xi32).
func.func private @ref_scalar_lhs_zp_perchannel_rhs_zp(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: i32, %rhs_zp: tensor<5xi32>) -> tensor<3x5xi32> {
  %init_lhs = tensor.empty() : tensor<3x4xi32>
  %lhs_dequant = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> ()>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%lhs, %lhs_zp : tensor<3x4xi8>, i32)
    outs(%init_lhs : tensor<3x4xi32>) {
    ^bb0(%a: i8, %zp: i32, %out: i32):
      %ext = arith.extsi %a : i8 to i32
      %sub = arith.subi %ext, %zp : i32
      linalg.yield %sub : i32
  } -> tensor<3x4xi32>

  %init_rhs = tensor.empty() : tensor<4x5xi32>
  %rhs_dequant = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } ins(%rhs, %rhs_zp : tensor<4x5xi8>, tensor<5xi32>)
    outs(%init_rhs : tensor<4x5xi32>) {
    ^bb0(%b: i8, %zp: i32, %out: i32):
      %ext = arith.extsi %b : i8 to i32
      %sub = arith.subi %ext, %zp : i32
      linalg.yield %sub : i32
  } -> tensor<4x5xi32>

  %c0 = arith.constant 0 : i32
  %init_result = tensor.empty() : tensor<3x5xi32>
  %zero_acc = linalg.fill ins(%c0 : i32) outs(%init_result : tensor<3x5xi32>) -> tensor<3x5xi32>
  %result = linalg.matmul ins(%lhs_dequant, %rhs_dequant : tensor<3x4xi32>, tensor<4x5xi32>)
    outs(%zero_acc : tensor<3x5xi32>) -> tensor<3x5xi32>
  return %result : tensor<3x5xi32>
}

// ==========================================================================
// Quantized contractions (fused generics the pass should decompose)
// ==========================================================================

// Fused per-channel asymmetric contraction (both extsi).
func.func private @quantized_per_channel_both_extsi(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: tensor<3xi32>, %rhs_zp: tensor<5xi32>) -> tensor<3x5xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<3x5xi32>
  %zero = linalg.fill ins(%c0 : i32) outs(%init : tensor<3x5xi32>) -> tensor<3x5xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>)
    outs(%zero : tensor<3x5xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %acc: i32):
    %a_ext = arith.extsi %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %acc : i32
    linalg.yield %add : i32
  } -> tensor<3x5xi32>
  return %result : tensor<3x5xi32>
}

// Fused per-channel asymmetric contraction (mixed ext: LHS extui, RHS extsi).
func.func private @quantized_mixed_ext_per_channel(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: tensor<3xi32>, %rhs_zp: tensor<5xi32>) -> tensor<3x5xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<3x5xi32>
  %zero = linalg.fill ins(%c0 : i32) outs(%init : tensor<3x5xi32>) -> tensor<3x5xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>)
    outs(%zero : tensor<3x5xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %acc: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %acc : i32
    linalg.yield %add : i32
  } -> tensor<3x5xi32>
  return %result : tensor<3x5xi32>
}

// Fused mixed-granularity contraction: scalar LHS zp, per-channel RHS zp.
func.func private @quantized_scalar_lhs_zp_perchannel_rhs_zp(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: i32, %rhs_zp: tensor<5xi32>) -> tensor<3x5xi32> {
  %c0 = arith.constant 0 : i32
  %init = tensor.empty() : tensor<3x5xi32>
  %zero = linalg.fill ins(%c0 : i32) outs(%init : tensor<3x5xi32>) -> tensor<3x5xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> ()>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs, %lhs_zp, %rhs_zp : tensor<3x4xi8>, tensor<4x5xi8>, i32, tensor<5xi32>)
    outs(%zero : tensor<3x5xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %acc: i32):
    %a_ext = arith.extsi %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %acc : i32
    linalg.yield %add : i32
  } -> tensor<3x5xi32>
  return %result : tensor<3x5xi32>
}

// ==========================================================================
// Check functions
// ==========================================================================

func.func private @check_per_channel_both_extsi(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: tensor<3xi32>, %rhs_zp: tensor<5xi32>) {
  %fused = call @quantized_per_channel_both_extsi(%lhs, %rhs, %lhs_zp, %rhs_zp)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> tensor<3x5xi32>
  %ref = call @ref_per_channel_both_asymmetric_extsi(%lhs, %rhs, %lhs_zp, %rhs_zp)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> tensor<3x5xi32>
  check.expect_eq(%fused, %ref) : tensor<3x5xi32>
  return
}

func.func private @check_mixed_ext_per_channel(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: tensor<3xi32>, %rhs_zp: tensor<5xi32>) {
  %fused = call @quantized_mixed_ext_per_channel(%lhs, %rhs, %lhs_zp, %rhs_zp)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> tensor<3x5xi32>
  %ref = call @ref_mixed_ext_per_channel(%lhs, %rhs, %lhs_zp, %rhs_zp)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> tensor<3x5xi32>
  check.expect_eq(%fused, %ref) : tensor<3x5xi32>
  return
}

func.func private @check_scalar_lhs_zp_perchannel_rhs_zp(
    %lhs: tensor<3x4xi8>, %rhs: tensor<4x5xi8>,
    %lhs_zp: i32, %rhs_zp: tensor<5xi32>) {
  %fused = call @quantized_scalar_lhs_zp_perchannel_rhs_zp(%lhs, %rhs, %lhs_zp, %rhs_zp)
    : (tensor<3x4xi8>, tensor<4x5xi8>, i32, tensor<5xi32>) -> tensor<3x5xi32>
  %ref = call @ref_scalar_lhs_zp_perchannel_rhs_zp(%lhs, %rhs, %lhs_zp, %rhs_zp)
    : (tensor<3x4xi8>, tensor<4x5xi8>, i32, tensor<5xi32>) -> tensor<3x5xi32>
  check.expect_eq(%fused, %ref) : tensor<3x5xi32>
  return
}

// ==========================================================================
// Test entry point
// ==========================================================================

func.func @test_generic_quantized_contraction() {
  // Test data
  %lhs = util.unfoldable_constant dense<[
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12]]> : tensor<3x4xi8>
  %rhs = util.unfoldable_constant dense<[
      [5, 4, 3, 2, 9],
      [1, 0, -1, -2, 8],
      [-3, -4, -5, -6, 7],
      [2, 3, 5, 7, 11]]> : tensor<4x5xi8>
  // Matrices with larger values including bounds.
  %lhs2 = util.unfoldable_constant dense<[
      [127, -128, 0, 51],
      [-47, 101, -119, 0],
      [-128, 89, -63, 127]]> : tensor<3x4xi8>
  %rhs2 = util.unfoldable_constant dense<[
      [123, -125, 127, -128, 91],
      [-70, 37, 0, -40, 57],
      [-128, 127, -121, -100, 99],
      [127, 105, 83, 51, -128]]> : tensor<4x5xi8>

  // Per-channel zero points
  %lhs_zp_pc = util.unfoldable_constant dense<[3, -2, 5]> : tensor<3xi32>
  %rhs_zp_pc = util.unfoldable_constant dense<[1, -1, 0, 4, -3]> : tensor<5xi32>
  %lhs_zp_pc2 = util.unfoldable_constant dense<[-128, 0, 127]> : tensor<3xi32>
  %rhs_zp_pc2 = util.unfoldable_constant dense<[41, -57, 100, -100, 0]> : tensor<5xi32>

  // --- Per-channel both asymmetric (both extsi) ---
  call @check_per_channel_both_extsi(%lhs, %rhs, %lhs_zp_pc, %rhs_zp_pc)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> ()
  call @check_per_channel_both_extsi(%lhs2, %rhs2, %lhs_zp_pc2, %rhs_zp_pc2)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> ()

  // --- Mixed extensions (extui LHS, extsi RHS) with per-channel zps ---
  call @check_mixed_ext_per_channel(%lhs, %rhs, %lhs_zp_pc, %rhs_zp_pc)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> ()
  call @check_mixed_ext_per_channel(%lhs2, %rhs2, %lhs_zp_pc2, %rhs_zp_pc2)
    : (tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>) -> ()

  // --- Mixed granularity: scalar LHS zp + per-channel RHS zp ---
  %scalar_zp_3 = arith.constant 3 : i32
  %scalar_zp_m128 = arith.constant -128 : i32
  call @check_scalar_lhs_zp_perchannel_rhs_zp(%lhs, %rhs, %scalar_zp_3, %rhs_zp_pc)
    : (tensor<3x4xi8>, tensor<4x5xi8>, i32, tensor<5xi32>) -> ()
  call @check_scalar_lhs_zp_perchannel_rhs_zp(%lhs2, %rhs2, %scalar_zp_m128, %rhs_zp_pc2)
    : (tensor<3x4xi8>, tensor<4x5xi8>, i32, tensor<5xi32>) -> ()

  return
}
