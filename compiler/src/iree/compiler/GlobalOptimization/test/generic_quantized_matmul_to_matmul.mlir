// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-global-opt-quantized-matmul-to-matmul{shift-to-signed-domain=false}))" --split-input-file %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-global-opt-quantized-matmul-to-matmul{shift-to-signed-domain=true}))" --split-input-file %s | FileCheck %s --check-prefix=SHIFT

// -----
// LHS per-channel asymmetric | RHS per-channel asymmetric (both extui)
// No shifting needed since both sides use the same extension.
// -----

func.func @per_channel_both_asymmetric(
    %A: tensor<?x?xi8>, %B: tensor<?x?xi8>,
    %zp_a: tensor<?xi32>, %zp_b: tensor<?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %M = tensor.dim %A, %c0 : tensor<?x?xi8>
  %N = tensor.dim %B, %c1 : tensor<?x?xi8>
  %empty = tensor.empty(%M, %N) : tensor<?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a, %zp_b : tensor<?x?xi8>, tensor<?x?xi8>, tensor<?xi32>, tensor<?xi32>)
    outs(%zero : tensor<?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extui %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?xi32>
  return %result : tensor<?x?xi32>
}

// CHECK-LABEL: func.func @per_channel_both_asymmetric
// CHECK-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[ZP_A:.+]]: tensor<?xi32>, %[[ZP_B:.+]]: tensor<?xi32>
//
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.extui
// CHECK:         arith.muli
// CHECK:         arith.addi
//
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]] : tensor<?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.addi
//
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[B]] : tensor<?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.addi
//
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[ZP_B]], %[[RHS_SUMS]], %[[ZP_A]]
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:         arith.muli
// CHECK:         arith.muli
// CHECK:         arith.addi
// CHECK:       return %[[RESULT]]

// Both extensions are the same, so SHIFT mode produces identical output.
// SHIFT-LABEL: func.func @per_channel_both_asymmetric
// SHIFT:       linalg.generic
// SHIFT:         arith.extui
// SHIFT:         arith.extui
// SHIFT:       return

// -----
// LHS per-channel asymmetric (extui) | RHS symmetric (extsi)
// With shift-to-signed-domain=false: preserves mixed extui/extsi.
// With shift-to-signed-domain=true: shifts LHS to signed domain.
// -----

func.func @per_channel_lhs_asymmetric_rhs_symmetric(
    %A: tensor<?x?xi8>, %B: tensor<?x?xi8>,
    %zp_a: tensor<?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %M = tensor.dim %A, %c0 : tensor<?x?xi8>
  %N = tensor.dim %B, %c1 : tensor<?x?xi8>
  %empty = tensor.empty(%M, %N) : tensor<?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a : tensor<?x?xi8>, tensor<?x?xi8>, tensor<?xi32>)
    outs(%zero : tensor<?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %mul = arith.muli %a_adj, %b_ext : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?xi32>
  return %result : tensor<?x?xi32>
}

// Without shift: mixed extensions preserved.
// CHECK-LABEL: func.func @per_channel_lhs_asymmetric_rhs_symmetric
// CHECK-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[ZP_A:.+]]: tensor<?xi32>
//
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.extsi
// CHECK:         arith.muli
// CHECK:         arith.addi
//
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[B]] : tensor<?x?xi8>)
// CHECK:         arith.extsi
// CHECK:         arith.addi
//
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[MATMUL]], %[[RHS_SUMS]], %[[ZP_A]]
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:       return %[[RESULT]]

// With shift: LHS is shifted to signed domain.
// A' = A +_i8 -128, zp_a' = zp_a - 128
// Matmul uses extsi on both sides.
// SHIFT-LABEL: func.func @per_channel_lhs_asymmetric_rhs_symmetric
// SHIFT-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// SHIFT-SAME:    %[[ZP_A:.+]]: tensor<?xi32>
//
// Step 1: shift A into signed domain: A' = A +_i8 -128
// SHIFT:       %[[A_SHIFTED:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[A]] : tensor<?x?xi8>)
// SHIFT:         arith.addi
//
// Step 2: adjust zero point: zp_a' = zp_a - 128
// SHIFT:       %[[ZP_A_ADJ:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[ZP_A]] : tensor<?xi32>)
// SHIFT:         arith.subi
//
// Term 1: matmul with extsi on both sides
// SHIFT:       %[[MATMUL:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[A_SHIFTED]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi8>)
// SHIFT:         arith.extsi
// SHIFT:         arith.extsi
// SHIFT:         arith.muli
// SHIFT:         arith.addi
//
// Term 2: rhsSums (extsi since B was already signed)
// SHIFT:       %[[RHS_SUMS:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[B]] : tensor<?x?xi8>)
// SHIFT:         arith.extsi
//
// Term 3: correction using adjusted zero point
// SHIFT:       %[[RESULT:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[MATMUL]], %[[RHS_SUMS]], %[[ZP_A_ADJ]]
// SHIFT:       return %[[RESULT]]

// -----
// LHS per-block asymmetric | RHS per-block asymmetric (batch form, both extui)
// -----

func.func @per_block_both_asymmetric(
    %A: tensor<?x?x?xi8>, %B: tensor<?x?x?xi8>,
    %zp_a: tensor<?x?xi32>, %zp_b: tensor<?x?xi32>) -> tensor<?x?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  %KB = tensor.dim %A, %c0 : tensor<?x?x?xi8>
  %M = tensor.dim %A, %c1 : tensor<?x?x?xi8>
  %N = tensor.dim %B, %c2 : tensor<?x?x?xi8>
  %empty = tensor.empty(%KB, %M, %N) : tensor<?x?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(kb, m, n, b) -> (kb, m, b)>,
      affine_map<(kb, m, n, b) -> (kb, b, n)>,
      affine_map<(kb, m, n, b) -> (kb, m)>,
      affine_map<(kb, m, n, b) -> (kb, n)>,
      affine_map<(kb, m, n, b) -> (kb, m, n)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a, %zp_b
      : tensor<?x?x?xi8>, tensor<?x?x?xi8>, tensor<?x?xi32>, tensor<?x?xi32>)
    outs(%zero : tensor<?x?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extui %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?x?xi32>
  return %result : tensor<?x?x?xi32>
}

// CHECK-LABEL: func.func @per_block_both_asymmetric
// CHECK-SAME:    %[[A:.+]]: tensor<?x?x?xi8>, %[[B:.+]]: tensor<?x?x?xi8>
// CHECK-SAME:    %[[ZP_A:.+]]: tensor<?x?xi32>, %[[ZP_B:.+]]: tensor<?x?xi32>
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?x?xi8>, tensor<?x?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.extui
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]] : tensor<?x?x?xi8>)
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[B]] : tensor<?x?x?xi8>)
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[ZP_B]], %[[RHS_SUMS]], %[[ZP_A]]
// CHECK:       return %[[RESULT]]

// Same extensions, SHIFT doesn't change anything.
// SHIFT-LABEL: func.func @per_block_both_asymmetric
// SHIFT:       linalg.generic
// SHIFT:         arith.extui
// SHIFT:         arith.extui
// SHIFT:       return

// -----
// LHS per-block asymmetric (extui) | RHS symmetric (extsi) — mixed extensions
// -----

func.func @per_block_lhs_asymmetric_rhs_symmetric(
    %A: tensor<?x?x?xi8>, %B: tensor<?x?x?xi8>,
    %zp_a: tensor<?x?xi32>) -> tensor<?x?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  %KB = tensor.dim %A, %c0 : tensor<?x?x?xi8>
  %M = tensor.dim %A, %c1 : tensor<?x?x?xi8>
  %N = tensor.dim %B, %c2 : tensor<?x?x?xi8>
  %empty = tensor.empty(%KB, %M, %N) : tensor<?x?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(kb, m, n, b) -> (kb, m, b)>,
      affine_map<(kb, m, n, b) -> (kb, b, n)>,
      affine_map<(kb, m, n, b) -> (kb, m)>,
      affine_map<(kb, m, n, b) -> (kb, m, n)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a : tensor<?x?x?xi8>, tensor<?x?x?xi8>, tensor<?x?xi32>)
    outs(%zero : tensor<?x?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %mul = arith.muli %a_adj, %b_ext : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?x?xi32>
  return %result : tensor<?x?x?xi32>
}

// Without shift: mixed extensions preserved.
// CHECK-LABEL: func.func @per_block_lhs_asymmetric_rhs_symmetric
// CHECK-SAME:    %[[A:.+]]: tensor<?x?x?xi8>, %[[B:.+]]: tensor<?x?x?xi8>
// CHECK-SAME:    %[[ZP_A:.+]]: tensor<?x?xi32>
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?x?xi8>, tensor<?x?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.extsi
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[B]] : tensor<?x?x?xi8>)
// CHECK:         arith.extsi
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[MATMUL]], %[[RHS_SUMS]], %[[ZP_A]]
// CHECK:       return %[[RESULT]]

// With shift: LHS shifted to signed domain, matmul uses extsi on both sides.
// SHIFT-LABEL: func.func @per_block_lhs_asymmetric_rhs_symmetric
// SHIFT-SAME:    %[[A:.+]]: tensor<?x?x?xi8>, %[[B:.+]]: tensor<?x?x?xi8>
// SHIFT-SAME:    %[[ZP_A:.+]]: tensor<?x?xi32>
//
// Shift A: A' = A +_i8 -128
// SHIFT:       %[[A_SHIFTED:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[A]] : tensor<?x?x?xi8>)
// SHIFT:         arith.addi
//
// Adjust zp_a: zp_a' = zp_a - 128
// SHIFT:       %[[ZP_ADJ:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[ZP_A]] : tensor<?x?xi32>)
// SHIFT:         arith.subi
//
// Matmul with extsi on both sides
// SHIFT:       %[[MATMUL:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[A_SHIFTED]], %[[B]] : tensor<?x?x?xi8>, tensor<?x?x?xi8>)
// SHIFT:         arith.extsi
// SHIFT:         arith.extsi
//
// SHIFT:       return

// -----
// Static shapes: LHS per-channel asymmetric | RHS per-channel asymmetric
// (both extui — no shifting needed)
// -----

func.func @per_channel_both_asymmetric_static(
    %A: tensor<3x4xi8>, %B: tensor<4x5xi8>,
    %zp_a: tensor<3xi32>, %zp_b: tensor<5xi32>) -> tensor<3x5xi32> {
  %c0_i32 = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<3x5xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<3x5xi32>) -> tensor<3x5xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a, %zp_b : tensor<3x4xi8>, tensor<4x5xi8>, tensor<3xi32>, tensor<5xi32>)
    outs(%zero : tensor<3x5xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extui %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<3x5xi32>
  return %result : tensor<3x5xi32>
}

// CHECK-LABEL: func.func @per_channel_both_asymmetric_static
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<3x4xi8>, tensor<4x5xi8>)
// CHECK-SAME:    outs({{.*}} : tensor<3x5xi32>)
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<3x4xi8>)
// CHECK-SAME:    outs({{.*}} : tensor<3xi32>)
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins({{.*}} : tensor<4x5xi8>)
// CHECK-SAME:    outs({{.*}} : tensor<5xi32>)
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK:       return %[[RESULT]]

// SHIFT-LABEL: func.func @per_channel_both_asymmetric_static
// SHIFT:       linalg.generic
// SHIFT:         arith.extui
// SHIFT:         arith.extui
// SHIFT:       return

// -----
// RHS-only asymmetric (LHS symmetric extsi, RHS asymmetric extui) — mixed
// With shift: RHS gets shifted, LHS stays. Since LHS was symmetric and RHS
// had a zero point, after shifting RHS also gets zp adjusted.
// -----

func.func @per_channel_rhs_only_asymmetric(
    %A: tensor<?x?xi8>, %B: tensor<?x?xi8>,
    %zp_b: tensor<?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %M = tensor.dim %A, %c0 : tensor<?x?xi8>
  %N = tensor.dim %B, %c1 : tensor<?x?xi8>
  %empty = tensor.empty(%M, %N) : tensor<?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_b : tensor<?x?xi8>, tensor<?x?xi8>, tensor<?xi32>)
    outs(%zero : tensor<?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpb: i32, %out: i32):
    %a_ext = arith.extsi %a : i8 to i32
    %b_ext = arith.extui %b : i8 to i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_ext, %b_adj : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?xi32>
  return %result : tensor<?x?xi32>
}

// Without shift: mixed extensions preserved.
// CHECK-LABEL: func.func @per_channel_rhs_only_asymmetric
// CHECK-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[ZP_B:.+]]: tensor<?xi32>
//
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi8>)
// CHECK:         arith.extsi
// CHECK:         arith.extui
//
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]] : tensor<?x?xi8>)
// CHECK:         arith.extsi
//
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[ZP_B]]
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:       return %[[RESULT]]

// With shift: RHS (the extui side) gets shifted to signed domain.
// B' = B +_i8 -128, zp_b' = zp_b - 128
// Matmul uses extsi on both sides.
// SHIFT-LABEL: func.func @per_channel_rhs_only_asymmetric
// SHIFT-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// SHIFT-SAME:    %[[ZP_B:.+]]: tensor<?xi32>
//
// Shift B: B' = B +_i8 -128
// SHIFT:       %[[B_SHIFTED:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[B]] : tensor<?x?xi8>)
// SHIFT:         arith.addi
//
// Adjust zp_b: zp_b' = zp_b - 128
// SHIFT:       %[[ZP_B_ADJ:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[ZP_B]] : tensor<?xi32>)
// SHIFT:         arith.subi
//
// Matmul with extsi on both sides
// SHIFT:       %[[MATMUL:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[A]], %[[B_SHIFTED]] : tensor<?x?xi8>, tensor<?x?xi8>)
// SHIFT:         arith.extsi
// SHIFT:         arith.extsi
//
// SHIFT:       return

// -----
// Mixed granularity: LHS scalar zero point | RHS per-channel zero point
// -----

func.func @scalar_lhs_zp_perchannel_rhs_zp(
    %A: tensor<?x?xi8>, %B: tensor<?x?xi8>,
    %zp_a: i32, %zp_b: tensor<?xi32>) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %M = tensor.dim %A, %c0 : tensor<?x?xi8>
  %N = tensor.dim %B, %c1 : tensor<?x?xi8>
  %empty = tensor.empty(%M, %N) : tensor<?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> ()>,
      affine_map<(m, n, k) -> (n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a, %zp_b : tensor<?x?xi8>, tensor<?x?xi8>, i32, tensor<?xi32>)
    outs(%zero : tensor<?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extui %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?xi32>
  return %result : tensor<?x?xi32>
}

// CHECK-LABEL: func.func @scalar_lhs_zp_perchannel_rhs_zp
// CHECK-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[ZP_A:.+]]: i32, %[[ZP_B:.+]]: tensor<?xi32>
//
// Term 1: matmul
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.extui
//
// Term 2: lhsSums[m]
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]] : tensor<?x?xi8>)
// CHECK-SAME:    outs({{.*}} : tensor<?xi32>)
//
// Term 3: rhsSums[n]
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[B]] : tensor<?x?xi8>)
// CHECK-SAME:    outs({{.*}} : tensor<?xi32>)
//
// Correction: zp_b[n] is per-channel, zp_a is scalar
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel"]
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[ZP_B]], %[[RHS_SUMS]], %[[ZP_A]]
// CHECK-SAME:      tensor<?x?xi32>, tensor<?xi32>, tensor<?xi32>, tensor<?xi32>, i32
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:         arith.muli
// CHECK:         arith.muli
// CHECK:         arith.addi
// CHECK:       return %[[RESULT]]

// SHIFT-LABEL: func.func @scalar_lhs_zp_perchannel_rhs_zp
// SHIFT:       linalg.generic
// SHIFT:         arith.extui
// SHIFT:         arith.extui
// SHIFT:       return

// -----
// Mixed granularity: LHS per-block zero point | RHS scalar zero point
// (blockwise quantized LHS, per-tensor quantized RHS)
// -----

func.func @perblock_lhs_zp_scalar_rhs_zp(
    %A: tensor<?x?x?xi8>, %B: tensor<?x?x?xi8>,
    %zp_a: tensor<?x?xi32>, %zp_b: i32) -> tensor<?x?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c0_i32 = arith.constant 0 : i32
  %KB = tensor.dim %A, %c0 : tensor<?x?x?xi8>
  %M = tensor.dim %A, %c1 : tensor<?x?x?xi8>
  %N = tensor.dim %B, %c2 : tensor<?x?x?xi8>
  %empty = tensor.empty(%KB, %M, %N) : tensor<?x?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(kb, m, n, b) -> (kb, m, b)>,
      affine_map<(kb, m, n, b) -> (kb, b, n)>,
      affine_map<(kb, m, n, b) -> (kb, m)>,
      affine_map<(kb, m, n, b) -> ()>,
      affine_map<(kb, m, n, b) -> (kb, m, n)>
    ],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a, %zp_b
      : tensor<?x?x?xi8>, tensor<?x?x?xi8>, tensor<?x?xi32>, i32)
    outs(%zero : tensor<?x?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extui %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?x?xi32>
  return %result : tensor<?x?x?xi32>
}

// CHECK-LABEL: func.func @perblock_lhs_zp_scalar_rhs_zp
// CHECK-SAME:    %[[A:.+]]: tensor<?x?x?xi8>, %[[B:.+]]: tensor<?x?x?xi8>
// CHECK-SAME:    %[[ZP_A:.+]]: tensor<?x?xi32>, %[[ZP_B:.+]]: i32
//
// Term 1: matmul
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?x?xi8>, tensor<?x?x?xi8>)
//
// Term 2: lhsSums[kb,m]
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]] : tensor<?x?x?xi8>)
// CHECK-SAME:    outs({{.*}} : tensor<?x?xi32>)
//
// Term 3: rhsSums[kb,n]
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[B]] : tensor<?x?x?xi8>)
// CHECK-SAME:    outs({{.*}} : tensor<?x?xi32>)
//
// Correction: zp_a[kb,m] is per-block, zp_b is scalar
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    iterator_types = ["parallel", "parallel", "parallel"]
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[ZP_B]], %[[RHS_SUMS]], %[[ZP_A]]
// CHECK-SAME:      tensor<?x?x?xi32>, tensor<?x?xi32>, i32, tensor<?x?xi32>, tensor<?x?xi32>
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:         arith.muli
// CHECK:         arith.subi
// CHECK:         arith.muli
// CHECK:         arith.muli
// CHECK:         arith.addi
// CHECK:       return %[[RESULT]]

// SHIFT-LABEL: func.func @perblock_lhs_zp_scalar_rhs_zp
// SHIFT:       linalg.generic
// SHIFT:         arith.extui
// SHIFT:         arith.extui
// SHIFT:       return

// -----
// Mixed granularity + mixed extensions:
// LHS per-channel asymmetric (extui) | RHS scalar asymmetric (extsi)
// -----

func.func @perchannel_lhs_extui_scalar_rhs_extsi(
    %A: tensor<?x?xi8>, %B: tensor<?x?xi8>,
    %zp_a: tensor<?xi32>, %zp_b: i32) -> tensor<?x?xi32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %M = tensor.dim %A, %c0 : tensor<?x?xi8>
  %N = tensor.dim %B, %c1 : tensor<?x?xi8>
  %empty = tensor.empty(%M, %N) : tensor<?x?xi32>
  %zero = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x?xi32>) -> tensor<?x?xi32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m)>,
      affine_map<(m, n, k) -> ()>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B, %zp_a, %zp_b : tensor<?x?xi8>, tensor<?x?xi8>, tensor<?xi32>, i32)
    outs(%zero : tensor<?x?xi32>) {
  ^bb0(%a: i8, %b: i8, %zpa: i32, %zpb: i32, %out: i32):
    %a_ext = arith.extui %a : i8 to i32
    %b_ext = arith.extsi %b : i8 to i32
    %a_adj = arith.subi %a_ext, %zpa : i32
    %b_adj = arith.subi %b_ext, %zpb : i32
    %mul = arith.muli %a_adj, %b_adj : i32
    %add = arith.addi %mul, %out : i32
    linalg.yield %add : i32
  } -> tensor<?x?xi32>
  return %result : tensor<?x?xi32>
}

// Without shift: mixed extensions and mixed granularity preserved.
// CHECK-LABEL: func.func @perchannel_lhs_extui_scalar_rhs_extsi
// CHECK-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// CHECK-SAME:    %[[ZP_A:.+]]: tensor<?xi32>, %[[ZP_B:.+]]: i32
//
// CHECK:       %[[MATMUL:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi8>)
// CHECK:         arith.extui
// CHECK:         arith.extsi
//
// lhsSums with extui (matching A's extension)
// CHECK:       %[[LHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[A]] : tensor<?x?xi8>)
// CHECK:         arith.extui
//
// rhsSums with extsi (matching B's extension)
// CHECK:       %[[RHS_SUMS:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[B]] : tensor<?x?xi8>)
// CHECK:         arith.extsi
//
// Correction: zp_a[m] is per-channel tensor, zp_b is scalar
// CHECK:       %[[RESULT:.+]] = linalg.generic
// CHECK-SAME:    ins(%[[MATMUL]], %[[LHS_SUMS]], %[[ZP_B]], %[[RHS_SUMS]], %[[ZP_A]]
// CHECK-SAME:      tensor<?x?xi32>, tensor<?xi32>, i32, tensor<?xi32>, tensor<?xi32>
// CHECK:       return %[[RESULT]]

// With shift: LHS (extui) gets shifted to signed domain.
// zp_a is a tensor so it gets adjusted elementwise, zp_b is scalar (unchanged).
// SHIFT-LABEL: func.func @perchannel_lhs_extui_scalar_rhs_extsi
// SHIFT-SAME:    %[[A:.+]]: tensor<?x?xi8>, %[[B:.+]]: tensor<?x?xi8>
// SHIFT-SAME:    %[[ZP_A:.+]]: tensor<?xi32>, %[[ZP_B:.+]]: i32
//
// Shift A
// SHIFT:       %[[A_SHIFTED:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[A]] : tensor<?x?xi8>)
// SHIFT:         arith.addi
//
// Adjust tensor zp_a elementwise
// SHIFT:       %[[ZP_A_ADJ:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[ZP_A]] : tensor<?xi32>)
// SHIFT:         arith.subi
//
// Matmul: both extsi now
// SHIFT:       %[[MATMUL:.+]] = linalg.generic
// SHIFT-SAME:    ins(%[[A_SHIFTED]], %[[B]] : tensor<?x?xi8>, tensor<?x?xi8>)
// SHIFT:         arith.extsi
// SHIFT:         arith.extsi
//
// SHIFT:       return
