// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-tensor-ops-to-linalg))" --split-input-file %s | FileCheck %s

// first all tests where i assume no reduced dims
func.func @basic(%arg0: tensor<2x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<2x1x4xf32> {
  %gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<2x1xi64>) -> tensor<2x1x4xf32>
  return %gather : tensor<2x1x4xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @basic
// CHECK-NOT: tensor.gather
// CHECK: %[[C0:.*]] = arith.constant 0
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX2:.*]] = linalg.index 2
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: tensor.extract %arg1[%[[CASTED_IDX0]], %[[LINALG_IDX2]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @dyn_shape(%arg0: tensor<?x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<?x1x4xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<?x1xi64>) -> tensor<?x1x4xf32>
return %gather : tensor<?x1x4xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @dyn_shape
// CHECK-NOT: tensor.gather
// CHECK: %[[C0:.*]] = arith.constant 0
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX2:.*]] = linalg.index 2
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: tensor.extract %arg1[%[[CASTED_IDX0]], %[[LINALG_IDX2]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @many_dims(%arg0: tensor<5x10x2x2x1xi64>, %arg1: tensor<16x4xf32>) -> tensor<5x10x2x2x1x4xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0]) : (tensor<16x4xf32>, tensor<5x10x2x2x1xi64>) -> tensor<5x10x2x2x1x4xf32>
return %gather : tensor<5x10x2x2x1x4xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK: func.func @many_dims
// CHECK-NOT: tensor.gather
// CHECK: %[[C0:.*]] = arith.constant 0
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX1:.*]] = linalg.index 1
// CHECK: %[[LINALG_IDX2:.*]] = linalg.index 2
// CHECK: %[[LINALG_IDX3:.*]] = linalg.index 3
// CHECK: %[[LINALG_IDX5:.*]] = linalg.index 5
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[LINALG_IDX1]], %[[LINALG_IDX2]], %[[LINALG_IDX3]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: tensor.extract %arg1[%[[CASTED_IDX0]], %[[LINALG_IDX5]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x1x1x1(%arg0: tensor<1x3xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x1x1x1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1, 2]) : (tensor<3x3x3xf32>, tensor<1x3xi64>) -> tensor<1x1x1x1xf32>
return %gather : tensor<1x1x1x1xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @gather_1x1x1x1
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C2:.*]] = arith.constant 2
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: %[[INDICES_IDX1:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C1]]]
// CHECK: %[[CASTED_IDX1:.*]] = arith.index_cast %[[INDICES_IDX1]]
// CHECK: %[[INDICES_IDX2:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C2]]]
// CHECK: %[[CASTED_IDX2:.*]] = arith.index_cast %[[INDICES_IDX2]]
// CHECK: tensor.extract %arg1[%[[CASTED_IDX0]], %[[CASTED_IDX1]], %[[CASTED_IDX2]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x1x1x3(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x1x1x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x1x1x3xf32>
return %gather : tensor<1x1x1x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: func.func @gather_1x1x1x3
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX3:.*]] = linalg.index 3
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: %[[INDICES_IDX1:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C1]]]
// CHECK: %[[CASTED_IDX1:.*]] = arith.index_cast %[[INDICES_IDX1]]
// CHECK: tensor.extract %arg1[%[[CASTED_IDX0]], %[[CASTED_IDX1]], %[[LINALG_IDX3]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

// tests with reduced dims
func.func @gather_1(%arg0: tensor<1x3xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1, 2]) : (tensor<3x3x3xf32>, tensor<1x3xi64>) -> tensor<1xf32>
return %gather : tensor<1xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: func.func @gather_1
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK-DAG: %[[C2:.*]] = arith.constant 2
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: %[[INDICES_IDX1:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C1]]]
// CHECK: %[[CASTED_IDX1:.*]] = arith.index_cast %[[INDICES_IDX1]]
// CHECK: %[[INDICES_IDX2:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C2]]]
// CHECK: %[[CASTED_IDX2:.*]] = arith.index_cast %[[INDICES_IDX2]]
// CHECK: tensor.extract %arg1[%[[CASTED_IDX0]], %[[CASTED_IDX1]], %[[CASTED_IDX2]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x3(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([0, 1]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x3xf32>
return %gather : tensor<1x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @gather_1x3
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX1:.*]] = linalg.index 1
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: %[[INDICES_IDX1:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C1]]]
// CHECK: %[[CASTED_IDX1:.*]] = arith.index_cast %[[INDICES_IDX1]]
// CHECK: tensor.extract %arg1[%[[CASTED_IDX0]], %[[CASTED_IDX1]], %[[LINALG_IDX1]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x3x3_dim_1(%arg0: tensor<1x1xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([1]) : (tensor<3x3x3xf32>, tensor<1x1xi64>) -> tensor<1x3x3xf32>
return %gather : tensor<1x3x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @gather_1x3x3_dim_1
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX1:.*]] = linalg.index 1
// CHECK: %[[LINALG_IDX2:.*]] = linalg.index 2
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: tensor.extract %arg1[%[[LINALG_IDX1]], %[[CASTED_IDX0]], %[[LINALG_IDX2]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x3x3_dim_2(%arg0: tensor<1x1xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([2]) : (tensor<3x3x3xf32>, tensor<1x1xi64>) -> tensor<1x3x3xf32>
return %gather : tensor<1x3x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: func.func @gather_1x3x3_dim_2
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX1:.*]] = linalg.index 1
// CHECK: %[[LINALG_IDX2:.*]] = linalg.index 2
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: tensor.extract %arg1[%[[LINALG_IDX1]], %[[LINALG_IDX2]], %[[CASTED_IDX0]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @gather_1x3_dim_1_2(%arg0: tensor<1x2xi64>, %arg1: tensor<3x3x3xf32>) -> tensor<1x3xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([1, 2]) : (tensor<3x3x3xf32>, tensor<1x2xi64>) -> tensor<1x3xf32>
return %gather : tensor<1x3xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @gather_1x3_dim_1_2
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX1:.*]] = linalg.index 1
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: %[[INDICES_IDX1:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C1]]]
// CHECK: %[[CASTED_IDX1:.*]] = arith.index_cast %[[INDICES_IDX1]]
// CHECK: tensor.extract %arg1[%[[LINALG_IDX1]], %[[CASTED_IDX0]], %[[CASTED_IDX1]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----

func.func @source_1(%arg0: tensor<1x2xi64>, %arg1: tensor<1x1x1xf32>) -> tensor<1x1xf32> {
%gather = tensor.gather %arg1[%arg0] gather_dims([1, 2]) : (tensor<1x1x1xf32>, tensor<1x2xi64>) -> tensor<1x1xf32>
return %gather : tensor<1x1xf32>
}

// CHECK: #[[MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: func.func @source_1
// CHECK-NOT: tensor.gather
// CHECK-DAG: %[[C0:.*]] = arith.constant 0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1
// CHECK: tensor.empty
// CHECK: %[[RES:.+]] = linalg.generic
// CHECK-SAME: indexing_maps = [#[[MAP0]]]
// CHECK: %[[LINALG_IDX0:.*]] = linalg.index 0
// CHECK: %[[LINALG_IDX1:.*]] = linalg.index 1
// CHECK: %[[INDICES_IDX0:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C0]]]
// CHECK: %[[CASTED_IDX0:.*]] = arith.index_cast %[[INDICES_IDX0]]
// CHECK: %[[INDICES_IDX1:.*]] = tensor.extract %arg0[%[[LINALG_IDX0]], %[[C1]]]
// CHECK: %[[CASTED_IDX1:.*]] = arith.index_cast %[[INDICES_IDX1]]
// CHECK: tensor.extract %arg1[%[[LINALG_IDX1]], %[[CASTED_IDX0]], %[[CASTED_IDX1]]]
// CHECK: linalg.yield
// CHECK: return %[[RES]]

// -----