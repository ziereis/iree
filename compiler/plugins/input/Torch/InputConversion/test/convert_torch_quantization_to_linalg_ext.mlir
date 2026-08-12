// RUN: iree-opt --split-input-file --mlir-print-local-scope --pass-pipeline="builtin.module(func.func(torch-iree-torch-quantization-to-linalg-ext))" %s | FileCheck %s

// Per-tensor activations, as PT2E emits them: unsigned storage, a scalar scale
// and zero point, and an explicit quantization range. The parameters stay
// scalars, indexed by a map with no results.
func.func @quantize_dequantize_per_tensor(%x: !torch.vtensor<[4,64],f32>)
    -> !torch.vtensor<[4,64],f32> {
  %scale = torch.constant.float 2.500000e-01
  %zp = torch.constant.int 139
  %qmin = torch.constant.int 0
  %qmax = torch.constant.int 255
  %dtype = torch.constant.int 0
  %none = torch.constant.none
  %q = torch.quantized_decomposed.quantize_per_tensor %x, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,64],f32>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,64],ui8>
  %dq = torch.quantized_decomposed.dequantize_per_tensor %q, %scale, %zp, %qmin, %qmax, %dtype, %none
      : !torch.vtensor<[4,64],ui8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none
      -> !torch.vtensor<[4,64],f32>
  return %dq : !torch.vtensor<[4,64],f32>
}
// CHECK-LABEL: func.func @quantize_dequantize_per_tensor(
//  CHECK-SAME:     %[[X:[a-zA-Z0-9]+]]:
//   CHECK-DAG:   %[[ZP:.+]] = arith.constant 139 : i64
//   CHECK-DAG:   %[[SCALE:.+]] = arith.constant 2.500000e-01 : f32
// The torch signedness is dropped on the way to the builtin types, and carried
// by the storage_unsigned and input_unsigned flags instead.
//       CHECK:   %[[IN:.+]] = torch_c.to_builtin_tensor %[[X]]
//  CHECK-SAME:       -> tensor<4x64xf32>
//       CHECK:   %[[Q:.+]] = iree_linalg_ext.quantize_affine
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-SAME:                        affine_map<(d0, d1) -> ()>
//  CHECK-SAME:                        affine_map<(d0, d1) -> ()>
//  CHECK-SAME:                        affine_map<(d0, d1) -> (d0, d1)>]
//  CHECK-SAME:       quant_max = 255 : i64
//  CHECK-SAME:       quant_min = 0 : i64
//  CHECK-SAME:       storage_unsigned
//  CHECK-SAME:       ins(%[[IN]], %[[SCALE]], %[[ZP]] : tensor<4x64xf32>, f32, i64)
//  CHECK-SAME:       -> tensor<4x64xi8>
//       CHECK:   %[[QT:.+]] = torch_c.from_builtin_tensor %[[Q]]
//  CHECK-SAME:       -> !torch.vtensor<[4,64],ui8>
//       CHECK:   %[[DQIN:.+]] = torch_c.to_builtin_tensor %[[QT]]
//       CHECK:   %[[DQ:.+]] = iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       input_unsigned
//  CHECK-SAME:       ins(%[[DQIN]], %[[SCALE]], %[[ZP]] : tensor<4x64xi8>, f32, i64)
//  CHECK-SAME:       -> tensor<4x64xf32>
//       CHECK:   torch_c.from_builtin_tensor %[[DQ]]

// -----

// Per-channel weights over axis 0, with an i64 zero point. The affine
// quantization ops narrow a wide zero point when they lower, so the torch width
// carries through unchanged.
func.func @dequantize_per_channel(%w: !torch.vtensor<[32,64],si8>,
    %s: !torch.vtensor<[32],f32>, %z: !torch.vtensor<[32],si64>)
    -> !torch.vtensor<[32,64],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %dq = torch.quantized_decomposed.dequantize_per_channel %w, %s, %z, %axis, %qmin, %qmax, %dtype, %none
      : !torch.vtensor<[32,64],si8>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],si64>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none
      -> !torch.vtensor<[32,64],f32>
  return %dq : !torch.vtensor<[32,64],f32>
}
// CHECK-LABEL: func.func @dequantize_per_channel(
//   CHECK-DAG:   %[[IN:.+]] = torch_c.to_builtin_tensor %{{.+}} -> tensor<32x64xi8>
//   CHECK-DAG:   %[[S:.+]] = torch_c.to_builtin_tensor %{{.+}} -> tensor<32xf32>
//   CHECK-DAG:   %[[Z:.+]] = torch_c.to_builtin_tensor %{{.+}} -> tensor<32xi64>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-SAME:                        affine_map<(d0, d1) -> (d0)>
//  CHECK-SAME:                        affine_map<(d0, d1) -> (d0)>
//  CHECK-SAME:                        affine_map<(d0, d1) -> (d0, d1)>]
//  CHECK-SAME:       quant_max = 127 : i64
//  CHECK-SAME:       quant_min = -128 : i64
//   CHECK-NOT:       input_unsigned
//  CHECK-SAME:       ins(%[[IN]], %[[S]], %[[Z]] :

// -----

// A negative axis counts from the end, and the parameters land on the dimension
// it names rather than on dimension 0.
func.func @quantize_per_channel_negative_axis(%x: !torch.vtensor<[8,4,16],f32>,
    %s: !torch.vtensor<[16],f32>, %z: !torch.vtensor<[16],si32>)
    -> !torch.vtensor<[8,4,16],si8> {
  %axis = torch.constant.int -1
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %q = torch.quantized_decomposed.quantize_per_channel %x, %s, %z, %axis, %qmin, %qmax, %dtype
      : !torch.vtensor<[8,4,16],f32>, !torch.vtensor<[16],f32>, !torch.vtensor<[16],si32>, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[8,4,16],si8>
  return %q : !torch.vtensor<[8,4,16],si8>
}
// CHECK-LABEL: func.func @quantize_per_channel_negative_axis(
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2) -> (d2)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2) -> (d2)>
//  CHECK-SAME:                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]
//   CHECK-NOT:       storage_unsigned
//  CHECK-SAME:       -> tensor<8x4x16xi8>

// -----

// A None zero point is symmetric quantization, so the op is built without a zero
// point operand and with three indexing maps.
func.func @dequantize_per_channel_symmetric(%w: !torch.vtensor<[32,64],si8>,
    %s: !torch.vtensor<[32],f32>) -> !torch.vtensor<[32,64],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -127
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %dq = torch.quantized_decomposed.dequantize_per_channel %w, %s, %none, %axis, %qmin, %qmax, %dtype, %none
      : !torch.vtensor<[32,64],si8>, !torch.vtensor<[32],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none
      -> !torch.vtensor<[32,64],f32>
  return %dq : !torch.vtensor<[32,64],f32>
}
// CHECK-LABEL: func.func @dequantize_per_channel_symmetric(
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>
//  CHECK-SAME:                        affine_map<(d0, d1) -> (d0)>
//  CHECK-SAME:                        affine_map<(d0, d1) -> (d0, d1)>]
//  CHECK-SAME:       ins(%{{.+}}, %{{.+}} : tensor<32x64xi8>, tensor<32xf32>)

// -----

// Dynamic dimensions come from the input rather than the static shape.
func.func @dequantize_per_channel_dynamic(%w: !torch.vtensor<[?,64],si8>,
    %s: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?,64],f32> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %dq = torch.quantized_decomposed.dequantize_per_channel %w, %s, %none, %axis, %qmin, %qmax, %dtype, %none
      : !torch.vtensor<[?,64],si8>, !torch.vtensor<[?],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none
      -> !torch.vtensor<[?,64],f32>
  return %dq : !torch.vtensor<[?,64],f32>
}
// CHECK-LABEL: func.func @dequantize_per_channel_dynamic(
//       CHECK:   %[[IN:.+]] = torch_c.to_builtin_tensor %{{.+}} -> tensor<?x64xi8>
//       CHECK:   %[[DIM:.+]] = tensor.dim %[[IN]], %{{.+}} : tensor<?x64xi8>
//       CHECK:   tensor.empty(%[[DIM]]) : tensor<?x64xf32>
//       CHECK:   iree_linalg_ext.dequantize_affine

// -----

// A non-constant axis cannot be turned into an indexing map, so the op is left
// for the Torch to Linalg conversion to handle.
func.func @dequantize_per_channel_dynamic_axis(%w: !torch.vtensor<[32,64],si8>,
    %s: !torch.vtensor<[32],f32>, %axis: !torch.int) -> !torch.vtensor<[32,64],f32> {
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %none = torch.constant.none
  %dq = torch.quantized_decomposed.dequantize_per_channel %w, %s, %none, %axis, %qmin, %qmax, %dtype, %none
      : !torch.vtensor<[32,64],si8>, !torch.vtensor<[32],f32>, !torch.none, !torch.int, !torch.int, !torch.int, !torch.int, !torch.none
      -> !torch.vtensor<[32,64],f32>
  return %dq : !torch.vtensor<[32,64],f32>
}
// CHECK-LABEL: func.func @dequantize_per_channel_dynamic_axis(
//   CHECK-NOT:   iree_linalg_ext.dequantize_affine
//       CHECK:   torch.quantized_decomposed.dequantize_per_channel

// -----

// A half precision model keeps its f32 quantization parameters: PT2E computes
// in f32 whatever the value's dtype is, and the affine quantization ops do
// their arithmetic in the scale's element type.
func.func @quantize_dequantize_per_tensor_f16(%x: !torch.vtensor<[4,64],f16>)
    -> !torch.vtensor<[4,64],f16> {
  %scale = torch.constant.float 2.500000e-01
  %zp = torch.constant.int -8
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %dtypef16 = torch.constant.int 5
  %q = torch.quantized_decomposed.quantize_per_tensor %x, %scale, %zp, %qmin, %qmax, %dtype
      : !torch.vtensor<[4,64],f16>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,64],si8>
  %dq = torch.quantized_decomposed.dequantize_per_tensor %q, %scale, %zp, %qmin, %qmax, %dtype, %dtypef16
      : !torch.vtensor<[4,64],si8>, !torch.float, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[4,64],f16>
  return %dq : !torch.vtensor<[4,64],f16>
}
// CHECK-LABEL: func.func @quantize_dequantize_per_tensor_f16(
//   CHECK-DAG:   %[[SCALE:.+]] = arith.constant 2.500000e-01 : f32
//       CHECK:   iree_linalg_ext.quantize_affine
//  CHECK-SAME:       ins(%{{.+}}, %[[SCALE]], %{{.+}} : tensor<4x64xf16>, f32, i64)
//  CHECK-SAME:       -> tensor<4x64xi8>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       ins(%{{.+}}, %[[SCALE]], %{{.+}} : tensor<4x64xi8>, f32, i64)
//  CHECK-SAME:       -> tensor<4x64xf16>

// -----

// Per-channel f32 scales over an f16 value, which is what PT2E emits for the
// weights of a half precision model.
func.func @dequantize_per_channel_f16(%w: !torch.vtensor<[32,64],si8>,
    %s: !torch.vtensor<[32],f32>, %z: !torch.vtensor<[32],si8>)
    -> !torch.vtensor<[32,64],f16> {
  %axis = torch.constant.int 0
  %qmin = torch.constant.int -128
  %qmax = torch.constant.int 127
  %dtype = torch.constant.int 1
  %dtypef16 = torch.constant.int 5
  %dq = torch.quantized_decomposed.dequantize_per_channel %w, %s, %z, %axis, %qmin, %qmax, %dtype, %dtypef16
      : !torch.vtensor<[32,64],si8>, !torch.vtensor<[32],f32>, !torch.vtensor<[32],si8>, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int
      -> !torch.vtensor<[32,64],f16>
  return %dq : !torch.vtensor<[32,64],f16>
}
// CHECK-LABEL: func.func @dequantize_per_channel_f16(
//   CHECK-DAG:   %[[S:.+]] = torch_c.to_builtin_tensor %{{.+}} -> tensor<32xf32>
//       CHECK:   iree_linalg_ext.dequantize_affine
//  CHECK-SAME:       ins(%{{.+}}, %[[S]], %{{.+}} : tensor<32x64xi8>, tensor<32xf32>, tensor<32xi8>)
//  CHECK-SAME:       -> tensor<32x64xf16>
