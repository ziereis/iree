module {
  // Fairer MatMulNBits-style boundary: A enters as FP32 and is dynamically
  // quantized per 32-element K block before the integer scaled matmul.
  //
  // This prototype's VPDPBUSD path has no activation zero-point correction,
  // so this uses unsigned min/max quantization for nonnegative activations:
  //   scale = max(A_block) / 255
  //   A_ui8 = round(clamp(A / scale, 0, 255))
  // Supporting general signed activations requires adding zero-point
  // correction (or a signed-A intrinsic path) to the scaled MMA descriptor.
  func.func public @scaled_matmul_1024x1024x1024_with_lhs_quant(
      %a_f32: tensor<1024x32x32xf32>, %b: tensor<1024x32x32xi8>,
      %b_scales: tensor<1024x32xf32>) -> tensor<1024x1024xf32> {
    %zero = arith.constant 0.0 : f32
    %c255 = arith.constant 255.0 : f32
    %epsilon = arith.constant 1.000000e-12 : f32

    // Find the maximum activation in each [m, ko, 0:32] block.
    %max_empty = tensor.empty() : tensor<1024x32xf32>
    %max_init = linalg.fill ins(%zero : f32)
        outs(%max_empty : tensor<1024x32xf32>) -> tensor<1024x32xf32>
    %block_max = linalg.generic {
        indexing_maps = [affine_map<(m, ko, k0) -> (m, ko, k0)>,
                         affine_map<(m, ko, k0) -> (m, ko)>],
        iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a_f32 : tensor<1024x32x32xf32>)
        outs(%max_init : tensor<1024x32xf32>) {
    ^bb0(%a: f32, %current: f32):
      %next = arith.maximumf %a, %current : f32
      linalg.yield %next : f32
    } -> tensor<1024x32xf32>

    // Derive the dequantization scale for every activation block.
    %scale_empty = tensor.empty() : tensor<1024x32xf32>
    %a_scales = linalg.generic {
        indexing_maps = [affine_map<(m, ko) -> (m, ko)>,
                         affine_map<(m, ko) -> (m, ko)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%block_max : tensor<1024x32xf32>)
        outs(%scale_empty : tensor<1024x32xf32>) {
    ^bb0(%max: f32, %unused: f32):
      %safe_max = arith.maximumf %max, %epsilon : f32
      %scale = arith.divf %safe_max, %c255 : f32
      linalg.yield %scale : f32
    } -> tensor<1024x32xf32>

    // Quantize FP32 A to UI8. The i8 storage type is signless in MLIR; the
    // scaling_uitofp op below carries its unsigned interpretation.
    %quant_empty = tensor.empty() : tensor<1024x32x32xi8>
    %a_ui8 = linalg.generic {
        indexing_maps = [affine_map<(m, ko, k0) -> (m, ko, k0)>,
                         affine_map<(m, ko, k0) -> (m, ko)>,
                         affine_map<(m, ko, k0) -> (m, ko, k0)>],
        iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%a_f32, %a_scales : tensor<1024x32x32xf32>,
            tensor<1024x32xf32>)
        outs(%quant_empty : tensor<1024x32x32xi8>) {
    ^bb0(%a: f32, %scale: f32, %unused: i8):
      %normalized = arith.divf %a, %scale : f32
      %clamped_low = arith.maximumf %normalized, %zero : f32
      %clamped = arith.minimumf %clamped_low, %c255 : f32
      %rounded = math.roundeven %clamped : f32
      %quantized = arith.fptoui %rounded : f32 to i8
      linalg.yield %quantized : i8
    } -> tensor<1024x32x32xi8>

    %c_empty = tensor.empty() : tensor<1024x1024xf32>
    %c = linalg.fill ins(%zero : f32)
        outs(%c_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(m, n, ko, k0) -> (m, ko, k0)>,
          affine_map<(m, n, ko, k0) -> (n, ko, k0)>,
          affine_map<(m, n, ko, k0) -> (m, ko)>,
          affine_map<(m, n, ko, k0) -> (n, ko)>,
          affine_map<(m, n, ko, k0) -> (m, n)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%a_ui8, %b, %a_scales, %b_scales : tensor<1024x32x32xi8>,
            tensor<1024x32x32xi8>, tensor<1024x32xf32>,
            tensor<1024x32xf32>)
        outs(%c : tensor<1024x1024xf32>) {
    ^bb0(%av: i8, %bv: i8, %as: f32, %bs: f32, %acc: f32):
      %scaled_a = arith.scaling_uitofp %av, %as : i8, f32 to f32
      %scaled_b = arith.scaling_sitofp %bv, %bs : i8, f32 to f32
      %product = arith.mulf %scaled_a, %scaled_b : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<1024x1024xf32>
    return %result : tensor<1024x1024xf32>
  }
}
