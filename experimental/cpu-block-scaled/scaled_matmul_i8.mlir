// A=unsigned i8, B=signed i8, with one FP32 scale per four-element K block.
// The B matrix is supplied in transposed [N, Ko, K0] form.

module {
  func.func public @scaled_matmul_64x64x256(
      %a: tensor<64x64x4xi8>, %b: tensor<64x64x4xi8>,
      %a_scales: tensor<64x64xf32>, %b_scales: tensor<64x64xf32>)
      -> tensor<64x64xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %c = linalg.fill ins(%zero : f32) outs(%empty : tensor<64x64xf32>)
        -> tensor<64x64xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(m, n, ko, k0) -> (m, ko, k0)>,
          affine_map<(m, n, ko, k0) -> (n, ko, k0)>,
          affine_map<(m, n, ko, k0) -> (m, ko)>,
          affine_map<(m, n, ko, k0) -> (n, ko)>,
          affine_map<(m, n, ko, k0) -> (m, n)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%a, %b, %a_scales, %b_scales : tensor<64x64x4xi8>,
            tensor<64x64x4xi8>, tensor<64x64xf32>, tensor<64x64xf32>)
        outs(%c : tensor<64x64xf32>) {
    ^bb0(%av: i8, %bv: i8, %as: f32, %bs: f32, %acc: f32):
      %scaled_a = arith.scaling_uitofp %av, %as : i8, f32 to f32
      %scaled_b = arith.scaling_sitofp %bv, %bs : i8, f32 to f32
      %product = arith.mulf %scaled_a, %scaled_b : f32
      %sum = arith.addf %acc, %product : f32
      linalg.yield %sum : f32
    } -> tensor<64x64xf32>
    return %result : tensor<64x64xf32>
  }
}
