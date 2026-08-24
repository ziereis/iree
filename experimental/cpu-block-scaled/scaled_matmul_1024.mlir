module {
  // M=N=1024, K=32 blocks * 32 elements.
  func.func public @scaled_matmul_1024x1024x1024(
      %a: tensor<1024x32x32xi8>, %b: tensor<1024x32x32xi8>,
      %a_scales: tensor<1024x32xf32>, %b_scales: tensor<1024x32xf32>)
      -> tensor<1024x1024xf32> {
    %zero = arith.constant 0.0 : f32
    %empty = tensor.empty() : tensor<1024x1024xf32>
    %c = linalg.fill ins(%zero : f32)
        outs(%empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %result = linalg.generic {
        indexing_maps = [
          affine_map<(m, n, ko, k0) -> (m, ko, k0)>,
          affine_map<(m, n, ko, k0) -> (n, ko, k0)>,
          affine_map<(m, n, ko, k0) -> (m, ko)>,
          affine_map<(m, n, ko, k0) -> (n, ko)>,
          affine_map<(m, n, ko, k0) -> (m, n)>],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]}
        ins(%a, %b, %a_scales, %b_scales : tensor<1024x32x32xi8>,
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
