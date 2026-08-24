module {
  // Directly comparable i8 x i8 -> i32 mmt4d microkernel path.
  func.func public @unscaled_matmul_1024x1024x1024(
      %a: tensor<1024x1024xi8>, %b: tensor<1024x1024xi8>)
      -> tensor<1024x1024xi32> {
    %zero = arith.constant 0 : i32
    %empty = tensor.empty() : tensor<1024x1024xi32>
    %c = linalg.fill ins(%zero : i32)
        outs(%empty : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
    %result = linalg.matmul
        ins(%a, %b : tensor<1024x1024xi8>, tensor<1024x1024xi8>)
        outs(%c : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
    return %result : tensor<1024x1024xi32>
  }
}
