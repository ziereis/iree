func.func @matmul(%arg0: tensor<16x16xf16>, %arg1: tensor<16x16xf16>) -> tensor<16x16xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<16x16xf16>
    %0 = tensor.empty() : tensor<16x16xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<16x16xf16>, tensor<16x16xf16>) outs(%1 : tensor<16x16xf32>) -> tensor<16x16xf32>
    return %2 : tensor<16x16xf32>
}
