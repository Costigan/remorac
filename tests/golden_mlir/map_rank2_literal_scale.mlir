#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main() -> tensor<2x2xf32> {
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 3.000000e+00 : f32
    %cst_2 = arith.constant 4.000000e+00 : f32
    %from_elements = tensor.from_elements %cst, %cst_0, %cst_1, %cst_2 : tensor<2x2xf32>
    %0 = tensor.empty() : tensor<2x2xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%from_elements : tensor<2x2xf32>) outs(%0 : tensor<2x2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_3 = arith.constant 2.000000e+00 : f32
      %2 = arith.mulf %in, %cst_3 : f32
      linalg.yield %2 : f32
    } -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
