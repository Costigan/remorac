#map = affine_map<(d0) -> (d0)>
module {
  func.func @main() -> tensor<10xf32> {
    %0 = tensor.empty() : tensor<10xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<10xi32>) {
    ^bb0(%out: i32):
      %4 = linalg.index 0 : index
      %5 = arith.index_cast %4 : index to i32
      linalg.yield %5 : i32
    } -> tensor<10xi32>
    %2 = tensor.empty() : tensor<10xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<10xi32>) outs(%2 : tensor<10xf32>) {
    ^bb0(%in: i32, %out: f32):
      %cst = arith.constant 2.000000e+00 : f32
      %4 = arith.sitofp %in : i32 to f32
      %5 = arith.mulf %cst, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<10xf32>
    return %3 : tensor<10xf32>
  }
}
