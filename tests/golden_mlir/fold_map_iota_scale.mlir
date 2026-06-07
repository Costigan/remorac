#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @main() -> f32 {
    %0 = tensor.empty() : tensor<10xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<10xi32>) {
    ^bb0(%out: i32):
      %5 = linalg.index 0 : index
      %6 = arith.index_cast %5 : index to i32
      linalg.yield %6 : i32
    } -> tensor<10xi32>
    %2 = tensor.empty() : tensor<10xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%1 : tensor<10xi32>) outs(%2 : tensor<10xf32>) {
    ^bb0(%in: i32, %out: f32):
      %5 = arith.sitofp %in : i32 to f32
      %cst_0 = arith.constant 2.000000e+00 : f32
      %6 = arith.mulf %5, %cst_0 : f32
      linalg.yield %6 : f32
    } -> tensor<10xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %from_elements = tensor.from_elements %cst : tensor<f32>
    %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%3 : tensor<10xf32>) outs(%from_elements : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %5 = arith.addf %out, %in : f32
      linalg.yield %5 : f32
    } -> tensor<f32>
    %extracted = tensor.extract %4[] : tensor<f32>
    return %extracted : f32
  }
}
