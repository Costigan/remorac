#map = affine_map<(d0) -> (d0)>
module {
  func.func @main() -> tensor<10xi32> {
    %0 = tensor.empty() : tensor<10xi32>
    %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<10xi32>) {
    ^bb0(%out: i32):
      %2 = linalg.index 0 : index
      %3 = arith.index_cast %2 : index to i32
      linalg.yield %3 : i32
    } -> tensor<10xi32>
    return %1 : tensor<10xi32>
  }
}
