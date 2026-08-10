// Unfolded inputs keep the accumulator boundary checks on the execution path.

func.func @static_accumulator_endpoint() {
  %value = util.unfoldable_constant -128 : i8
  %aqi = tensor.empty() : tensor<1x131072xi8>
  %aq = linalg.fill ins(%value : i8) outs(%aqi : tensor<1x131072xi8>) -> tensor<1x131072xi8>
  %bqi = tensor.empty() : tensor<131072x1xi8>
  %bq = linalg.fill ins(%value : i8) outs(%bqi : tensor<131072x1xi8>) -> tensor<131072x1xi8>
  %scale = arith.constant 1.0 : f32
  %ai = tensor.empty() : tensor<1x131072xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %scale : tensor<1x131072xi8>, f32)
      outs(%ai : tensor<1x131072xf32>) -> tensor<1x131072xf32>
  %bi = tensor.empty() : tensor<131072x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %scale : tensor<131072x1xi8>, f32)
      outs(%bi : tensor<131072x1xf32>) -> tensor<131072x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x131072xf32>, tensor<131072x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<2147483648.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}

func.func @dynamic_accumulator_range() {
  %length = util.unfoldable_constant 262144 : index
  %value = util.unfoldable_constant -128 : i8
  %aqi = tensor.empty(%length) : tensor<1x?xi8>
  %aq = linalg.fill ins(%value : i8) outs(%aqi : tensor<1x?xi8>) -> tensor<1x?xi8>
  %bqi = tensor.empty(%length) : tensor<?x1xi8>
  %bq = linalg.fill ins(%value : i8) outs(%bqi : tensor<?x1xi8>) -> tensor<?x1xi8>
  %scale = arith.constant 1.0 : f32
  %ai = tensor.empty(%length) : tensor<1x?xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %scale : tensor<1x?xi8>, f32)
      outs(%ai : tensor<1x?xf32>) -> tensor<1x?xf32>
  %bi = tensor.empty(%length) : tensor<?x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %scale : tensor<?x1xi8>, f32)
      outs(%bi : tensor<?x1xf32>) -> tensor<?x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x?xf32>, tensor<?x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<4294967296.0> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}
