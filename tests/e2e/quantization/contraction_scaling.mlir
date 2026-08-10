// Runs with integer reassociation enabled and with the default QDQ path.
// Inputs stay unfolded so these checks execute the compiled contractions.

func.func @half_accumulator_range() {
  %aq = util.unfoldable_constant dense<127> : tensor<1x8xi8>
  %bq = util.unfoldable_constant dense<127> : tensor<8x1xi8>
  %scale = util.unfoldable_constant 0.0078125 : f32
  %ai = tensor.empty() : tensor<1x8xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %scale : tensor<1x8xi8>, f32)
      outs(%ai : tensor<1x8xf16>) -> tensor<1x8xf16>
  %bi = tensor.empty() : tensor<8x1xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %scale : tensor<8x1xi8>, f32)
      outs(%bi : tensor<8x1xf16>) -> tensor<8x1xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<1x1xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x1xf16>) -> tensor<1x1xf16>
  %result = linalg.matmul ins(%a, %b : tensor<1x8xf16>, tensor<8x1xf16>)
      outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
  check.expect_eq_const(%result, dense<7.875> : tensor<1x1xf16>) : tensor<1x1xf16>
  return
}

func.func @half_scale_product_range() {
  %aq = util.unfoldable_constant dense<127> : tensor<1x1xi8>
  %bq = util.unfoldable_constant dense<127> : tensor<1x1xi8>
  %scale = util.unfoldable_constant 0.0001 : f32
  %ai = tensor.empty() : tensor<1x1xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %scale : tensor<1x1xi8>, f32)
      outs(%ai : tensor<1x1xf16>) -> tensor<1x1xf16>
  %bi = tensor.empty() : tensor<1x1xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %scale : tensor<1x1xi8>, f32)
      outs(%bi : tensor<1x1xf16>) -> tensor<1x1xf16>
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<1x1xf16>
  %init = linalg.fill ins(%zero : f16) outs(%empty : tensor<1x1xf16>) -> tensor<1x1xf16>
  %result = linalg.matmul ins(%a, %b : tensor<1x1xf16>, tensor<1x1xf16>)
      outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
  check.expect_almost_eq_const(%result, dense<0.0001613> : tensor<1x1xf16>, atol 1.0e-6, rtol 0.0) : tensor<1x1xf16>
  return
}

// Each block has an integer sum beyond the f16 range. The final reduction
// consumes scaled partials in f32 before narrowing its result.
func.func @half_blockwise_accumulator_range() {
  %aq = util.unfoldable_constant dense<127> : tensor<1x2x8xi8>
  %bq = util.unfoldable_constant dense<127> : tensor<2x8x1xi8>
  %sa = util.unfoldable_constant dense<0.0078125> : tensor<1x2xf16>
  %za = util.unfoldable_constant dense<0> : tensor<1x2xi8>
  %sb = util.unfoldable_constant 0.0078125 : f32
  %ainit = tensor.empty() : tensor<1x2x8xf16>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1)>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%aq, %sa, %za : tensor<1x2x8xi8>, tensor<1x2xf16>, tensor<1x2xi8>)
      outs(%ainit : tensor<1x2x8xf16>) -> tensor<1x2x8xf16>
  %binit = tensor.empty() : tensor<2x8x1xf16>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                        affine_map<(d0, d1, d2) -> ()>,
                        affine_map<(d0, d1, d2) -> (d0, d1, d2)>]}
      ins(%bq, %sb : tensor<2x8x1xi8>, f32)
      outs(%binit : tensor<2x8x1xf16>) -> tensor<2x8x1xf16>
  %cst = arith.constant 0.000000e+00 : f16
  %init = tensor.empty() : tensor<1x1xf16>
  %fill = linalg.fill ins(%cst : f16) outs(%init : tensor<1x1xf16>) -> tensor<1x1xf16>
  %c = linalg.contract
      indexing_maps = [affine_map<(m, n, g, l) -> (m, g, l)>,
                       affine_map<(m, n, g, l) -> (g, l, n)>,
                       affine_map<(m, n, g, l) -> (m, n)>]
      ins(%a, %b : tensor<1x2x8xf16>, tensor<2x8x1xf16>)
      outs(%fill : tensor<1x1xf16>) -> tensor<1x1xf16>
  check.expect_almost_eq_const(%c, dense<15.75> : tensor<1x1xf16>, atol 0.02, rtol 0.0) : tensor<1x1xf16>
  return
}

func.func @nonzero_zero_point_corrections() {
  %aq = util.unfoldable_constant dense<5> : tensor<1x8xi8>
  %bq = util.unfoldable_constant dense<7> : tensor<8x1xi8>
  %scale = util.unfoldable_constant 0.125 : f32
  %za = util.unfoldable_constant 3 : i8
  %zb = util.unfoldable_constant 6 : i8
  %ai = tensor.empty() : tensor<1x8xf32>
  %a = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%aq, %scale, %za : tensor<1x8xi8>, f32, i8)
      outs(%ai : tensor<1x8xf32>) -> tensor<1x8xf32>
  %bi = tensor.empty() : tensor<8x1xf32>
  %b = iree_linalg_ext.dequantize_affine
      {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->()>, affine_map<(d0,d1)->(d0,d1)>]}
      ins(%bq, %scale, %zb : tensor<8x1xi8>, f32, i8)
      outs(%bi : tensor<8x1xf32>) -> tensor<8x1xf32>
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<1x1xf32>
  %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<1x1xf32>) -> tensor<1x1xf32>
  %result = linalg.matmul ins(%a, %b : tensor<1x8xf32>, tensor<8x1xf32>)
      outs(%init : tensor<1x1xf32>) -> tensor<1x1xf32>
  check.expect_eq_const(%result, dense<0.25> : tensor<1x1xf32>) : tensor<1x1xf32>
  return
}
