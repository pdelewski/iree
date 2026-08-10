// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Rewrites contractions over affine-dequantized operands using
//
//   sum_k (Aq - zA)*(Bq - zB) = D - zB*RA - zA*RB + N*zA*zB,
//
// where D is the integer contraction and RA/RB sum the corresponding operand.
// Quantization parameters must be invariant over the integer reduction dims.
// Other reduction dims become parallel in D and are reduced after scaling in
// the epilogue. Maps below carry these dependencies in contraction coordinates.
// Integer corrections precede conversion to float to retain cancellation when
// D exceeds the floating-point mantissa.

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_CONVERTQDQTOINTEGERMATHPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;

namespace {

static constexpr unsigned kAccumulatorWidth = 32;

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// One dequantized operand, with maps from the contraction iteration space.
struct QuantizedOperand {
  DequantizeAffineOp dequantize;
  /// Contraction iteration coordinates to quantized storage coordinates.
  AffineMap inputMap;
  /// Contraction iteration coordinates to scale coordinates.
  AffineMap scaleMap;
  /// Contraction iteration coordinates to zero-point coordinates, null if
  /// absent.
  AffineMap zeroPointMap;
};

/// A matched contraction with a nonempty, statically bounded integer reduction.
/// Construction succeeds only for a plain multiply-add body and a zero init.
struct QuantizedContraction {
  QuantizedOperand lhs;
  QuantizedOperand rhs;
  /// Contraction iteration coordinates to output coordinates.
  AffineMap outputMap;
  /// Reduction dims over which all quantization parameters are invariant.
  SmallVector<unsigned> integerReductionDims;
  /// Reduction dims retained as parallel dims until the scaling epilogue.
  SmallVector<unsigned> floatingReductionDims;
  /// Product of integer reduction extents, bounded for the i32 accumulator.
  int64_t reductionExtent = 1;

  // Each operand sum is multiplied by the opposite operand's zero point.
  bool needsLhsSum() { return !rhs.dequantize.isSymmetric(); }
  bool needsRhsSum() { return !lhs.dequantize.isSymmetric(); }
};

/// Bounds every integer partial sum and correction intermediate by
/// N * (|Aq|max + |zA|max) * (|Bq|max + |zB|max) <= INT32_MAX.
/// Zero points have their own types and signedness. Division avoids overflow
/// in the bound itself and preserves the positive endpoint of signed i32.
/// Returns zero when no positive reduction extent can be proven safe.
static int64_t getMaxReductionExtent(QuantizedContraction &contraction) {
  auto magnitude = [](Type type, bool isUnsigned) -> int64_t {
    unsigned width = cast<IntegerType>(type).getWidth();
    if (width >= kAccumulatorWidth) {
      // Zero denotes an unsupported bound, not a zero-valued operand.
      return 0;
    }
    return isUnsigned ? llvm::maxUIntN(width) : int64_t{1} << (width - 1);
  };
  auto operandMagnitude = [&](DequantizeAffineOp dequantize) -> int64_t {
    int64_t input = magnitude(dequantize.getInputType().getElementType(),
                              dequantize.getInputUnsigned());
    if (!input || dequantize.isSymmetric()) {
      return input;
    }
    int64_t zeroPoint =
        magnitude(getElementTypeOrSelf(dequantize.getZeroPoint().getType()),
                  dequantize.getZpUnsigned());
    return zeroPoint ? input + zeroPoint : 0;
  };
  int64_t lhsMagnitude = operandMagnitude(contraction.lhs.dequantize);
  int64_t rhsMagnitude = operandMagnitude(contraction.rhs.dequantize);
  if (!lhsMagnitude || !rhsMagnitude) {
    return 0;
  }
  return llvm::maxIntN(kAccumulatorWidth) / lhsMagnitude / rhsMagnitude;
}

/// Matches exactly two scalar operations: input multiplication and accumulator
/// addition, in either operand order. Casts and other intervening operations
/// prevent matching.
static bool hasPlainFloatMulAddBody(linalg::LinalgOp op) {
  Block *body = op.getBlock();
  if (body->getNumArguments() != 3 || body->getOperations().size() != 3) {
    return false;
  }
  auto yield = dyn_cast<linalg::YieldOp>(body->getTerminator());
  if (!yield || yield.getNumOperands() != 1) {
    return false;
  }
  auto addition = yield.getValues()[0].getDefiningOp<arith::AddFOp>();
  if (!addition) {
    return false;
  }
  Value accumulator = body->getArgument(2);
  Value product;
  if (addition.getLhs() == accumulator) {
    product = addition.getRhs();
  } else if (addition.getRhs() == accumulator) {
    product = addition.getLhs();
  } else {
    return false;
  }
  auto multiplication = product.getDefiningOp<arith::MulFOp>();
  if (!multiplication) {
    return false;
  }
  Value lhs = body->getArgument(0);
  Value rhs = body->getArgument(1);
  return (multiplication.getLhs() == lhs && multiplication.getRhs() == rhs) ||
         (multiplication.getLhs() == rhs && multiplication.getRhs() == lhs);
}

/// The inverse DQ output map moves contraction accesses into DQ iteration
/// coordinates. Composing the input and parameter maps then names their
/// storage coordinates, including any permutation folded into the DQ.
static QuantizedOperand getQuantizedOperand(DequantizeAffineOp dequantize,
                                            AffineMap operandMap) {
  AffineMap toDequantizeIteration =
      inversePermutation(dequantize.getOutputMap()).compose(operandMap);
  return {dequantize, dequantize.getInputMap().compose(toDequantizeIteration),
          dequantize.getScaleMap().compose(toDequantizeIteration),
          dequantize.isSymmetric()
              ? AffineMap()
              : dequantize.getZeroPointMap().compose(toDequantizeIteration)};
}

static FailureOr<QuantizedContraction>
getQuantizedContraction(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1 ||
      !op.hasPureTensorSemantics() || !hasPlainFloatMulAddBody(op)) {
    return failure();
  }
  auto fill = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  if (!fill || !matchPattern(fill.getInputs()[0], m_AnyZeroFloat())) {
    return failure();
  }

  auto lhs = op.getDpsInputs()[0].getDefiningOp<DequantizeAffineOp>();
  auto rhs = op.getDpsInputs()[1].getDefiningOp<DequantizeAffineOp>();
  if (!lhs || !rhs) {
    return failure();
  }
  QuantizedContraction detail;
  detail.lhs = getQuantizedOperand(
      lhs, op.getMatchingIndexingMap(op.getDpsInputOperand(0)));
  detail.rhs = getQuantizedOperand(
      rhs, op.getMatchingIndexingMap(op.getDpsInputOperand(1)));

  detail.outputMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  // The output map has to be a projected permutation for the epilogue to be
  // expressible over the result index space.
  if (!detail.outputMap.isProjectedPermutation()) {
    return failure();
  }

  int64_t maxReductionExtent = getMaxReductionExtent(detail);
  if (maxReductionExtent == 0) {
    return failure();
  }

  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  for (auto [dim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != utils::IteratorType::reduction) {
      continue;
    }
    // Restrict matching to reductions shared by both inputs.
    if (!detail.lhs.inputMap.isFunctionOfDim(dim) ||
        !detail.rhs.inputMap.isFunctionOfDim(dim)) {
      return failure();
    }
    if (detail.lhs.scaleMap.isFunctionOfDim(dim) ||
        detail.rhs.scaleMap.isFunctionOfDim(dim) ||
        (detail.lhs.zeroPointMap &&
         detail.lhs.zeroPointMap.isFunctionOfDim(dim)) ||
        (detail.rhs.zeroPointMap &&
         detail.rhs.zeroPointMap.isFunctionOfDim(dim))) {
      detail.floatingReductionDims.push_back(dim);
      continue;
    }
    detail.integerReductionDims.push_back(dim);
    int64_t extent = loopRanges[dim];
    // Unknown and empty reductions retain the original computation. Checking
    // before multiplying also prevents overflow for very large static shapes.
    if (ShapedType::isDynamic(extent) || extent == 0 ||
        detail.reductionExtent > maxReductionExtent / extent) {
      return failure();
    }
    detail.reductionExtent *= extent;
  }
  if (detail.integerReductionDims.empty()) {
    return failure();
  }
  return detail;
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

static Value createZeroFilledTensor(OpBuilder &b, Location loc,
                                    ArrayRef<OpFoldResult> sizes,
                                    Type elementType) {
  Value empty = tensor::EmptyOp::create(b, loc, sizes, elementType);
  Value zero = arith::ConstantOp::create(b, loc, b.getZeroAttr(elementType));
  return linalg::FillOp::create(b, loc, zero, empty).getResult(0);
}

static SmallVector<OpFoldResult>
getIterationSizes(OpBuilder &b, Location loc, linalg::LinalgOp op,
                  QuantizedContraction &detail) {
  // Query storage shapes so tensor.dim uses do not keep dequantize ops alive
  // after the rewrite. Account for permutations folded into dequantization.
  SmallVector<OpFoldResult> flatShapes;
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    if (auto dequantize = value.getDefiningOp<DequantizeAffineOp>()) {
      SmallVector<OpFoldResult> inputSizes =
          tensor::getMixedSizes(b, loc, dequantize.getInput());
      AffineMap outputShapeMap = dequantize.getOutputMap().compose(
          inversePermutation(dequantize.getInputMap()));
      llvm::append_range(flatShapes, applyPermutationMap<OpFoldResult>(
                                         outputShapeMap, inputSizes));
    } else {
      llvm::append_range(flatShapes, tensor::getMixedSizes(b, loc, value));
    }
  }
  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      op.getShapesToLoopsMap().getResults(), [&](AffineExpr expr) {
        return affine::makeComposedFoldedAffineApply(b, loc, expr, flatShapes);
      });

  // Use init extents for result dimensions to preserve the original result
  // type, even when an input provides a static extent for a dynamic result dim.
  SmallVector<OpFoldResult> initSizes =
      tensor::getMixedSizes(b, loc, op.getDpsInits()[0]);
  for (auto [position, expr] : llvm::enumerate(detail.outputMap.getResults())) {
    sizes[cast<AffineDimExpr>(expr).getPosition()] = initSizes[position];
  }
  return sizes;
}

static Value buildIntegerContraction(OpBuilder &b, Location loc,
                                     linalg::LinalgOp op,
                                     QuantizedContraction &detail,
                                     ArrayRef<OpFoldResult> iterationSizes,
                                     IntegerType accumulatorType) {
  Value lhs = detail.lhs.dequantize.getInput();
  Value rhs = detail.rhs.dequantize.getInput();
  bool lhsUnsigned = detail.lhs.dequantize.getInputUnsigned();
  bool rhsUnsigned = detail.rhs.dequantize.getInputUnsigned();

  MLIRContext *context = b.getContext();

  // Retain a partial result for each block; the epilogue scales and reduces it.
  SmallVector<AffineExpr> resultExprs(detail.outputMap.getResults());
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  for (unsigned dim : detail.floatingReductionDims) {
    resultExprs.push_back(getAffineDimExpr(dim, context));
    iteratorTypes[dim] = utils::IteratorType::parallel;
  }
  AffineMap resultMap =
      AffineMap::get(detail.outputMap.getNumDims(), 0, resultExprs, context);
  SmallVector<OpFoldResult> resultSizes =
      applyPermutationMap<OpFoldResult>(resultMap, iterationSizes);

  Value init = createZeroFilledTensor(b, loc, resultSizes, accumulatorType);

  SmallVector<AffineMap> maps{detail.lhs.inputMap, detail.rhs.inputMap,
                              resultMap};
  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), ValueRange{lhs, rhs}, ValueRange{init}, maps,
      iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value lhsValue =
            convertScalarToDtype(nested, nestedLoc, args[0], accumulatorType,
                                 /*isUnsignedCast=*/lhsUnsigned);
        Value rhsValue =
            convertScalarToDtype(nested, nestedLoc, args[1], accumulatorType,
                                 /*isUnsignedCast=*/rhsUnsigned);
        Value product =
            arith::MulIOp::create(nested, nestedLoc, lhsValue, rhsValue);
        Value sum = arith::AddIOp::create(nested, nestedLoc, args[2], product);
        linalg::YieldOp::create(nested, nestedLoc, sum);
      });
  return genericOp.getResult(0);
}

struct QuantizedOperandSum {
  /// Integer sum of storage values over the integer reduction dimensions.
  Value value;
  /// Contraction iteration dimension for each axis of the sum tensor.
  SmallVector<unsigned> iterationDims;
};

/// Sums storage values over integer reduction dims, retaining only coordinates
/// the operand depends on. For convolution images this is a windowed sum: the
/// image map contains stride/dilation expressions, so reducing storage axes
/// directly would describe a different computation. The returned dimensions
/// describe how to broadcast the sum in the correction epilogue.
static QuantizedOperandSum
sumQuantizedOperand(OpBuilder &b, Location loc, QuantizedOperand &operand,
                    ArrayRef<unsigned> reducedDims,
                    ArrayRef<OpFoldResult> iterationSizes,
                    IntegerType accumulatorType) {
  DequantizeAffineOp dequantize = operand.dequantize;
  AffineMap operandMap = operand.inputMap;
  MLIRContext *context = b.getContext();
  unsigned numDims = operandMap.getNumDims();

  // Omit dimensions absent from the operand map, such as output channels for
  // a convolution image, to avoid computing duplicate sums.
  SmallVector<unsigned> sumDims;
  for (unsigned dim : llvm::seq<unsigned>(numDims)) {
    if (operandMap.isFunctionOfDim(dim)) {
      sumDims.push_back(dim);
    }
  }

  // Unused dimensions have no references in operandMap; their replacements
  // are ignored.
  SmallVector<AffineExpr> toSumSpace(numDims,
                                     getAffineConstantExpr(0, context));
  SmallVector<utils::IteratorType> iteratorTypes;
  SmallVector<AffineExpr> resultExprs;
  SmallVector<unsigned> resultIterationDims;
  for (auto [position, dim] : llvm::enumerate(sumDims)) {
    toSumSpace[dim] = getAffineDimExpr(position, context);
    if (llvm::is_contained(reducedDims, dim)) {
      iteratorTypes.push_back(utils::IteratorType::reduction);
      continue;
    }
    iteratorTypes.push_back(utils::IteratorType::parallel);
    resultExprs.push_back(getAffineDimExpr(position, context));
    resultIterationDims.push_back(dim);
  }
  unsigned sumRank = sumDims.size();
  AffineMap inputMap = operandMap.replaceDimsAndSymbols(
      toSumSpace, /*symReplacements=*/{}, sumRank, /*numResultSyms=*/0);
  AffineMap resultMap = AffineMap::get(sumRank, 0, resultExprs, context);

  SmallVector<Value> inputs{dequantize.getInput()};
  SmallVector<AffineMap> maps{inputMap};

  // Linalg's shapes-to-loops map requires each dimension to appear as a bare
  // map result. Window dimensions appear only in stride/dilation expressions,
  // so supply their extents through an unused operand, as linalg.pooling_*
  // does with its kernel operand.
  llvm::SmallBitVector directlyIndexedDims(sumRank);
  for (AffineMap map : {inputMap, resultMap}) {
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        directlyIndexedDims.set(dimExpr.getPosition());
      }
    }
  }
  SmallVector<AffineExpr> witnessExprs;
  SmallVector<OpFoldResult> witnessSizes;
  for (unsigned position : llvm::seq<unsigned>(sumRank)) {
    if (directlyIndexedDims.test(position)) {
      continue;
    }
    witnessExprs.push_back(getAffineDimExpr(position, context));
    witnessSizes.push_back(iterationSizes[sumDims[position]]);
  }
  if (!witnessExprs.empty()) {
    inputs.push_back(tensor::EmptyOp::create(
        b, loc, witnessSizes, dequantize.getInputType().getElementType()));
    maps.push_back(AffineMap::get(sumRank, 0, witnessExprs, context));
  }
  maps.push_back(resultMap);

  SmallVector<OpFoldResult> resultSizes = llvm::map_to_vector(
      resultIterationDims, [&](unsigned dim) { return iterationSizes[dim]; });
  Value init = createZeroFilledTensor(b, loc, resultSizes, accumulatorType);

  bool isUnsigned = dequantize.getInputUnsigned();
  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), inputs, ValueRange{init}, maps, iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value value =
            convertScalarToDtype(nested, nestedLoc, args[0], accumulatorType,
                                 /*isUnsignedCast=*/isUnsigned);
        // The witness operand, when present, sits between the input and the
        // init, so the accumulator is always the last block argument.
        Value sum =
            arith::AddIOp::create(nested, nestedLoc, args.back(), value);
        linalg::YieldOp::create(nested, nestedLoc, sum);
      });
  return {genericOp.getResult(0), std::move(resultIterationDims)};
}

/// Rewrites a map expressed over the contraction's iteration space into one
/// over the epilogue's iteration space, which is the result index space
/// followed by the reduction dims the quantization parameters vary along.
/// When there are no such dims, the epilogue is elementwise over the result.
static AffineMap toEpilogueSpace(AffineMap map, AffineMap outputMap,
                                 ArrayRef<unsigned> floatingReductionDims,
                                 MLIRContext *context) {
  SmallVector<AffineExpr> replacements(map.getNumDims(),
                                       getAffineConstantExpr(0, context));
  for (auto [resultPosition, expr] : llvm::enumerate(outputMap.getResults())) {
    unsigned iterationDim = cast<AffineDimExpr>(expr).getPosition();
    replacements[iterationDim] = getAffineDimExpr(resultPosition, context);
  }
  unsigned epilogueRank = outputMap.getNumResults();
  for (unsigned dim : floatingReductionDims) {
    replacements[dim] = getAffineDimExpr(epilogueRank++, context);
  }
  return map.replaceDimsAndSymbols(replacements, /*symReplacements=*/{},
                                   epilogueRank, /*numResultSyms=*/0);
}

struct EpilogueOperand {
  /// Tensor or scalar consumed by the epilogue.
  Value value;
  /// Epilogue iteration coordinates to operand coordinates.
  AffineMap map;
};

/// A subtraction term pairing an operand sum with the opposite zero point.
struct ZeroPointCorrection {
  /// Block argument containing the integer operand sum.
  unsigned sumIndex;
  /// Block argument containing the opposite operand's zero point.
  unsigned zeroPointIndex;
  /// Interpretation used when converting the zero point to the accumulator.
  bool zeroPointUnsigned;
};

/// Keeps zero-point cancellation in the bounded integer accumulator before
/// conversion to float, even when the raw sum exceeds the float mantissa.
static Value buildCorrectedIntegerSum(OpBuilder &b, Location loc,
                                      ValueRange args,
                                      ArrayRef<ZeroPointCorrection> corrections,
                                      int64_t reductionExtent) {
  Value accumulated = args[0];
  auto accumulatorType = cast<IntegerType>(accumulated.getType());
  SmallVector<Value, 2> zeroPoints;
  for (const ZeroPointCorrection &correction : corrections) {
    zeroPoints.push_back(convertScalarToDtype(
        b, loc, args[correction.zeroPointIndex], accumulatorType,
        /*isUnsignedCast=*/correction.zeroPointUnsigned));
  }
  for (auto [correction, zeroPoint] :
       llvm::zip_equal(corrections, zeroPoints)) {
    Value term =
        arith::MulIOp::create(b, loc, zeroPoint, args[correction.sumIndex]);
    accumulated = arith::SubIOp::create(b, loc, accumulated, term);
  }
  // The expansion of (Aq - zA)*(Bq - zB) includes +zA*zB per
  // reduction element.
  if (zeroPoints.size() == 2) {
    Value extent = arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(accumulatorType, reductionExtent));
    Value product = arith::MulIOp::create(b, loc, zeroPoints[0], zeroPoints[1]);
    Value term = arith::MulIOp::create(b, loc, product, extent);
    accumulated = arith::AddIOp::create(b, loc, accumulated, term);
  }
  return accumulated;
}

/// A finite f16 result need not have an f16-representable integer accumulator
/// or scale product. Scaling and block accumulation use at least f32,
/// preserving wider scale and output types as well.
static FloatType getEpilogueComputeType(Type resultType, Type lhsScaleType,
                                        Type rhsScaleType) {
  FloatType computeType = Float32Type::get(resultType.getContext());
  for (Type type : {resultType, lhsScaleType, rhsScaleType}) {
    auto floatType = cast<FloatType>(type);
    if (floatType.getWidth() > computeType.getWidth()) {
      computeType = floatType;
    }
  }
  return computeType;
}

/// Applies integer corrections, then scales each partial contraction. A
/// blockwise epilogue reduces partial results before narrowing to the output.
static Value buildCorrectionEpilogue(OpBuilder &b, Location loc,
                                     linalg::LinalgOp op,
                                     QuantizedContraction &detail,
                                     Value contraction,
                                     ArrayRef<OpFoldResult> iterationSizes) {
  MLIRContext *context = b.getContext();
  auto accumulatorType =
      cast<IntegerType>(getElementTypeOrSelf(contraction.getType()));
  auto outputType = cast<RankedTensorType>(op.getDpsInits()[0].getType());
  unsigned resultRank = outputType.getRank();
  ArrayRef<unsigned> floatingDims = detail.floatingReductionDims;
  bool isReduction = !floatingDims.empty();
  unsigned epilogueRank = resultRank + floatingDims.size();
  AffineMap resultMap = AffineMap::get(
      epilogueRank, 0,
      llvm::map_to_vector(llvm::seq<unsigned>(resultRank),
                          [&](unsigned dim) -> AffineExpr {
                            return getAffineDimExpr(dim, context);
                          }),
      context);
  AffineMap contractionMap =
      AffineMap::getMultiDimIdentityMap(epilogueRank, context);

  // Maps an operand indexed by a subset of the contraction's iteration dims
  // into the epilogue's iteration space.
  auto dimsToEpilogueMap = [&](ArrayRef<unsigned> dims) {
    SmallVector<AffineExpr> exprs =
        llvm::map_to_vector(dims, [&](unsigned dim) -> AffineExpr {
          return getAffineDimExpr(dim, context);
        });
    return toEpilogueSpace(
        AffineMap::get(detail.outputMap.getNumDims(), 0, exprs, context),
        detail.outputMap, floatingDims, context);
  };

  SmallVector<EpilogueOperand> operands;
  operands.push_back({contraction, contractionMap});

  SmallVector<ZeroPointCorrection, 2> corrections;
  auto addOperand = [&](Value value, AffineMap map) -> unsigned {
    unsigned index = operands.size();
    operands.push_back({value, map});
    return index;
  };
  auto addCorrection = [&](QuantizedOperand &operand,
                           QuantizedOperand &opposite) {
    QuantizedOperandSum sum =
        sumQuantizedOperand(b, loc, operand, detail.integerReductionDims,
                            iterationSizes, accumulatorType);
    unsigned sumIndex =
        addOperand(sum.value, dimsToEpilogueMap(sum.iterationDims));
    unsigned zeroPointIndex =
        addOperand(opposite.dequantize.getZeroPoint(),
                   toEpilogueSpace(opposite.zeroPointMap, detail.outputMap,
                                   floatingDims, context));
    corrections.push_back(
        {sumIndex, zeroPointIndex, opposite.dequantize.getZpUnsigned()});
  };
  if (detail.needsLhsSum()) {
    addCorrection(detail.lhs, detail.rhs);
  }
  if (detail.needsRhsSum()) {
    addCorrection(detail.rhs, detail.lhs);
  }
  unsigned lhsScaleIndex =
      addOperand(detail.lhs.dequantize.getScale(),
                 toEpilogueSpace(detail.lhs.scaleMap, detail.outputMap,
                                 floatingDims, context));
  unsigned rhsScaleIndex =
      addOperand(detail.rhs.dequantize.getScale(),
                 toEpilogueSpace(detail.rhs.scaleMap, detail.outputMap,
                                 floatingDims, context));

  Type realType = outputType.getElementType();
  FloatType computeType = getEpilogueComputeType(
      realType, detail.lhs.dequantize.getScaleElementType(),
      detail.rhs.dequantize.getScaleElementType());
  Type epilogueResultType = isReduction ? computeType : realType;

  SmallVector<Value> inputs = llvm::map_to_vector(
      operands, [](const EpilogueOperand &operand) { return operand.value; });
  SmallVector<AffineMap> maps = llvm::map_to_vector(
      operands, [](const EpilogueOperand &operand) { return operand.map; });
  maps.push_back(resultMap);

  // A reduction epilogue accumulates into its init, so it has to start at zero;
  // the elementwise form writes every element exactly once.
  SmallVector<OpFoldResult> resultSizes =
      applyPermutationMap<OpFoldResult>(detail.outputMap, iterationSizes);
  Value init =
      isReduction
          ? createZeroFilledTensor(b, loc, resultSizes, epilogueResultType)
          : tensor::EmptyOp::create(b, loc, resultSizes, epilogueResultType)
                .getResult();
  SmallVector<utils::IteratorType> iteratorTypes(resultRank,
                                                 utils::IteratorType::parallel);
  iteratorTypes.append(floatingDims.size(), utils::IteratorType::reduction);

  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), inputs, ValueRange{init}, maps, iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value accumulated = buildCorrectedIntegerSum(
            nested, nestedLoc, args, corrections, detail.reductionExtent);
        Value real = arith::SIToFPOp::create(nested, nestedLoc, computeType,
                                             accumulated);
        Value lhsScale = convertScalarToDtype(nested, nestedLoc,
                                              args[lhsScaleIndex], computeType,
                                              /*isUnsignedCast=*/false);
        Value rhsScale = convertScalarToDtype(nested, nestedLoc,
                                              args[rhsScaleIndex], computeType,
                                              /*isUnsignedCast=*/false);
        Value scale =
            arith::MulFOp::create(nested, nestedLoc, lhsScale, rhsScale);
        Value result = arith::MulFOp::create(nested, nestedLoc, real, scale);
        if (isReduction) {
          result =
              arith::AddFOp::create(nested, nestedLoc, args.back(), result);
        }
        result =
            convertScalarToDtype(nested, nestedLoc, result, epilogueResultType,
                                 /*isUnsignedCast=*/false);
        linalg::YieldOp::create(nested, nestedLoc, result);
      });
  if (epilogueResultType == realType) {
    return genericOp.getResult(0);
  }

  // Blockwise accumulation stays wide; only its final result is narrowed.
  AffineMap identity = AffineMap::getMultiDimIdentityMap(resultRank, context);
  Value output = tensor::EmptyOp::create(b, loc, resultSizes, realType);
  return linalg::GenericOp::create(
             b, loc, outputType, ValueRange{genericOp.getResult(0)},
             ValueRange{output}, SmallVector<AffineMap>{identity, identity},
             SmallVector<utils::IteratorType>(resultRank,
                                              utils::IteratorType::parallel),
             [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
               Value result =
                   convertScalarToDtype(nested, nestedLoc, args[0], realType,
                                        /*isUnsignedCast=*/false);
               linalg::YieldOp::create(nested, nestedLoc, result);
             })
      .getResult(0);
}

struct ConvertQDQToIntegerMath : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<QuantizedContraction> maybeDetail = getQuantizedContraction(op);
    if (failed(maybeDetail)) {
      return failure();
    }
    QuantizedContraction &detail = *maybeDetail;

    Location loc = op.getLoc();
    auto accumulatorType =
        IntegerType::get(rewriter.getContext(), kAccumulatorWidth);

    SmallVector<OpFoldResult> iterationSizes =
        getIterationSizes(rewriter, loc, op, detail);

    Value contraction = buildIntegerContraction(
        rewriter, loc, op, detail, iterationSizes, accumulatorType);

    Value result = buildCorrectionEpilogue(rewriter, loc, op, detail,
                                           contraction, iterationSizes);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertQDQToIntegerMathPass final
    : impl::ConvertQDQToIntegerMathPassBase<ConvertQDQToIntegerMathPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, linalg::LinalgDialect,
                tensor::TensorDialect, IREE::LinalgExt::IREELinalgExtDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.insert<ConvertQDQToIntegerMath>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
