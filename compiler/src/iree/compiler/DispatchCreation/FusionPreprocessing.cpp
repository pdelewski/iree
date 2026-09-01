// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===--------------- FusionPreprocessing.cpp ------------------------------===//
//
// Miscellaneous patterns run before fusion.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/Utils.h"
#include "iree/compiler/DispatchCreation/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// mlir-obs experiment (docs/decompose-fused-projection-plan.md in the
// mlir-obs repo): decompose a fused projection whose result is only ever
// consumed as disjoint static slices (the HF fused-QKV layout) into
// per-slice contractions over sliced weights, so the slices are born
// contiguous and the slow_memcpy materializations disappear. Correctness
// is shape-generic under the preconditions; PROFITABILITY IS NOT (fused
// projections are a deliberate win on tensor-core targets), hence the
// flag, default off. The grown-up form is a predicate: fire only where
// the consumer analysis says the slices would otherwise materialize.
static llvm::cl::opt<bool> clDecomposeFusedProjections(
    "iree-dispatch-creation-decompose-fused-projections",
    llvm::cl::desc("Decompose matmul+bias chains whose results are only "
                   "consumed as disjoint static slices into per-slice "
                   "matmuls over sliced weights (experimental)"),
    llvm::cl::init(false));

namespace mlir::iree_compiler::DispatchCreation {

#define GEN_PASS_DEF_FUSIONPREPROCESSINGPASS
#include "iree/compiler/DispatchCreation/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// ElementwiseOpInterchangePattern
//===----------------------------------------------------------------------===//

// If possible, interchange indexing maps to make input maps all identity.
struct ElementwiseOpInterchangePattern final
    : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    if (!linalg::isElementwise(genericOp) || genericOp.getNumResults() != 1 ||
        genericOp.getNumDpsInputs() == 0) {
      return failure();
    }

    // All input maps must be equal and non-identity. All maps, including
    // output, must be be permutations. Permutation maps are checked by
    // isElementwise but may be removed.
    AffineMap inputMap = genericOp.getIndexingMapsArray().front();
    auto *initOperand = genericOp.getDpsInitOperand(0);
    if (inputMap.isIdentity() || !inputMap.isPermutation() ||
        !genericOp.getMatchingIndexingMap(initOperand).isPermutation()) {
      return failure();
    }
    for (auto *operand : genericOp.getDpsInputOperands()) {
      if (genericOp.getMatchingIndexingMap(operand) != inputMap) {
        return failure();
      }
    }

    // Make all inputs identity.
    ArrayRef<AffineExpr> exprs = inputMap.getResults();
    auto perm = llvm::map_to_vector(exprs, [](AffineExpr e) -> unsigned {
      return cast<AffineDimExpr>(e).getPosition();
    });
    return linalg::interchangeGenericOp(rewriter, genericOp, perm);
  }
};

//===----------------------------------------------------------------------===//
// FoldSuccessiveTensorInsertSliceOps
//===----------------------------------------------------------------------===//

/// Pattern to fold
///
/// ```
/// %0 = linalg.fill ins(%cst : )
/// %1 = tensor.insert_slice %a into %0
/// %2 = linalg.fill ins(%cst : )
/// %3 = tensor.insert_slice %1 into %2
/// ```
///
/// to
///
/// ```
/// %2 = linalg.fill ins(%cst : )
/// %3 = tensor.insert_slice %a into %2
/// ```
struct FoldSuccessiveTensorInsertSliceOps final
    : OpRewritePattern<tensor::InsertSliceOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(tensor::InsertSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto sourceInsertSlice =
        sliceOp.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!sourceInsertSlice) {
      return failure();
    }
    auto sourceSliceFillOp =
        sourceInsertSlice.getDest().getDefiningOp<linalg::FillOp>();
    auto destSliceFillOp = sliceOp.getDest().getDefiningOp<linalg::FillOp>();
    if (!sourceSliceFillOp || !destSliceFillOp) {
      return rewriter.notifyMatchFailure(
          sliceOp, "dest of both insert_slices expected to be fill operations");
    }
    if (sourceSliceFillOp.getDpsInputOperand(0)->get() !=
        destSliceFillOp.getDpsInputOperand(0)->get()) {
      return rewriter.notifyMatchFailure(
          sliceOp, "dest of both insert_slices expected "
                   "to be fill operation with same value");
    }

    if (!llvm::all_of(sliceOp.getMixedStrides(), isOneInteger) ||
        !llvm::all_of(sliceOp.getMixedStrides(), isOneInteger)) {
      return rewriter.notifyMatchFailure(
          sliceOp, "unhandled non-unit strides of slices");
    }

    SmallVector<OpFoldResult> sourceSliceOffsets =
        sourceInsertSlice.getMixedOffsets();
    SmallVector<OpFoldResult> destSliceOffsets = sliceOp.getMixedOffsets();
    AffineExpr d0, d1;
    bindDims(rewriter.getContext(), d0, d1);
    AffineExpr addExpr = d0 + d1;
    SmallVector<OpFoldResult> offsets = llvm::map_to_vector(
        llvm::zip_equal(sourceSliceOffsets, destSliceOffsets), [&](auto it) {
          return affine::makeComposedFoldedAffineApply(
              rewriter, sliceOp.getLoc(), addExpr,
              {std::get<0>(it), std::get<1>(it)});
        });
    SmallVector<OpFoldResult> sizes = sourceInsertSlice.getMixedSizes();
    SmallVector<OpFoldResult> strides(offsets.size(), rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        sliceOp, sourceInsertSlice.getSource(), sliceOp.getDest(), offsets,
        sizes, strides);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DecomposeFusedProjection
//===----------------------------------------------------------------------===//

/// Matches (the shape the ONNX importer produces for fused QKV):
///
///   %mm  = linalg.generic {(d0,d2),(d2,d1)->(d0,d1)} ins(%x, %W)
///          outs(linalg.fill)                          : MxN
///   %bb  = linalg.generic {(d1)->(d0,d1)} ins(%bias)  : MxN (broadcast)
///   %add = linalg.generic {id,id->id} addf(%mm, %bb)  : MxN
///   %e   = tensor.expand_shape %add : MxN -> 1xMxN
///   slices: every user of %e is a rank-reducing static
///           tensor.extract_slice [0,0,off][1,M,n][1,1,1], disjoint and
///           jointly covering N.
///
/// Rewrites each slice into its own matmul+bias chain over
/// extract_slices of %W (dim 1) and %bias, replacing the slice's result
/// (type Mxn) directly. v1 non-goals (fail closed): dynamic shapes,
/// overlapping or partial slice covers, extra consumers anywhere in the
/// chain, non-2D projections, reassociations that split the sliced dim.
struct DecomposeFusedProjection final
    : OpRewritePattern<tensor::ExpandShapeOp> {
  using Base::Base;

  // The elementwise add generic: two identity-mapped inputs, body is
  // addf(arg0, arg1) + yield.
  static bool isBiasAdd(linalg::GenericOp op) {
    if (!op || !linalg::isElementwise(op) || op.getNumResults() != 1 ||
        op.getNumDpsInputs() != 2)
      return false;
    for (OpOperand &operand : op->getOpOperands())
      if (!op.getMatchingIndexingMap(&operand).isIdentity())
        return false;
    Block &b = op.getRegion().front();
    if (std::distance(b.begin(), b.end()) != 2)
      return false;
    auto add = dyn_cast<arith::AddFOp>(&b.front());
    auto yield = dyn_cast<linalg::YieldOp>(b.getTerminator());
    return add && yield && yield.getOperand(0) == add.getResult() &&
           llvm::is_contained({Value(b.getArgument(0)), Value(b.getArgument(1))},
                              add.getLhs()) &&
           llvm::is_contained({Value(b.getArgument(0)), Value(b.getArgument(1))},
                              add.getRhs());
  }

  // The bias broadcast: one 1-D input mapped (d1), identity output,
  // body is a bare yield of the input.
  static bool isBiasBroadcast(linalg::GenericOp op) {
    if (!op || !linalg::isElementwise(op) || op.getNumResults() != 1 ||
        op.getNumDpsInputs() != 1)
      return false;
    AffineMap inMap = op.getIndexingMapsArray().front();
    if (inMap.getNumResults() != 1 || inMap.getNumDims() != 2 ||
        inMap.getResult(0) != getAffineDimExpr(1, op.getContext()))
      return false;
    Block &b = op.getRegion().front();
    if (std::distance(b.begin(), b.end()) != 1)
      return false;
    auto yield = cast<linalg::YieldOp>(b.getTerminator());
    return yield.getOperand(0) == b.getArgument(0);
  }

  // The projection: a (M,N,K) contraction x:(d0,d2), W:(d2,d1),
  // out:(d0,d1).
  static bool isProjection(linalg::GenericOp op) {
    if (!op || op.getNumResults() != 1 || op.getNumDpsInputs() != 2)
      return false;
    MLIRContext *ctx = op.getContext();
    AffineExpr d0 = getAffineDimExpr(0, ctx), d1 = getAffineDimExpr(1, ctx),
               d2 = getAffineDimExpr(2, ctx);
    auto maps = op.getIndexingMapsArray();
    return maps.size() == 3 &&
           maps[0] == AffineMap::get(3, 0, {d0, d2}, ctx) &&
           maps[1] == AffineMap::get(3, 0, {d2, d1}, ctx) &&
           maps[2] == AffineMap::get(3, 0, {d0, d1}, ctx);
  }

  LogicalResult matchAndRewrite(tensor::ExpandShapeOp expandOp,
                                PatternRewriter &rewriter) const override {
    if (!clDecomposeFusedProjections)
      return failure();
    auto srcType = dyn_cast<RankedTensorType>(expandOp.getSrcType());
    auto resType = dyn_cast<RankedTensorType>(expandOp.getResultType());
    if (!srcType || !resType || !srcType.hasStaticShape() ||
        srcType.getRank() != 2 || resType.getRank() != 3 ||
        resType.getDimSize(0) != 1)
      return rewriter.notifyMatchFailure(expandOp, "not a 2D->1xMxN expand");
    int64_t M = srcType.getDimSize(0), N = srcType.getDimSize(1);

    // Every user is a static rank-reducing slice over N, full elsewhere.
    SmallVector<tensor::ExtractSliceOp> slices;
    for (Operation *user : expandOp->getUsers()) {
      auto slice = dyn_cast<tensor::ExtractSliceOp>(user);
      if (!slice)
        return rewriter.notifyMatchFailure(expandOp, "non-slice user");
      if (!llvm::all_of(slice.getMixedStrides(), isOneInteger))
        return rewriter.notifyMatchFailure(expandOp, "non-unit stride");
      auto offs = slice.getStaticOffsets();
      auto sizes = slice.getStaticSizes();
      if (offs.size() != 3 || offs[0] != 0 || offs[1] != 0 ||
          ShapedType::isDynamicShape(offs) ||
          ShapedType::isDynamicShape(sizes) || sizes[0] != 1 ||
          sizes[1] != M)
        return rewriter.notifyMatchFailure(expandOp, "slice not N-only");
      auto sliceType = dyn_cast<RankedTensorType>(slice.getType());
      if (!sliceType || sliceType.getRank() != 2 ||
          sliceType.getDimSize(0) != M || sliceType.getDimSize(1) != sizes[2])
        return rewriter.notifyMatchFailure(expandOp, "not rank-reduced Mxn");
      slices.push_back(slice);
    }
    if (slices.size() < 2)
      return rewriter.notifyMatchFailure(expandOp, "fewer than 2 slices");
    // Disjoint and covering along N.
    llvm::sort(slices, [](tensor::ExtractSliceOp a, tensor::ExtractSliceOp b) {
      return a.getStaticOffsets()[2] < b.getStaticOffsets()[2];
    });
    int64_t cursor = 0;
    for (auto slice : slices) {
      if (slice.getStaticOffsets()[2] != cursor)
        return rewriter.notifyMatchFailure(expandOp, "slices not covering");
      cursor += slice.getStaticSizes()[2];
    }
    if (cursor != N)
      return rewriter.notifyMatchFailure(expandOp, "slices do not cover N");

    // The producer chain, each link single-use.
    auto addOp = expandOp.getSrc().getDefiningOp<linalg::GenericOp>();
    if (!isBiasAdd(addOp) || !addOp->hasOneUse())
      return rewriter.notifyMatchFailure(expandOp, "no single-use bias add");
    linalg::GenericOp mmOp, bbOp;
    for (Value in : addOp.getDpsInputs()) {
      auto def = in.getDefiningOp<linalg::GenericOp>();
      if (isProjection(def))
        mmOp = def;
      else if (isBiasBroadcast(def))
        bbOp = def;
    }
    if (!mmOp || !bbOp || !mmOp->hasOneUse() || !bbOp->hasOneUse())
      return rewriter.notifyMatchFailure(expandOp,
                                         "chain is not matmul + bias");
    auto fillOp = mmOp.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
    if (!fillOp)
      return rewriter.notifyMatchFailure(expandOp, "matmul init is not fill");
    Value x = mmOp.getDpsInputs()[0];
    Value w = mmOp.getDpsInputs()[1];
    Value bias = bbOp.getDpsInputs()[0];
    auto wType = dyn_cast<RankedTensorType>(w.getType());
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!wType || wType.getRank() != 2 || wType.getDimSize(1) != N ||
        !wType.hasStaticShape() || !biasType || biasType.getRank() != 1 ||
        biasType.getDimSize(0) != N)
      return rewriter.notifyMatchFailure(expandOp, "unexpected W/bias shape");
    int64_t K = wType.getDimSize(0);
    Type elemTy = srcType.getElementType();
    Location loc = expandOp.getLoc();

    // Build one matmul+bias chain per slice; replace the slice directly
    // (its type is Mxn, exactly what the new chain produces).
    for (auto slice : slices) {
      int64_t off = slice.getStaticOffsets()[2];
      int64_t n = slice.getStaticSizes()[2];
      auto one = rewriter.getIndexAttr(1);
      auto zero = rewriter.getIndexAttr(0);
      Value wSlice = tensor::ExtractSliceOp::create(
          rewriter, loc, w,
          ArrayRef<OpFoldResult>{zero, rewriter.getIndexAttr(off)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(K),
                                 rewriter.getIndexAttr(n)},
          ArrayRef<OpFoldResult>{one, one});
      Value bSlice = tensor::ExtractSliceOp::create(
          rewriter, loc, bias,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(off)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(n)},
          ArrayRef<OpFoldResult>{one});
      auto outType = RankedTensorType::get({M, n}, elemTy);
      Value empty = tensor::EmptyOp::create(rewriter, loc,
                                            ArrayRef<int64_t>{M, n}, elemTy);
      Value filled =
          linalg::FillOp::create(rewriter, loc,
                                 fillOp.getDpsInputOperand(0)->get(), empty)
              .getResult(0);
      auto cloneWith = [&](linalg::GenericOp proto, ValueRange ins,
                           Value out) -> Value {
        auto g = linalg::GenericOp::create(
            rewriter, loc, TypeRange{outType}, ins, ValueRange{out},
            proto.getIndexingMapsArray(), proto.getIteratorTypesArray());
        g.getRegion().getBlocks().clear();
        IRMapping map;
        proto.getRegion().cloneInto(&g.getRegion(), map);
        return g.getResult(0);
      };
      Value mm = cloneWith(mmOp, {x, wSlice}, filled);
      Value emptyB = tensor::EmptyOp::create(rewriter, loc,
                                             ArrayRef<int64_t>{M, n}, elemTy);
      Value bb = cloneWith(bbOp, {bSlice}, emptyB);
      Value emptyA = tensor::EmptyOp::create(rewriter, loc,
                                             ArrayRef<int64_t>{M, n}, elemTy);
      SmallVector<Value> addIns = {mm, bb};
      if (addOp.getDpsInputs()[0] == bbOp.getResult(0))
        std::swap(addIns[0], addIns[1]);
      Value sum = cloneWith(addOp, addIns, emptyA);
      rewriter.replaceOp(slice, sum);
    }
    rewriter.eraseOp(expandOp);
    return success();
  }
};

struct FusionPreprocessingPass final
    : impl::FusionPreprocessingPassBase<FusionPreprocessingPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ElementwiseOpInterchangePattern,
                 FoldSuccessiveTensorInsertSliceOps>(&getContext());
    if (clDecomposeFusedProjections) {
      patterns.add<DecomposeFusedProjection>(&getContext());
    }

    // Fold away `tensor.dim` operations that can be resolved in terms of its
    // operand shapes.
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    memref::populateResolveShapedTypeResultDimsPatterns(patterns);
    linalg::populateFoldIntoPackAndUnpackPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::DispatchCreation
