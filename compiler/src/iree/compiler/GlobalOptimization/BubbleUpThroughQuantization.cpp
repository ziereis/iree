// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Moves the index remapping ops that models put between a quantization op and
// its consumer out of the way, so that the consumer sees the quantization op
// directly.
//
// This exists to establish one invariant for the QDQ-to-integer-math rewrite
// that follows it: a dequantize feeding a contraction is that contraction's
// immediate producer. The rewrite needs that to match, and leaving it to
// whichever general purpose pass happens to run first makes the rewrite's
// success depend on pipeline order rather than on anything local. Exported
// models routinely break the adjacency:
//
//   * `aten.linear` dequantizes the weights and then transposes them.
//   * a padded convolution pads the dequantized image.
//   * a classifier reshapes a pooled activation before the matmul.
//
// Reshapes and transposes feeding a quantization op are handled by the existing
// LinalgExt propagation patterns, which are generic over the op. The two
// directions that need their own pattern are a transpose consuming a
// quantization op, and a pad, which is quantization specific: padding the real
// valued side with zero is the same as padding the quantized side with the zero
// point, and only the quantization op knows what its zero point is.
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Transforms.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_BUBBLEUPTHROUGHQUANTIZATIONPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

using IREE::LinalgExt::DequantizeAffineOp;
using IREE::LinalgExt::QuantizeAffineOp;

namespace {

/// True when both value maps of `op` are the identity, which is the form the
/// quantization ops are built in and the form these patterns produce. A
/// permuted value map would mean the pad amounts and reshape reassociations
/// need permuting alongside, which nothing generates today.
template <typename OpTy>
static bool hasIdentityValueMaps(OpTy op) {
  return op.getInputMap().isIdentity() && op.getOutputMap().isIdentity();
}

/// Bubbles a transpose that consumes a quantization op above it, so that the
/// transpose lands on the quantized side. That is both cheaper, since it moves
/// fewer bits, and more likely to disappear entirely, since quantized weights
/// are usually constants that the transpose folds into.
///
/// For an elementwise op, `transpose(f(x)) == f(transpose(x))` once the
/// quantization parameters follow the permutation: an output element that used
/// to sit at `pi(j)` now sits at `j`, so each parameter map is composed with
/// `pi`, the map of the inverse permutation.
template <typename OpTy>
struct BubbleTransposeThroughQuantization
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    auto quantOp = transposeOp.getInput().getDefiningOp<OpTy>();
    if (!quantOp) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "input is not a quantization op");
    }
    // With other readers the quantization op has to stay, so moving the
    // transpose would duplicate it rather than simplify anything.
    if (!quantOp->getResult(0).hasOneUse()) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "quantization op has other users");
    }
    if (!hasIdentityValueMaps(quantOp)) {
      return rewriter.notifyMatchFailure(transposeOp,
                                         "value maps are not the identity");
    }

    ArrayRef<int64_t> permutation = transposeOp.getPermutation();
    AffineMap inverseMap = AffineMap::getPermutationMap(
        invertPermutationVector(permutation), rewriter.getContext());

    Location loc = transposeOp.getLoc();
    // The transpose moves onto the quantized operand, and its init has the
    // right shape for that operand once its element type is the quantized one.
    Value quantizedInit = tensor::EmptyOp::create(
        rewriter, loc,
        applyPermutation(tensor::getMixedSizes(rewriter, loc, quantOp.getInput()),
                         permutation),
        quantOp.getInputType().getElementType());
    Value transposedInput =
        linalg::TransposeOp::create(rewriter, loc, quantOp.getInput(),
                                    quantizedInit, permutation)
            ->getResult(0);

    SmallVector<AffineMap> maps = quantOp.getIndexingMapsArray();
    // The value maps stay the identity; only the parameters are reindexed.
    for (AffineMap &map : MutableArrayRef<AffineMap>(maps).drop_front().drop_back()) {
      map = map.compose(inverseMap);
    }

    SmallVector<Value> operands = quantOp->getOperands();
    operands.front() = transposedInput;
    operands.back() = transposeOp.getDpsInits()[0];
    auto newOp = cast<OpTy>(mlir::clone(rewriter, quantOp,
                                        transposeOp->getResultTypes(),
                                        operands));
    newOp.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    rewriter.replaceOp(transposeOp, newOp->getResults());
    rewriter.eraseOp(quantOp);
    return success();
  }
};

/// Bubbles a pad of a dequantized value onto the quantized side.
///
/// Padding the real valued side with zero and padding the quantized side with
/// the zero point describe the same tensor, because dequantizing the zero point
/// yields `(zp - zp) * scale`, which is exactly zero. Moving the pad below the
/// dequantize makes the dequantize the immediate producer of whatever consumed
/// the pad, and pads narrower data on the way.
struct BubblePadThroughDequantize : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern<tensor::PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    auto dequantizeOp = padOp.getSource().getDefiningOp<DequantizeAffineOp>();
    if (!dequantizeOp) {
      return rewriter.notifyMatchFailure(padOp, "source is not a dequantize");
    }
    if (!hasIdentityValueMaps(dequantizeOp)) {
      return rewriter.notifyMatchFailure(padOp,
                                         "value maps are not the identity");
    }

    // Only a constant zero pad is the zero point in disguise.
    Value padValue = padOp.getConstantPaddingValue();
    if (!padValue || !matchPattern(padValue, m_AnyZeroFloat())) {
      return rewriter.notifyMatchFailure(padOp, "pad value is not a zero "
                                                "constant");
    }

    // The whole padded region takes one value, so the zero point has to be a
    // single value rather than one per channel.
    Value zeroPoint = dequantizeOp.getZeroPoint();
    if (zeroPoint && isa<ShapedType>(zeroPoint.getType())) {
      return rewriter.notifyMatchFailure(padOp, "zero point is not a scalar");
    }

    // A padded dimension that indexes the quantization parameters would leave
    // the parameters too short for the padded value.
    AffineMap scaleMap = dequantizeOp.getScaleMap();
    for (auto [dim, low, high] :
         llvm::zip_equal(llvm::seq<unsigned>(padOp.getSourceType().getRank()),
                         padOp.getMixedLowPad(), padOp.getMixedHighPad())) {
      if (isConstantIntValue(low, 0) && isConstantIntValue(high, 0)) {
        continue;
      }
      if (scaleMap.isFunctionOfDim(dim)) {
        return rewriter.notifyMatchFailure(
            padOp, "a padded dimension indexes the quantization parameters");
      }
    }

    Location loc = padOp.getLoc();
    Type storageType = dequantizeOp.getInputType().getElementType();
    // Dequantizing the zero point gives zero, so it is what pads the quantized
    // side. A symmetric dequantize has an implicit zero point of zero.
    Value quantizedPadValue;
    if (zeroPoint) {
      quantizedPadValue = convertScalarToDtype(
          rewriter, loc, zeroPoint, storageType,
          /*isUnsignedCast=*/dequantizeOp.getZpUnsigned());
    } else {
      quantizedPadValue = arith::ConstantOp::create(
          rewriter, loc, rewriter.getZeroAttr(storageType));
    }

    auto paddedInput = tensor::PadOp::create(
        rewriter, loc, /*resultType=*/Type(), dequantizeOp.getInput(),
        padOp.getMixedLowPad(), padOp.getMixedHighPad(), quantizedPadValue,
        padOp.getNofold());

    Value init = tensor::EmptyOp::create(
        rewriter, loc, tensor::getMixedSizes(rewriter, loc, paddedInput),
        padOp.getResultType().getElementType());
    SmallVector<Value> operands = dequantizeOp->getOperands();
    operands.front() = paddedInput.getResult();
    operands.back() = init;
    Operation *dequantizedPad = mlir::clone(
        rewriter, dequantizeOp, padOp->getResultTypes(), operands);
    rewriter.replaceOp(padOp, dequantizedPad->getResults());
    return success();
  }
};

/// Decides which reshapes the LinalgExt propagation patterns may fold into a
/// quantization op. They come in a producer- and a consumer-side flavour behind
/// one control function, and the operand they hand it tells the two apart: the
/// producer side passes an operand of the quantization op, the consumer side
/// passes the reshape's source.
///
/// Only the consumer side bubbles. The producer side expands the op and
/// re-collapses its result, which puts a reshape back below the op instead of
/// above it.
///
/// The consumer side additionally needs the result to have a single use.
/// Absorbing a reshape re-ranks the quantization op, so with a second reader
/// still at the old rank it does not remove a reshape, it moves it - and the
/// reader it moves it onto can be the contraction this pass exists to keep
/// adjacent. That is what a folded unit dimension looks like: unit extent
/// folding leaves the op at the contraction's rank with an `expand_shape` for
/// some other reader, and absorbing that expand_shape undoes the folding and
/// strands the contraction behind a fresh `collapse_shape`.
static bool canAbsorbReshape(OpOperand *operand) {
  if (!isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(
          operand->getOwner())) {
    return false;
  }
  return operand->get().hasOneUse();
}

struct BubbleUpThroughQuantizationPass
    : public impl::BubbleUpThroughQuantizationPassBase<
          BubbleUpThroughQuantizationPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    // Reshapes and producer side transposes come from the existing LinalgExt
    // propagation patterns, which are generic over the op type.
    IREE::LinalgExt::populateFoldReshapeOpsByExpansionPatterns(
        patterns, canAbsorbReshape);
    IREE::LinalgExt::populateFuseLinalgExtOpsWithTransposes(
        patterns, [](OpOperand *) { return true; });
    patterns.add<BubbleTransposeThroughQuantization<QuantizeAffineOp>,
                 BubbleTransposeThroughQuantization<DequantizeAffineOp>,
                 BubblePadThroughDequantize>(context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
