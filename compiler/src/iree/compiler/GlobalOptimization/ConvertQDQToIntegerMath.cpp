// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Rewrites a contraction whose operands are produced by
// iree_linalg_ext.dequantize_affine into an integer contraction plus a
// correction epilogue.
//
// A quantized value dequantizes as
//
//   A[i] = sA * (Aq[i] - zA)
//
// so a matmul over two dequantized operands is
//
//   C[i,j] = sum_k sA*(Aq[i,k] - zA) * sB*(Bq[k,j] - zB)
//
// If the scales and zero points do not vary along k they come out of the sum:
//
//   C[i,j] = sA*sB * sum_k (Aq[i,k] - zA)*(Bq[k,j] - zB)
//
// and expanding the product moves the zero points out of the reduction too:
//
//   sum_k (Aq - zA)*(Bq - zB)
//     = sum_k Aq*Bq - zB*sum_k Aq - zA*sum_k Bq + N*zA*zB
//     = D           - zB*RA       - zA*RB       + N*zA*zB
//
// D is a plain integer contraction, RA and RB are reductions of the quantized
// operands over the contracted dims, and N is the number of values each output
// element reduces over. The point of the expansion is that D needs no widening
// and no subtraction inside the hot loop, while RA and RB are cheap by
// comparison: each drops the parallel dims its operand does not depend on, and
// RB reduces over weights, which are usually constant and fold away entirely.
// The exception is a convolution's image sum, which is a pooling over the
// window and drops only F, so a depthwise convolution's is the size of the
// convolution itself. It is still worth building; see `sumQuantizedOperand`.
//
// Which reduction dims the quantization parameters vary along is decided by
// carrying the contraction's operand map through the dequantize's own maps,
// giving a map from the iteration space to the quantization parameter index
// space. Splitting the reduction dims by whether that map depends on them
// leaves three cases:
//
//  * none of them: one integer contraction over all the reduction dims, then an
//    elementwise epilogue. Covers per-tensor, per-channel and multi-axis.
//  * some but not all: the ones the parameters vary along stay parallel in the
//    integer contraction, which then yields one partial result per block, and
//    the epilogue reduces across those in floating point. This is what
//    blockwise (group) quantization needs, where K is expanded to (G, L) and
//    the scales are indexed by G.
//  * all of them: nothing comes out of the sum, there is no integer contraction
//    to form, and the op is left alone.
//
// The corrections are computed in the integer accumulator and converted to
// floating point once at the end, because D exceeds the f32 mantissa at
// realistic contraction depths.
//===----------------------------------------------------------------------===//

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

/// Width of the integer accumulator. Whether it is wide enough depends on the
/// storage types and the contraction depth, which `accumulatorFits` checks.
static constexpr unsigned kAccumulatorWidth = 32;

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// A contraction-shaped `linalg` op whose inputs are both produced by
/// `iree_linalg_ext.dequantize_affine`, together with the partition of its
/// reduction dimensions by whether the quantization parameters vary along them.
struct QuantizedContractionDetail {
  enum class Form {
    /// No quantization parameter varies along a reduction dim. One integer
    /// contraction over all of them, then an elementwise epilogue.
    Full,
    /// Some but not all reduction dims carry varying quantization parameters.
    /// The integer contraction keeps those dims parallel and produces one
    /// partial result per block, which the epilogue reduces in floating point.
    Partial,
    /// Every reduction dim carries varying quantization parameters, so there
    /// is no integer contraction to form at all. Not rewritten.
    None,
  };

  Form form;
  DequantizeAffineOp lhsDequant, rhsDequant;
  /// Iteration space to the index space of the quantized lhs and rhs, i.e. the
  /// dequantize inputs the integer contraction reads, and to the result.
  AffineMap lhsMap, rhsMap, outMap;
  /// Iteration space to each operand's quantization parameter index space.
  AffineMap lhsScaleMap, rhsScaleMap;
  std::optional<AffineMap> lhsZeroPointMap, rhsZeroPointMap;
  /// The reduction dims the quantization parameters are invariant over, and
  /// the ones they are not.
  SmallVector<unsigned> intReductionDims, floatReductionDims;
  /// The number of values each output element of the integer contraction
  /// reduces over, i.e. the product of the `intReductionDims` extents, or
  /// nullopt when any of them is dynamic. Taken from the iteration domain so
  /// that a convolution counts its window rather than its input extent.
  std::optional<int64_t> staticReductionExtent;

  /// The `- zB*RA` term needs a reduction of the lhs, and exists only if the
  /// *rhs* carries a zero point. Symmetrically for the other side.
  bool needsLhsSum() { return !rhsDequant.isSymmetric(); }
  bool needsRhsSum() { return !lhsDequant.isSymmetric(); }
};

/// Returns true if the integer contraction and its zero point corrections
/// provably cannot overflow an `accumulatorWidth`-bit signed accumulator.
///
/// A signed b-bit operand has magnitude at most 2^(b-1) and an unsigned one at
/// most 2^b - 1; call that its magnitude bits. Each product is then bounded by
/// 2^(lhsBits + rhsBits), the sum runs over `reductionExtent` of them, every
/// correction term is bounded by that same amount, and the accumulator needs a
/// sign bit. Counting the terms rather than assuming the worst matters: a
/// symmetric contraction is one term and gets the full depth, while an
/// asymmetric one is four and loses two bits of it.
///
/// A dynamic extent has nothing to bound against, so it is assumed to fit
/// rather than declining every dynamically shaped contraction.
static bool accumulatorFits(unsigned accumulatorWidth,
                            QuantizedContractionDetail &detail) {
  if (!detail.staticReductionExtent) {
    return true;
  }
  auto storageBits = [](DequantizeAffineOp dequant) {
    unsigned width =
        cast<IntegerType>(dequant.getInputType().getElementType()).getWidth();
    return dequant.getInputUnsigned() ? width : width - 1;
  };
  // D, plus one term per operand sum, plus the N*zA*zB term when both sides
  // carry a zero point.
  unsigned numTerms = 1 + detail.needsLhsSum() + detail.needsRhsSum() +
                      (detail.needsLhsSum() && detail.needsRhsSum());
  unsigned requiredBits =
      storageBits(detail.lhsDequant) + storageBits(detail.rhsDequant) +
      llvm::Log2_64_Ceil(std::max<int64_t>(*detail.staticReductionExtent, 1)) +
      llvm::Log2_64_Ceil(numTerms) + 1;
  return requiredBits <= accumulatorWidth;
}

/// Returns true if `op`'s body multiplies its two inputs and accumulates the
/// product, which is what the rewrite's algebra assumes. The maps and iterator
/// types alone do not imply it: the same shape of op with a `maximumf` in place
/// of the `addf` is a completely different computation.
///
/// `linalg::isaContractionOpInterface` cannot be used for this. Besides the
/// body it requires all indexing maps to be projected permutations, which every
/// convolution violates (an image map contains `oh * stride + kh * dilation`),
/// so it would reject the convolutions this pass is meant to handle. Only its
/// body check is applicable, and that is available on its own.
static bool hasFloatMulAddBody(linalg::LinalgOp op) {
  return linalg::detail::isContractionBody(
      *op.getBlock(), [](Operation *elementwise, Operation *reduction) {
        return isa<arith::MulFOp>(elementwise) && isa<arith::AddFOp>(reduction);
      });
}

/// Moves `operandMap`, which indexes a dequantized contraction operand, onto
/// the index spaces of the dequantize's own operands: the quantized input and
/// the quantization parameters.
///
/// A dequantize states all of its maps over its own iteration space, and the
/// output map is what relates that space to the dequantized value the
/// contraction reads. Inverting it carries the contraction's operand map into
/// the dequantize's iteration space, and from there each of the dequantize's
/// maps names the index space of the operand it belongs to. This is where a
/// permutation folded into the dequantize becomes a permutation of the integer
/// contraction's indexing maps, and where the classification picks up its
/// dependence on the contraction: the same scale map is reduction-invariant in
/// `matmul` and reduction-varying in `matmul_transpose_b`.
static std::tuple<AffineMap, AffineMap, std::optional<AffineMap>>
getInputAndQParamMaps(DequantizeAffineOp dequant, AffineMap operandMap) {
  AffineMap toDequantIteration =
      inversePermutation(dequant.getOutputMap()).compose(operandMap);
  return {dequant.getInputMap().compose(toDequantIteration),
          dequant.getScaleMap().compose(toDequantIteration),
          dequant.isSymmetric()
              ? std::nullopt
              : std::optional(
                    dequant.getZeroPointMap().compose(toDequantIteration))};
}

/// Analyzes `op`, returning failure if it is not a quantized contraction at
/// all. A successful result may still have `Form::Partial` or `Form::None`,
/// which the rewrite declines.
static FailureOr<QuantizedContractionDetail>
getQuantizedContraction(linalg::LinalgOp op) {
  if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) {
    return failure();
  }
  if (!op.hasPureTensorSemantics() || !hasFloatMulAddBody(op)) {
    return failure();
  }

  // The dequantize has to be the immediate producer.
  QuantizedContractionDetail detail;
  detail.lhsDequant = op.getDpsInputs()[0].getDefiningOp<DequantizeAffineOp>();
  detail.rhsDequant = op.getDpsInputs()[1].getDefiningOp<DequantizeAffineOp>();
  if (!detail.lhsDequant || !detail.rhsDequant) {
    return failure();
  }

  // The body may reach its operands through casts, so the dequantized values
  // are not necessarily the contraction's element type. The epilogue multiplies
  // the scales into the result directly, so require them to agree.
  Type resultElementType =
      cast<ShapedType>(op.getDpsInits()[0].getType()).getElementType();
  for (DequantizeAffineOp dequant : {detail.lhsDequant, detail.rhsDequant}) {
    if (dequant.getOutputType().getElementType() != resultElementType) {
      return failure();
    }
  }

  detail.outMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  // The output map has to be a projected permutation for the epilogue to be
  // expressible over the result index space.
  if (!detail.outMap.isProjectedPermutation()) {
    return failure();
  }

  std::tie(detail.lhsMap, detail.lhsScaleMap, detail.lhsZeroPointMap) =
      getInputAndQParamMaps(detail.lhsDequant, op.getMatchingIndexingMap(
                                                   op.getDpsInputOperand(0)));
  std::tie(detail.rhsMap, detail.rhsScaleMap, detail.rhsZeroPointMap) =
      getInputAndQParamMaps(detail.rhsDequant, op.getMatchingIndexingMap(
                                                   op.getDpsInputOperand(1)));

  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  SmallVector<int64_t> loopRanges = op.getStaticLoopRanges();
  int64_t reductionExtent = 1;
  for (auto [dim, iteratorType] : llvm::enumerate(iteratorTypes)) {
    if (iteratorType != utils::IteratorType::reduction) {
      continue;
    }
    // Every reduction dim must be contracted over both inputs, otherwise this
    // is not a contraction and the algebra does not apply.
    if (!detail.lhsMap.isFunctionOfDim(dim) ||
        !detail.rhsMap.isFunctionOfDim(dim)) {
      return failure();
    }
    if (detail.lhsScaleMap.isFunctionOfDim(dim) ||
        detail.rhsScaleMap.isFunctionOfDim(dim) ||
        (detail.lhsZeroPointMap &&
         detail.lhsZeroPointMap->isFunctionOfDim(dim)) ||
        (detail.rhsZeroPointMap &&
         detail.rhsZeroPointMap->isFunctionOfDim(dim))) {
      detail.floatReductionDims.push_back(dim);
      continue;
    }
    detail.intReductionDims.push_back(dim);
    if (ShapedType::isDynamic(loopRanges[dim])) {
      reductionExtent = ShapedType::kDynamic;
    } else if (reductionExtent != ShapedType::kDynamic) {
      reductionExtent *= loopRanges[dim];
    }
  }
  if (reductionExtent != ShapedType::kDynamic) {
    detail.staticReductionExtent = reductionExtent;
  }

  if (detail.intReductionDims.empty()) {
    detail.form = QuantizedContractionDetail::Form::None;
  } else if (detail.floatReductionDims.empty()) {
    detail.form = QuantizedContractionDetail::Form::Full;
  } else {
    detail.form = QuantizedContractionDetail::Form::Partial;
  }
  return detail;
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

/// Creates a zero-filled tensor of `sizes` and `elementType`. Sizes are
/// OpFoldResults so that a dynamic dimension contributes its extent as a value.
static Value createZeroFilledTensor(OpBuilder &b, Location loc,
                                    ArrayRef<OpFoldResult> sizes,
                                    Type elementType) {
  Value empty = tensor::EmptyOp::create(b, loc, sizes, elementType);
  Value zero = arith::ConstantOp::create(b, loc, b.getZeroAttr(elementType));
  return linalg::FillOp::create(b, loc, zero, empty).getResult(0);
}

/// The extent of every iteration dimension of `op`, static where known and an
/// index value otherwise.
static SmallVector<OpFoldResult>
getIterationSizes(OpBuilder &b, Location loc, linalg::LinalgOp op,
                  QuantizedContractionDetail &detail) {
  // The standard way to size the iteration space is to concatenate the operand
  // dims and run them through the shapes-to-loops map. Substitute the quantized
  // input for each dequantized operand while doing so: a tensor.dim on a
  // dequantize result would give that op a use outliving the rewrite meant to
  // remove it, and dequantize_affine preserves shape so the extents are equal.
  SmallVector<OpFoldResult> flatShapes;
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    if (auto dequant = value.getDefiningOp<DequantizeAffineOp>()) {
      value = dequant.getInput();
    }
    llvm::append_range(flatShapes, tensor::getMixedSizes(b, loc, value));
  }
  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      op.getShapesToLoopsMap().getResults(), [&](AffineExpr expr) {
        return affine::makeComposedFoldedAffineApply(b, loc, expr, flatShapes);
      });

  // The map above resolves each dim from the first operand that mentions it,
  // which for a result dim is an input rather than the init. Override those
  // from the init: the replacement has to carry the original result type, so a
  // dim that is dynamic there stays dynamic even where an input knows it
  // statically.
  SmallVector<OpFoldResult> initSizes =
      tensor::getMixedSizes(b, loc, op.getDpsInits()[0]);
  for (auto [position, expr] : llvm::enumerate(detail.outMap.getResults())) {
    sizes[cast<AffineDimExpr>(expr).getPosition()] = initSizes[position];
  }
  return sizes;
}

/// Builds the integer contraction `D = sum_k Aq[k] * Bq[k]`, reusing the
/// original op's iteration space, indexing maps and iterator types so that
/// every contraction and convolution variant is handled without special casing.
static Value buildIntegerContraction(OpBuilder &b, Location loc,
                                     linalg::LinalgOp op,
                                     QuantizedContractionDetail &detail,
                                     ArrayRef<OpFoldResult> iterationSizes,
                                     IntegerType accumulatorType) {
  Value lhs = detail.lhsDequant.getInput();
  Value rhs = detail.rhsDequant.getInput();
  bool lhsUnsigned = detail.lhsDequant.getInputUnsigned();
  bool rhsUnsigned = detail.rhsDequant.getInputUnsigned();

  MLIRContext *ctx = b.getContext();

  // For the partial form the dims the quantization parameters vary along stay
  // parallel here and are appended to the result, so that the contraction is
  // integer within a block and the epilogue reduces across blocks.
  SmallVector<AffineExpr> resultExprs(detail.outMap.getResults());
  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
  for (unsigned dim : detail.floatReductionDims) {
    resultExprs.push_back(getAffineDimExpr(dim, ctx));
    iteratorTypes[dim] = utils::IteratorType::parallel;
  }
  AffineMap resultMap =
      AffineMap::get(detail.outMap.getNumDims(), 0, resultExprs, ctx);
  SmallVector<OpFoldResult> resultSizes =
      applyPermutationMap<OpFoldResult>(resultMap, iterationSizes);

  Value init = createZeroFilledTensor(b, loc, resultSizes, accumulatorType);

  SmallVector<AffineMap> maps{detail.lhsMap, detail.rhsMap, resultMap};
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

/// Sums one quantized operand over the contraction dims that the quantization
/// parameters do not vary along.
///
/// The sum is stated over the contraction's own iteration space, restricted to
/// the dims the operand depends on, rather than over the operand's index space.
/// That is what lets a windowed access work: a convolution's image reaches the
/// window dims only through `oh * stride + kh * dilation`, which no mask over
/// the image's own dimensions can express, and the resulting sum is a pooling
/// over the window rather than a reduction of the image.
///
/// `resultIterationDims` receives the iteration dims the result ends up indexed
/// by, in order, which is what the epilogue needs to broadcast it against the
/// result.
///
/// A windowed sum re-reads each element once per window position, so its trip
/// count is the contraction's divided by the extents of the dims its operand
/// does not depend on: F for a convolution, and 1 for a depthwise one, where
/// the sum is a second loop nest the size of the convolution itself. That worst
/// case is still a large win and is not worth gating on. The float path it
/// replaces fuses both dequantizes into the convolution, so it pays a widen, a
/// convert, a zero point subtract and a scale multiply per operand per multiply
/// accumulate, which costs far more than a second integer pass.
static Value
sumQuantizedOperand(OpBuilder &b, Location loc, DequantizeAffineOp dequant,
                    AffineMap operandMap, ArrayRef<unsigned> reducedDims,
                    ArrayRef<OpFoldResult> iterationSizes, Type accumulatorType,
                    SmallVector<unsigned> &resultIterationDims) {
  MLIRContext *ctx = b.getContext();
  unsigned numDims = operandMap.getNumDims();

  // The sum's iteration space is the dims the operand depends on. Every reduced
  // dim is among them, because getQuantizedContraction requires each reduction
  // dim to be contracted over both operands. Dropping the rest is what keeps
  // the result from being broadcast over dims it does not vary along, which for
  // a convolution's image sum is F.
  SmallVector<unsigned> sumDims;
  for (unsigned dim : llvm::seq<unsigned>(numDims)) {
    if (operandMap.isFunctionOfDim(dim)) {
      sumDims.push_back(dim);
    }
  }

  // Compress the contraction's dim positions down into the sum's own space. A
  // dim the operand does not depend on is unreachable from these maps, so its
  // replacement is never used.
  SmallVector<AffineExpr> toSumSpace(numDims, getAffineConstantExpr(0, ctx));
  SmallVector<utils::IteratorType> iteratorTypes;
  SmallVector<AffineExpr> resultExprs;
  resultIterationDims.clear();
  for (auto [position, dim] : llvm::enumerate(sumDims)) {
    toSumSpace[dim] = getAffineDimExpr(position, ctx);
    if (llvm::is_contained(reducedDims, dim)) {
      iteratorTypes.push_back(utils::IteratorType::reduction);
      continue;
    }
    // A reduction dim the quantization parameters vary along stays parallel, so
    // the blockwise form gets one partial sum per block, matching the partial
    // results of the integer contraction.
    iteratorTypes.push_back(utils::IteratorType::parallel);
    resultExprs.push_back(getAffineDimExpr(position, ctx));
    resultIterationDims.push_back(dim);
  }
  unsigned sumRank = sumDims.size();
  AffineMap inputMap = operandMap.replaceDimsAndSymbols(
      toSumSpace, /*symReplacements=*/{}, sumRank, /*numResultSyms=*/0);
  AffineMap resultMap = AffineMap::get(sumRank, 0, resultExprs, ctx);

  SmallVector<Value> inputs{dequant.getInput()};
  SmallVector<AffineMap> maps{inputMap};

  // linalg derives the iteration domain by inverting the concatenated operand
  // maps, and `LinalgOp::verify` rejects an op where that inverse does not
  // exist. Inversion only recovers a dim that appears as a bare result, which a
  // window dim never does: the image names it only inside the strided
  // expression. Give those dims a shape-only operand that the body ignores,
  // which is the same device `linalg.pooling_*` uses for its kernel operand.
  llvm::SmallBitVector named(sumRank);
  for (AffineMap map : {inputMap, resultMap}) {
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        named.set(dimExpr.getPosition());
      }
    }
  }
  SmallVector<AffineExpr> witnessExprs;
  SmallVector<OpFoldResult> witnessSizes;
  for (unsigned position : llvm::seq<unsigned>(sumRank)) {
    if (named.test(position)) {
      continue;
    }
    witnessExprs.push_back(getAffineDimExpr(position, ctx));
    witnessSizes.push_back(iterationSizes[sumDims[position]]);
  }
  if (!witnessExprs.empty()) {
    inputs.push_back(tensor::EmptyOp::create(
        b, loc, witnessSizes, dequant.getInputType().getElementType()));
    maps.push_back(AffineMap::get(sumRank, 0, witnessExprs, ctx));
  }
  maps.push_back(resultMap);

  SmallVector<OpFoldResult> resultSizes = llvm::map_to_vector(
      resultIterationDims, [&](unsigned dim) { return iterationSizes[dim]; });
  Value init = createZeroFilledTensor(b, loc, resultSizes, accumulatorType);

  bool isUnsigned = dequant.getInputUnsigned();
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
  return genericOp.getResult(0);
}

/// Rewrites a map expressed over the contraction's iteration space into one
/// over the epilogue's iteration space, which is the result index space
/// followed by the dims the quantization parameters vary along (empty for the
/// full form).
static AffineMap toEpilogueSpace(AffineMap map, AffineMap outMap,
                                 ArrayRef<unsigned> floatReductionDims,
                                 MLIRContext *ctx) {
  // outMap is a projected permutation, so each of its results names the
  // iteration dim that a given result position corresponds to.
  SmallVector<AffineExpr> replacements(map.getNumDims(),
                                       getAffineConstantExpr(0, ctx));
  for (auto [resultPos, expr] : llvm::enumerate(outMap.getResults())) {
    unsigned iterationDim = cast<AffineDimExpr>(expr).getPosition();
    replacements[iterationDim] = getAffineDimExpr(resultPos, ctx);
  }
  unsigned epilogueRank = outMap.getNumResults();
  for (unsigned dim : floatReductionDims) {
    replacements[dim] = getAffineDimExpr(epilogueRank++, ctx);
  }
  return map.replaceDimsAndSymbols(replacements, /*symReplacements=*/{},
                                   epilogueRank, /*numResultSyms=*/0);
}

namespace {
/// One input of the correction epilogue: a value and how the epilogue's
/// iteration space indexes it.
struct EpilogueOperand {
  Value value;
  AffineMap map;
};
} // namespace

/// Builds the epilogue
///   C = sA*sB*(D - zB*RA - zA*RB + N*zA*zB)
/// as a single elementwise `linalg.generic` over the result index space. The
/// corrections are done in the accumulator type and converted to float once,
/// because D exceeds the f32 mantissa for realistic contraction depths.
static Value buildCorrectionEpilogue(
    OpBuilder &b, Location loc, linalg::LinalgOp op,
    QuantizedContractionDetail &detail, Value contraction, Value lhsSum,
    ArrayRef<unsigned> lhsSumDims, Value rhsSum, ArrayRef<unsigned> rhsSumDims,
    Value reductionExtent, ArrayRef<OpFoldResult> iterationSizes,
    IntegerType accumulatorType) {
  MLIRContext *ctx = b.getContext();
  auto outputType = cast<RankedTensorType>(op.getDpsInits()[0].getType());
  unsigned resultRank = outputType.getRank();
  ArrayRef<unsigned> floatDims = detail.floatReductionDims;
  // In the partial form the epilogue reduces across the dims the quantization
  // parameters vary along, so its iteration space is the result index space
  // followed by those dims.
  bool isReduction = !floatDims.empty();
  unsigned epilogueRank = resultRank + floatDims.size();
  AffineMap resultMap =
      AffineMap::get(epilogueRank, 0,
                     llvm::map_to_vector(llvm::seq<unsigned>(resultRank),
                                         [&](unsigned dim) -> AffineExpr {
                                           return getAffineDimExpr(dim, ctx);
                                         }),
                     ctx);
  AffineMap contractionMap =
      AffineMap::getMultiDimIdentityMap(epilogueRank, ctx);

  // Maps an operand indexed by a subset of the contraction's iteration dims
  // into the epilogue's iteration space.
  auto dimsToEpilogueMap = [&](ArrayRef<unsigned> dims) {
    SmallVector<AffineExpr> exprs =
        llvm::map_to_vector(dims, [&](unsigned dim) -> AffineExpr {
          return getAffineDimExpr(dim, ctx);
        });
    return toEpilogueSpace(
        AffineMap::get(detail.outMap.getNumDims(), 0, exprs, ctx),
        detail.outMap, floatDims, ctx);
  };

  SmallVector<EpilogueOperand> operands;
  operands.push_back({contraction, contractionMap});

  // Index of each optional operand in the block arguments, or -1 if absent.
  int lhsSumIdx = -1, rhsSumIdx = -1, lhsZpIdx = -1, rhsZpIdx = -1;
  auto addOperand = [&](Value value, AffineMap map) {
    operands.push_back({value, map});
    return static_cast<int>(operands.size()) - 1;
  };

  AffineMap lhsScaleResultMap =
      toEpilogueSpace(detail.lhsScaleMap, detail.outMap, floatDims, ctx);
  AffineMap rhsScaleResultMap =
      toEpilogueSpace(detail.rhsScaleMap, detail.outMap, floatDims, ctx);

  if (lhsSum) {
    lhsSumIdx = addOperand(lhsSum, dimsToEpilogueMap(lhsSumDims));
    rhsZpIdx = addOperand(detail.rhsDequant.getZeroPoint(),
                          toEpilogueSpace(*detail.rhsZeroPointMap,
                                          detail.outMap, floatDims, ctx));
  }
  if (rhsSum) {
    rhsSumIdx = addOperand(rhsSum, dimsToEpilogueMap(rhsSumDims));
    lhsZpIdx = addOperand(detail.lhsDequant.getZeroPoint(),
                          toEpilogueSpace(*detail.lhsZeroPointMap,
                                          detail.outMap, floatDims, ctx));
  }
  int lhsScaleIdx = addOperand(detail.lhsDequant.getScale(), lhsScaleResultMap);
  int rhsScaleIdx = addOperand(detail.rhsDequant.getScale(), rhsScaleResultMap);

  bool lhsZpUnsigned = detail.lhsDequant.getZpUnsigned();
  bool rhsZpUnsigned = detail.rhsDequant.getZpUnsigned();
  Type realType = outputType.getElementType();

  SmallVector<Value> inputs = llvm::map_to_vector(
      operands, [](const EpilogueOperand &operand) { return operand.value; });
  SmallVector<AffineMap> maps = llvm::map_to_vector(
      operands, [](const EpilogueOperand &operand) { return operand.map; });
  maps.push_back(resultMap);

  // A reduction epilogue accumulates into its init, so it has to start at zero;
  // the elementwise form writes every element exactly once.
  SmallVector<OpFoldResult> resultSizes =
      applyPermutationMap<OpFoldResult>(detail.outMap, iterationSizes);
  Value init =
      isReduction
          ? createZeroFilledTensor(b, loc, resultSizes, realType)
          : tensor::EmptyOp::create(b, loc, resultSizes, realType).getResult();
  SmallVector<utils::IteratorType> iteratorTypes(resultRank,
                                                 utils::IteratorType::parallel);
  iteratorTypes.append(floatDims.size(), utils::IteratorType::reduction);

  auto genericOp = linalg::GenericOp::create(
      b, loc, init.getType(), inputs, ValueRange{init}, maps, iteratorTypes,
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value accumulated = args[0];
        Value lhsZp, rhsZp;
        if (lhsZpIdx >= 0) {
          lhsZp = convertScalarToDtype(nested, nestedLoc, args[lhsZpIdx],
                                       accumulatorType,
                                       /*isUnsignedCast=*/lhsZpUnsigned);
        }
        if (rhsZpIdx >= 0) {
          rhsZp = convertScalarToDtype(nested, nestedLoc, args[rhsZpIdx],
                                       accumulatorType,
                                       /*isUnsignedCast=*/rhsZpUnsigned);
        }
        // - zB*RA
        if (lhsSumIdx >= 0) {
          Value term =
              arith::MulIOp::create(nested, nestedLoc, rhsZp, args[lhsSumIdx]);
          accumulated =
              arith::SubIOp::create(nested, nestedLoc, accumulated, term);
        }
        // - zA*RB
        if (rhsSumIdx >= 0) {
          Value term =
              arith::MulIOp::create(nested, nestedLoc, lhsZp, args[rhsSumIdx]);
          accumulated =
              arith::SubIOp::create(nested, nestedLoc, accumulated, term);
        }
        // + N*zA*zB
        if (lhsZp && rhsZp) {
          Value product =
              arith::MulIOp::create(nested, nestedLoc, lhsZp, rhsZp);
          Value term = arith::MulIOp::create(nested, nestedLoc, product,
                                             reductionExtent);
          accumulated =
              arith::AddIOp::create(nested, nestedLoc, accumulated, term);
        }
        // Convert once, then apply the scales. A dequantization op is free to
        // carry its scale in a wider type than its output, so the scales are
        // narrowed here; this rewrite has already restructured the arithmetic
        // into `sum(qa*qb) * (sa*sb)`, so where the narrowing happens within it
        // is not observable against the reference either way.
        Value real =
            arith::SIToFPOp::create(nested, nestedLoc, realType, accumulated);
        Value lhsScale =
            convertScalarToDtype(nested, nestedLoc, args[lhsScaleIdx], realType,
                                 /*isUnsignedCast=*/false);
        Value rhsScale =
            convertScalarToDtype(nested, nestedLoc, args[rhsScaleIdx], realType,
                                 /*isUnsignedCast=*/false);
        Value scale =
            arith::MulFOp::create(nested, nestedLoc, lhsScale, rhsScale);
        Value result = arith::MulFOp::create(nested, nestedLoc, real, scale);
        if (isReduction) {
          result =
              arith::AddFOp::create(nested, nestedLoc, args.back(), result);
        }
        linalg::YieldOp::create(nested, nestedLoc, result);
      });
  return genericOp.getResult(0);
}

/// Returns true if `init` is a fill with a floating point zero, which is what
/// lets the integer contraction start from an integer zero without having to
/// carry the original accumulator into the epilogue.
static bool isZeroFill(Value init) {
  auto fillOp = init.getDefiningOp<linalg::FillOp>();
  if (!fillOp) {
    return false;
  }
  return matchPattern(fillOp.getInputs()[0], m_AnyZeroFloat());
}

struct ConvertQDQToIntegerMath : OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<QuantizedContractionDetail> maybeDetail =
        getQuantizedContraction(op);
    if (failed(maybeDetail)) {
      return failure();
    }
    QuantizedContractionDetail detail = *maybeDetail;

    // None can never become an integer contraction: there is no reduction dim
    // left to contract over. It keeps the float path, which stays correct.
    if (detail.form == QuantizedContractionDetail::Form::None) {
      return rewriter.notifyMatchFailure(
          op, "quantization parameters vary along every reduction dim");
    }

    if (!isZeroFill(op.getDpsInits()[0])) {
      return rewriter.notifyMatchFailure(op, "accumulator init is not a zero "
                                             "fill");
    }

    if (!accumulatorFits(kAccumulatorWidth, detail)) {
      return rewriter.notifyMatchFailure(
          op, "storage types and reduction depth could overflow the integer "
              "accumulator");
    }

    Location loc = op.getLoc();
    auto accumulatorType =
        IntegerType::get(rewriter.getContext(), kAccumulatorWidth);

    SmallVector<OpFoldResult> iterationSizes =
        getIterationSizes(rewriter, loc, op, detail);

    Value contraction = buildIntegerContraction(
        rewriter, loc, op, detail, iterationSizes, accumulatorType);

    SmallVector<unsigned> lhsSumDims, rhsSumDims;
    Value lhsSum, rhsSum;
    if (detail.needsLhsSum()) {
      lhsSum = sumQuantizedOperand(rewriter, loc, detail.lhsDequant,
                                   detail.lhsMap, detail.intReductionDims,
                                   iterationSizes, accumulatorType, lhsSumDims);
    }
    if (detail.needsRhsSum()) {
      rhsSum = sumQuantizedOperand(rewriter, loc, detail.rhsDequant,
                                   detail.rhsMap, detail.intReductionDims,
                                   iterationSizes, accumulatorType, rhsSumDims);
    }

    // The N*zA*zB term is the only user of the extent, so only materialize it
    // when both sides carry a zero point.
    Value reductionExtent;
    if (detail.needsLhsSum() && detail.needsRhsSum()) {
      if (detail.staticReductionExtent) {
        reductionExtent = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getIntegerAttr(accumulatorType,
                                    *detail.staticReductionExtent));
      } else {
        Value extent = arith::ConstantIndexOp::create(rewriter, loc, 1);
        for (unsigned dim : detail.intReductionDims) {
          Value size = getValueOrCreateConstantIndexOp(rewriter, loc,
                                                       iterationSizes[dim]);
          extent = arith::MulIOp::create(rewriter, loc, extent, size);
        }
        reductionExtent =
            arith::IndexCastOp::create(rewriter, loc, accumulatorType, extent);
      }
    }

    Value result = buildCorrectionEpilogue(
        rewriter, loc, op, detail, contraction, lhsSum, lhsSumDims, rhsSum,
        rhsSumDims, reductionExtent, iterationSizes, accumulatorType);
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
