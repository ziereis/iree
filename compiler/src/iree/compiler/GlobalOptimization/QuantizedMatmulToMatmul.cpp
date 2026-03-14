// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::GlobalOptimization {

#define GEN_PASS_DEF_LINALGQUANTIZEDMATMULTOMATMULPASS
#include "iree/compiler/GlobalOptimization/Passes.h.inc"

namespace {

bool isConstantZero(Value val) {
  auto constIntOp = val.getDefiningOp<arith::ConstantIntOp>();
  return constIntOp && constIntOp.value() == 0;
}

// Pattern lowering quantized_matmul to matmul and quantized_batch_matmul to
// batch_matmul op.
// This is implementing the math explained in Section 2.3 of
// https://arxiv.org/abs/1712.05877.
struct QuantizedMatmulToMatmul
    : public OpInterfaceRewritePattern<linalg::LinalgOp> {
  using OpInterfaceRewritePattern<linalg::LinalgOp>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(linalg::LinalgOp op,
                                PatternRewriter &rewriter) const override {
    // Fails when the operation is neither quantized_matmul or
    // quantized_batch_matmul.
    if (!isa<linalg::QuantizedMatmulOp, linalg::QuantizedBatchMatmulOp>(op)) {
      return failure();
    }

    Location loc = op.getLoc();
    SmallVector<Value> inputs = op.getDpsInputs();
    bool batch = isa<linalg::QuantizedBatchMatmulOp>(op) ? true : false;
    ImplicitLocOpBuilder builder(loc, rewriter);
    assert(inputs.size() == 4);
    Value lhs = inputs[0];
    Value rhs = inputs[1];
    Value lhsZp = inputs[2];
    Value rhsZp = inputs[3];
    auto lhsTy = dyn_cast<ShapedType>(lhs.getType());
    unsigned lhsRank = lhsTy.getRank();
    Value acc = op.getDpsInits()[0];
    // Compute the matmul part.
    Value matmul = batch ? linalg::BatchMatmulOp::create(
                               builder, ValueRange{lhs, rhs}, ValueRange{acc})
                               .getResult(0)
                         : linalg::MatmulOp::create(
                               builder, ValueRange{lhs, rhs}, ValueRange{acc})
                               .getResult(0);
    bool lhsZpIsConstantZero = isConstantZero(lhsZp);
    bool rhsZpIsConstantZero = isConstantZero(rhsZp);
    if (lhsZpIsConstantZero && rhsZpIsConstantZero) {
      // Easy case: both zero points are constant zeros, so the quantized_matmul
      // was just a matmul all along.
      rewriter.replaceOp(op, matmul);
      return success();
    }
    // Create the result. No need to zero-fill it as we will overwrite it.
    ShapedType accType = cast<ShapedType>(acc.getType());
    Value initResult = tensor::EmptyOp::create(
        builder, tensor::getMixedSizes(builder, loc, acc),
        accType.getElementType());
    // Create the indexing maps for the generic.
    MLIRContext *context = rewriter.getContext();
    AffineExpr b, m, n;
    batch ? bindDims(context, b, m, n) : bindDims(context, m, n);
    AffineMap mapToNone = AffineMap::get(lhsRank, 0, context);
    AffineMap mapToRowDim = batch ? AffineMap::get(lhsRank, 0, {b, m}, context)
                                  : AffineMap::get(lhsRank, 0, m, context);
    AffineMap mapToColumnDim = batch
                                   ? AffineMap::get(lhsRank, 0, {b, n}, context)
                                   : AffineMap::get(lhsRank, 0, n, context);
    AffineMap mapIdentity = batch
                                ? AffineMap::get(lhsRank, 0, {b, m, n}, context)
                                : AffineMap::get(lhsRank, 0, {m, n}, context);
    SmallVector<AffineMap> indexingMaps;
    SmallVector<Value> ins;
    auto addInput = [&](Value val, AffineMap map) -> int {
      ins.push_back(val);
      indexingMaps.push_back(map);
      return ins.size() - 1;
    };
    int indexOfMatmulInput = addInput(matmul, mapIdentity);
    int indexOfLhsSumsInput = 0;
    int indexOfLhsZpInput = 0;
    int indexOfRhsSumsInput = 0;
    int indexOfRhsZpInput = 0;
    int indexOfLhsZpTimesRhsZpTimesKSizeInput = 0;
    Type accElTy = accType.getElementType();
    if (!rhsZpIsConstantZero) {
      SmallVector<bool> colRedIterator(lhsRank, false);
      colRedIterator.back() = true;
      Value lhsSums =
          sumReduceDimensionSubset(builder, lhs, accElTy, colRedIterator);
      indexOfLhsSumsInput = addInput(lhsSums, mapToRowDim);
      indexOfRhsZpInput = addInput(rhsZp, mapToNone);
    }
    if (!lhsZpIsConstantZero) {
      SmallVector<bool> rowRedIterator(lhsRank, false);
      rowRedIterator[static_cast<int>(batch)] = true;
      Value rhsSums =
          sumReduceDimensionSubset(builder, rhs, accElTy, rowRedIterator);
      indexOfRhsSumsInput = addInput(rhsSums, mapToColumnDim);
      indexOfLhsZpInput = addInput(lhsZp, mapToNone);
    }
    if (!lhsZpIsConstantZero && !rhsZpIsConstantZero) {
      Value lhsZpTimesRhsZp = arith::MulIOp::create(builder, lhsZp, rhsZp);

      Value kSize = arith::IndexCastOp::create(
          rewriter, loc, accElTy,
          tensor::DimOp::create(builder, lhs, batch ? 2 : 1));
      Value lhsZpTimesRhsZpTimesKSize =
          arith::MulIOp::create(builder, lhsZpTimesRhsZp, kSize);
      indexOfLhsZpTimesRhsZpTimesKSizeInput =
          addInput(lhsZpTimesRhsZpTimesKSize, mapToNone);
    }
    // Add the indexing map for the initResult 'output' even though it's unused
    indexingMaps.push_back(mapIdentity);
    // Create the generic putting all the terms together.
    SmallVector<utils::IteratorType> iterators(lhsRank,
                                               utils::IteratorType::parallel);
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, acc.getType(), ins, ValueRange{initResult}, indexingMaps, iterators,
        [=](OpBuilder &b, Location loc, ValueRange args) {
          Value matmulEl = args[indexOfMatmulInput];
          Value lhsSumsEl = args[indexOfLhsSumsInput];
          Value rhsSumsEl = args[indexOfRhsSumsInput];
          Value lhsZp = args[indexOfLhsZpInput];
          Value rhsZp = args[indexOfRhsZpInput];
          Value lhsZpTimesRhsZpTimesKSize =
              args[indexOfLhsZpTimesRhsZpTimesKSizeInput];
          Value result = matmulEl;
          // If the rhs zero-point is not a constant zero, we need to add it
          // times the sums along rows of lhs.
          if (!rhsZpIsConstantZero) {
            Value lhsSumsElTimesRhsZp =
                arith::MulIOp::create(b, loc, lhsSumsEl, rhsZp);
            result = arith::SubIOp::create(b, loc, result, lhsSumsElTimesRhsZp);
          }
          // If the lhs zero-point is not a constant zero, we need to add it
          // times the sums along columns of rhs.
          if (!lhsZpIsConstantZero) {
            Value rhsSumsElTimesLhsZp =
                arith::MulIOp::create(b, loc, rhsSumsEl, lhsZp);
            result = arith::SubIOp::create(b, loc, result, rhsSumsElTimesLhsZp);
          }
          // Add the final correction term, if neither zero-point is cst zero.
          if (!lhsZpIsConstantZero && !rhsZpIsConstantZero) {
            result = arith::AddIOp::create(b, loc, result,
                                           lhsZpTimesRhsZpTimesKSize);
          }
          linalg::YieldOp::create(b, loc, result);
        });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Generic quantized contraction decomposition
//===----------------------------------------------------------------------===//
//
// Matches a linalg.generic that computes:
//   result = (ext(A) - zp_a) * (ext(B) - zp_b) + acc   [both asymmetric]
//   result = (ext(A) - zp_a) * ext(B) + acc             [LHS asymmetric only]
//   result = ext(A) * (ext(B) - zp_b) + acc             [RHS asymmetric only]
//
// where ext is extsi or extui (possibly different per side), and zero points
// can be scalar, per-channel, or per-block (any rank/indexing).
//
// Decomposes into:
//   term1: matmul[...] = sum_k ext(A) * ext(B)
//   term2: lhsSums[...] = sum_k ext(A)          (if rhs has zero point)
//   term3: rhsSums[...] = sum_k ext(B)          (if lhs has zero point)
//   term4: result = matmul - zp_b * lhsSums - zp_a * rhsSums
//                   + zp_a * zp_b * K            (if both have zero points)
//===----------------------------------------------------------------------===//

/// Information extracted from analyzing the body of a linalg.generic that
/// represents a quantized contraction.
struct QuantizedContractionInfo {
  // Block argument indices for the matmul operands (the narrow-type tensors).
  int lhsArgIdx;
  int rhsArgIdx;
  // Extension kind for each side.
  IntExtKind lhsExt;
  IntExtKind rhsExt;
  // Block argument index for the zero point, or -1 if symmetric (no zp).
  int lhsZpArgIdx = -1;
  int rhsZpArgIdx = -1;
  // Block argument index for the accumulator.
  int accArgIdx;
};

/// Try to match a block argument through an optional extension op.
/// Returns the block argument and the extension kind if found.
static std::optional<std::pair<BlockArgument, IntExtKind>>
matchExtendedBlockArg(Value val) {
  if (auto extsi = val.getDefiningOp<arith::ExtSIOp>()) {
    if (auto arg = dyn_cast<BlockArgument>(extsi.getIn()))
      return std::make_pair(arg, IntExtKind::ExtSI);
  }
  if (auto extui = val.getDefiningOp<arith::ExtUIOp>()) {
    if (auto arg = dyn_cast<BlockArgument>(extui.getIn()))
      return std::make_pair(arg, IntExtKind::ExtUI);
  }
  return std::nullopt;
}

/// Analyze the body of a linalg.generic to extract quantized contraction info.
///
/// Expected body patterns (order of multiply operands may vary):
///   (ext(a) - zp_a) * (ext(b) - zp_b) + acc
///   (ext(a) - zp_a) * ext(b) + acc
///   ext(a) * (ext(b) - zp_b) + acc
///
/// The body must have exactly these operations plus the yield.
static std::optional<QuantizedContractionInfo>
matchQuantizedContractionBody(linalg::GenericOp genericOp) {
  Block *body = genericOp.getBody();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  if (yieldOp.getNumOperands() != 1)
    return std::nullopt;

  // Walk backward from yield: yield(addi(muli(...), acc))
  auto addOp = yieldOp.getOperand(0).getDefiningOp<arith::AddIOp>();
  if (!addOp)
    return std::nullopt;

  // One operand of the add should be the accumulator block arg,
  // the other should be a muli.
  arith::MulIOp mulOp = nullptr;
  BlockArgument accArg;
  for (int i = 0; i < 2; i++) {
    if (auto mul = addOp.getOperand(i).getDefiningOp<arith::MulIOp>()) {
      if (auto arg = dyn_cast<BlockArgument>(addOp.getOperand(1 - i))) {
        mulOp = mul;
        accArg = arg;
        break;
      }
    }
  }
  if (!mulOp)
    return std::nullopt;

  // Each side of the multiply is either:
  //   (a) ext(block_arg) - block_arg   [asymmetric, has zero point]
  //   (b) ext(block_arg)               [symmetric, no zero point]
  //
  // We need to handle both orderings of the multiply operands.
  struct SideInfo {
    BlockArgument dataArg;
    IntExtKind extKind;
    int zpArgIdx = -1; // -1 means symmetric
  };

  auto matchSide = [](Value val) -> std::optional<SideInfo> {
    SideInfo info;
    // Try pattern (a): subi(ext(block_arg), block_arg)
    if (auto subOp = val.getDefiningOp<arith::SubIOp>()) {
      auto extResult = matchExtendedBlockArg(subOp.getLhs());
      if (!extResult)
        return std::nullopt;
      info.dataArg = extResult->first;
      info.extKind = extResult->second;
      // The rhs of subi must be a block arg (the zero point).
      auto zpArg = dyn_cast<BlockArgument>(subOp.getRhs());
      if (!zpArg)
        return std::nullopt;
      info.zpArgIdx = zpArg.getArgNumber();
      return info;
    }
    // Try pattern (b): ext(block_arg)
    auto extResult = matchExtendedBlockArg(val);
    if (!extResult)
      return std::nullopt;
    info.dataArg = extResult->first;
    info.extKind = extResult->second;
    return info;
  };

  auto lhsInfo = matchSide(mulOp.getLhs());
  auto rhsInfo = matchSide(mulOp.getRhs());
  if (!lhsInfo || !rhsInfo)
    return std::nullopt;

  // At least one side must have a zero point, otherwise this is just a
  // regular matmul and doesn't need decomposition.
  if (lhsInfo->zpArgIdx == -1 && rhsInfo->zpArgIdx == -1)
    return std::nullopt;

  QuantizedContractionInfo result;
  result.lhsArgIdx = lhsInfo->dataArg.getArgNumber();
  result.rhsArgIdx = rhsInfo->dataArg.getArgNumber();
  result.lhsExt = lhsInfo->extKind;
  result.rhsExt = rhsInfo->extKind;
  result.lhsZpArgIdx = lhsInfo->zpArgIdx;
  result.rhsZpArgIdx = rhsInfo->zpArgIdx;
  result.accArgIdx = accArg.getArgNumber();
  return result;
}

/// Create an elementwise linalg.generic that adds a constant i8 value to every
/// element of the input tensor. Used to shift unsigned values into the signed
/// domain: extui(x) = extsi(x +_i8 -128) + 128.
static Value createI8ElementwiseAdd(ImplicitLocOpBuilder &builder, Value input,
                                    int8_t constant) {
  auto inputType = cast<RankedTensorType>(input.getType());
  int rank = inputType.getRank();
  MLIRContext *ctx = builder.getContext();
  SmallVector<AffineMap> maps(2, AffineMap::getMultiDimIdentityMap(rank, ctx));
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  Value init = tensor::EmptyOp::create(
      builder, tensor::getMixedSizes(builder, builder.getLoc(), input),
      inputType.getElementType());
  Value cst = arith::ConstantIntOp::create(builder, inputType.getElementType(),
                                           static_cast<int64_t>(constant));
  return linalg::GenericOp::create(
             builder, inputType, ValueRange{input}, ValueRange{init}, maps,
             iterators,
             [&](OpBuilder &b, Location loc, ValueRange args) {
               Value add = arith::AddIOp::create(b, loc, args[0], cst);
               linalg::YieldOp::create(b, loc, add);
             })
      .getResult(0);
}

/// Adjust a zero-point value by subtracting an integer constant.
/// If the zero point is a shaped type (tensor), creates an elementwise generic.
/// If it is a scalar, creates an arith.subi.
static Value adjustZeroPoint(ImplicitLocOpBuilder &builder, Value zp,
                             int32_t adjustment) {
  Type zpTy = zp.getType();
  if (auto shapedTy = dyn_cast<ShapedType>(zpTy)) {
    int rank = shapedTy.getRank();
    MLIRContext *ctx = builder.getContext();
    SmallVector<AffineMap> maps(2,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);
    Value init = tensor::EmptyOp::create(
        builder, tensor::getMixedSizes(builder, builder.getLoc(), zp),
        shapedTy.getElementType());
    Value cst = arith::ConstantIntOp::create(builder,
                                             shapedTy.getElementType(),
                                             static_cast<int64_t>(adjustment));
    return linalg::GenericOp::create(
               builder, shapedTy, ValueRange{zp}, ValueRange{init}, maps,
               iterators,
               [&](OpBuilder &b, Location loc, ValueRange args) {
                 Value sub = arith::SubIOp::create(b, loc, args[0], cst);
                 linalg::YieldOp::create(b, loc, sub);
               })
        .getResult(0);
  }
  // Scalar case.
  Value cst = arith::ConstantIntOp::create(builder, zpTy,
                                           static_cast<int64_t>(adjustment));
  return arith::SubIOp::create(builder, zp, cst);
}

/// Pattern to decompose a linalg.generic representing a quantized contraction
/// into a matmul + correction terms.
struct GenericQuantizedContractionDecomposition
    : public OpRewritePattern<linalg::GenericOp> {
  bool shiftToSignedDomain;
  GenericQuantizedContractionDecomposition(MLIRContext *context,
                                          bool shiftToSignedDomain)
      : OpRewritePattern<linalg::GenericOp>(context),
        shiftToSignedDomain(shiftToSignedDomain) {}

  LogicalResult matchAndRewrite(linalg::GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    // Must have at least one reduction dimension.
    if (genericOp.getNumReductionLoops() == 0)
      return failure();

    // Must have exactly one output.
    if (genericOp.getNumDpsInits() != 1)
      return failure();

    // All indexing maps must be projected permutations.
    if (!genericOp.hasPureTensorSemantics())
      return failure();
    for (auto map : genericOp.getIndexingMapsArray()) {
      if (!map.isProjectedPermutation())
        return failure();
    }

    // Try to match the body.
    auto info = matchQuantizedContractionBody(genericOp);
    if (!info)
      return failure();

    // Get the indexing maps and iterator types.
    SmallVector<AffineMap> allMaps = genericOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> iteratorTypes =
        genericOp.getIteratorTypesArray();
    int numLoops = iteratorTypes.size();

    // Extract the maps for the contraction operands.
    AffineMap lhsMap = allMaps[info->lhsArgIdx];
    AffineMap rhsMap = allMaps[info->rhsArgIdx];
    // The acc map is the output map (last map).
    AffineMap accMap = allMaps.back();

    // Validate contraction structure using the LHS, RHS, ACC maps.
    auto contractionDims =
        linalg::inferContractionDims(SmallVector<AffineMap>{lhsMap, rhsMap, accMap});
    if (failed(contractionDims))
      return failure();

    // Must have at least one M, N, and K dimension.
    if (contractionDims->m.empty() || contractionDims->n.empty() ||
        contractionDims->k.empty())
      return failure();

    // Gather values from the original op.
    SmallVector<Value> opOperands;
    for (auto operand : genericOp->getOperands())
      opOperands.push_back(operand);

    Value lhsVal = opOperands[info->lhsArgIdx];
    Value rhsVal = opOperands[info->rhsArgIdx];
    Value accVal = genericOp.getDpsInits()[0];

    Value lhsZpVal = info->lhsZpArgIdx >= 0
                         ? opOperands[info->lhsZpArgIdx]
                         : nullptr;
    Value rhsZpVal = info->rhsZpArgIdx >= 0
                         ? opOperands[info->rhsZpArgIdx]
                         : nullptr;

    bool hasLhsZp = lhsZpVal != nullptr;
    bool hasRhsZp = rhsZpVal != nullptr;

    Location loc = genericOp.getLoc();
    ImplicitLocOpBuilder builder(loc, rewriter);
    ShapedType accType = cast<ShapedType>(accVal.getType());
    Type accElTy = accType.getElementType();

    IntExtKind lhsExt = info->lhsExt;
    IntExtKind rhsExt = info->rhsExt;

    // --- Domain shifting ---
    // When shiftToSignedDomain is enabled and extensions are mixed (one extui,
    // one extsi), shift the unsigned operand into the signed domain so the
    // matmul uses extsi on both sides.
    //
    // The identity: extui(x) = extsi(x +_i8 -128) + 128
    //
    // This means:
    //   (extui(a) - zp_a) * extsi(b)
    //     = (extsi(a +_i8 -128) + 128 - zp_a) * extsi(b)
    //     = (extsi(a') - zp_a') * extsi(b)
    //   where a' = a +_i8 -128, zp_a' = zp_a - 128
    //
    // If the unsigned side was symmetric (no zero point), it becomes
    // asymmetric with zp = 128.
    if (shiftToSignedDomain && lhsExt != rhsExt) {
      // Determine which side needs shifting (the one using extui).
      auto shiftSide = [&](Value &dataVal, IntExtKind &ext, Value &zpVal,
                           bool &hasZp, AffineMap dataMap, int zpArgIdx) {
        if (ext != IntExtKind::ExtUI)
          return;
        // Shift data: a' = a +_i8 -128
        dataVal = createI8ElementwiseAdd(builder, dataVal, -128);
        ext = IntExtKind::ExtSI;
        if (hasZp) {
          // Adjust existing zero point: zp' = zp - 128
          zpVal = adjustZeroPoint(builder, zpVal, 128);
        } else {
          // Create a new zero point of 128 (scalar i32).
          zpVal = arith::ConstantIntOp::create(builder, accElTy,
                                               static_cast<int64_t>(128));
          hasZp = true;
        }
      };

      shiftSide(lhsVal, lhsExt, lhsZpVal, hasLhsZp, lhsMap,
                info->lhsZpArgIdx);
      shiftSide(rhsVal, rhsExt, rhsZpVal, hasRhsZp, rhsMap,
                info->rhsZpArgIdx);
    }

    // --- Term 1: matmul = sum_k ext(A) * ext(B) ---
    // Build a new generic with only LHS, RHS inputs and the accumulator output.
    SmallVector<AffineMap> matmulMaps = {lhsMap, rhsMap, accMap};
    Value matmul =
        linalg::GenericOp::create(
            builder, accType, ValueRange{lhsVal, rhsVal}, ValueRange{accVal},
            matmulMaps, iteratorTypes,
            [&](OpBuilder &b, Location loc, ValueRange args) {
              Value lhsEl, rhsEl;
              if (lhsExt == IntExtKind::ExtUI)
                lhsEl = arith::ExtUIOp::create(b, loc, accElTy, args[0]);
              else
                lhsEl = arith::ExtSIOp::create(b, loc, accElTy, args[0]);
              if (rhsExt == IntExtKind::ExtUI)
                rhsEl = arith::ExtUIOp::create(b, loc, accElTy, args[1]);
              else
                rhsEl = arith::ExtSIOp::create(b, loc, accElTy, args[1]);
              Value mul = arith::MulIOp::create(b, loc, lhsEl, rhsEl);
              Value add = arith::AddIOp::create(b, loc, mul, args[2]);
              linalg::YieldOp::create(b, loc, add);
            })
            .getResult(0);

    // If neither side has a zero point after all (shouldn't happen due to
    // earlier check, but be defensive), just replace with the matmul.
    if (!hasLhsZp && !hasRhsZp) {
      rewriter.replaceOp(genericOp, matmul);
      return success();
    }

    // Identify which loop dimensions are reduction dims.
    SmallVector<bool> isReductionDim(numLoops, false);
    for (unsigned k : contractionDims->k)
      isReductionDim[k] = true;

    // Build the reduction mask for LHS (reduce the K dims as they appear in
    // LHS's indexing map) to compute lhsSums.
    auto buildReductionMask = [&](AffineMap map) -> SmallVector<bool> {
      // The tensor rank equals the number of results in its indexing map.
      int tensorRank = map.getNumResults();
      SmallVector<bool> mask(tensorRank, false);
      for (unsigned i = 0; i < map.getNumResults(); i++) {
        auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
        if (dimExpr && isReductionDim[dimExpr.getPosition()])
          mask[i] = true;
      }
      return mask;
    };

    // --- Term 2: lhsSums = sum_k ext(A) (if rhs has zero point) ---
    Value lhsSums;
    if (hasRhsZp) {
      SmallVector<bool> lhsRedMask = buildReductionMask(lhsMap);
      lhsSums =
          sumReduceDimensionSubset(builder, lhsVal, accElTy, lhsRedMask,
                                   lhsExt);
    }

    // --- Term 3: rhsSums = sum_k ext(B) (if lhs has zero point) ---
    Value rhsSums;
    if (hasLhsZp) {
      SmallVector<bool> rhsRedMask = buildReductionMask(rhsMap);
      rhsSums =
          sumReduceDimensionSubset(builder, rhsVal, accElTy, rhsRedMask,
                                   rhsExt);
    }

    // --- Compute K size for the zp_a * zp_b * K term ---
    Value kSizeVal;
    if (hasLhsZp && hasRhsZp) {
      // K size is the product of all reduction dimension sizes.
      // Get the size from LHS tensor using the first K dim's position in LHS.
      Value kSize;
      for (unsigned k : contractionDims->k) {
        // Find which tensor dimension of LHS corresponds to loop dim k.
        for (unsigned i = 0; i < lhsMap.getNumResults(); i++) {
          auto dimExpr = dyn_cast<AffineDimExpr>(lhsMap.getResult(i));
          if (dimExpr && dimExpr.getPosition() == k) {
            Value dimSize = arith::IndexCastOp::create(
                builder, accElTy, tensor::DimOp::create(builder, lhsVal, i));
            kSize = kSize ? arith::MulIOp::create(builder, kSize, dimSize)
                                .getResult()
                          : dimSize;
            break;
          }
        }
      }
      kSizeVal = kSize;
    }

    // --- Term 4: Combine correction terms ---
    // Build the output map: same as accMap but only over parallel dims.
    // We need to build a pointwise generic over the parallel dimensions.

    // The correction generic iterates over only the parallel dimensions of the
    // output. We need to figure out indexing maps for all inputs.
    //
    // The output shape is the accumulator shape (accMap over parallel dims).
    // For lhsSums/rhsSums, their shapes correspond to the non-reduction dims
    // of the respective input tensors, which we need to map into the output's
    // iteration space.
    //
    // For zero points, their original indexing maps (from the source generic)
    // already express how they map to the full iteration space. We just need
    // to drop the reduction dimensions from these maps.

    // Build a map from old loop dims to new (parallel-only) loop dims.
    SmallVector<unsigned> parallelDims;
    for (int i = 0; i < numLoops; i++) {
      if (!isReductionDim[i])
        parallelDims.push_back(i);
    }
    int numParallelDims = parallelDims.size();

    // Map from old dim position -> new dim position (or -1 if reduction).
    SmallVector<int> oldToNewDim(numLoops, -1);
    for (int i = 0; i < numParallelDims; i++)
      oldToNewDim[parallelDims[i]] = i;

    // Helper to remap an affine map to only use parallel dimensions.
    auto remapToParallel = [&](AffineMap map) -> AffineMap {
      MLIRContext *ctx = rewriter.getContext();
      SmallVector<AffineExpr> newExprs;
      for (unsigned i = 0; i < map.getNumResults(); i++) {
        auto dimExpr = dyn_cast<AffineDimExpr>(map.getResult(i));
        assert(dimExpr && "expected projected permutation");
        int newPos = oldToNewDim[dimExpr.getPosition()];
        assert(newPos >= 0 && "zero point should not index reduction dims");
        newExprs.push_back(getAffineDimExpr(newPos, ctx));
      }
      return AffineMap::get(numParallelDims, 0, newExprs, ctx);
    };

    AffineMap corrAccMap = remapToParallel(accMap);

    // Build indexing maps and inputs for the correction generic.
    SmallVector<AffineMap> corrMaps;
    SmallVector<Value> corrIns;
    auto addCorrInput = [&](Value val, AffineMap map) -> int {
      corrIns.push_back(val);
      corrMaps.push_back(map);
      return corrIns.size() - 1;
    };

    int corrIdxMatmul = addCorrInput(matmul, corrAccMap);

    // Helper to compute the correction-space map for a zero point.
    // If the zero point existed in the original op, remap its original indexing
    // map. If it was introduced by domain shifting (scalar constant), use a
    // scalar map.
    auto getZpCorrMap = [&](int origZpArgIdx, Value zpVal) -> AffineMap {
      if (origZpArgIdx >= 0) {
        return remapToParallel(allMaps[origZpArgIdx]);
      }
      // Newly introduced scalar zero point from domain shifting.
      return AffineMap::get(numParallelDims, 0, rewriter.getContext());
    };

    int corrIdxLhsSums = 0, corrIdxRhsZp = 0;
    if (hasRhsZp) {
      // lhsSums has shape = LHS with K dims removed. Its indexing map in the
      // correction space maps the non-K dims of LHS to parallel dims.
      SmallVector<AffineExpr> lhsSumsExprs;
      for (unsigned i = 0; i < lhsMap.getNumResults(); i++) {
        auto dimExpr = dyn_cast<AffineDimExpr>(lhsMap.getResult(i));
        if (dimExpr && !isReductionDim[dimExpr.getPosition()]) {
          int newPos = oldToNewDim[dimExpr.getPosition()];
          lhsSumsExprs.push_back(
              getAffineDimExpr(newPos, rewriter.getContext()));
        }
      }
      AffineMap lhsSumsMap = AffineMap::get(numParallelDims, 0, lhsSumsExprs,
                                            rewriter.getContext());
      corrIdxLhsSums = addCorrInput(lhsSums, lhsSumsMap);

      AffineMap rhsZpMap = getZpCorrMap(info->rhsZpArgIdx, rhsZpVal);
      corrIdxRhsZp = addCorrInput(rhsZpVal, rhsZpMap);
    }

    int corrIdxRhsSums = 0, corrIdxLhsZp = 0;
    if (hasLhsZp) {
      SmallVector<AffineExpr> rhsSumsExprs;
      for (unsigned i = 0; i < rhsMap.getNumResults(); i++) {
        auto dimExpr = dyn_cast<AffineDimExpr>(rhsMap.getResult(i));
        if (dimExpr && !isReductionDim[dimExpr.getPosition()]) {
          int newPos = oldToNewDim[dimExpr.getPosition()];
          rhsSumsExprs.push_back(
              getAffineDimExpr(newPos, rewriter.getContext()));
        }
      }
      AffineMap rhsSumsMap = AffineMap::get(numParallelDims, 0, rhsSumsExprs,
                                            rewriter.getContext());
      corrIdxRhsSums = addCorrInput(rhsSums, rhsSumsMap);

      AffineMap lhsZpMap = getZpCorrMap(info->lhsZpArgIdx, lhsZpVal);
      corrIdxLhsZp = addCorrInput(lhsZpVal, lhsZpMap);
    }

    int corrIdxKTerm = 0;
    if (hasLhsZp && hasRhsZp) {
      // Compute zp_a * zp_b * K as a scalar-like input.
      // But zp_a and zp_b may be tensors (per-channel/per-block), so we
      // cannot premultiply them. Instead, we pass K as a separate input
      // and compute all three multiplications inside the body.
      AffineMap scalarMap =
          AffineMap::get(numParallelDims, 0, rewriter.getContext());
      corrIdxKTerm = addCorrInput(kSizeVal, scalarMap);
    }

    // Output map
    corrMaps.push_back(corrAccMap);

    // Create the output tensor.
    Value corrInit = tensor::EmptyOp::create(
        builder, tensor::getMixedSizes(builder, loc, accVal),
        accElTy);

    SmallVector<utils::IteratorType> corrIterators(
        numParallelDims, utils::IteratorType::parallel);

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        genericOp, accType, corrIns, ValueRange{corrInit}, corrMaps,
        corrIterators,
        [=](OpBuilder &b, Location loc, ValueRange args) {
          Value result = args[corrIdxMatmul];
          if (hasRhsZp) {
            Value lhsSumsTimesRhsZp =
                arith::MulIOp::create(b, loc, args[corrIdxLhsSums],
                                      args[corrIdxRhsZp]);
            result = arith::SubIOp::create(b, loc, result, lhsSumsTimesRhsZp);
          }
          if (hasLhsZp) {
            Value rhsSumsTimesLhsZp =
                arith::MulIOp::create(b, loc, args[corrIdxRhsSums],
                                      args[corrIdxLhsZp]);
            result = arith::SubIOp::create(b, loc, result, rhsSumsTimesLhsZp);
          }
          if (hasLhsZp && hasRhsZp) {
            Value zpProduct =
                arith::MulIOp::create(b, loc, args[corrIdxLhsZp],
                                      args[corrIdxRhsZp]);
            Value zpProductTimesK =
                arith::MulIOp::create(b, loc, zpProduct, args[corrIdxKTerm]);
            result = arith::AddIOp::create(b, loc, result, zpProductTimesK);
          }
          linalg::YieldOp::create(b, loc, result);
        });

    return success();
  }
};

/// Pass that lowers quantized_matmul to matmul.
class LinalgQuantizedMatmulToMatmulPass final
    : public impl::LinalgQuantizedMatmulToMatmulPassBase<
          LinalgQuantizedMatmulToMatmulPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    RewritePatternSet patterns(context);
    patterns.add<QuantizedMatmulToMatmul>(context);
    patterns.add<GenericQuantizedContractionDecomposition>(
        context, shiftToSignedDomain);
    memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::GlobalOptimization
