// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Converts the PT2E quantization ops that appear in exported PyTorch graphs
// into iree_linalg_ext.quantize_affine and dequantize_affine.
//
// PT2E spells quantization as `quantized_decomposed.{,de}quantize_per_tensor`
// and `..._per_channel`, carrying the scale and zero point as operands and the
// storage range as `quant_min`/`quant_max` operands. That maps onto the affine
// quantization ops directly: per-tensor parameters become scalar operands
// indexed by a zero-result map, and per-channel parameters become rank one
// operands indexed by the single dimension `axis` names. Signedness has to be
// read from the torch tensor types, which state it explicitly, because the
// builtin types they convert to are signless.
//
// Running ahead of ConvertTorchToLinalg is what keeps these ops from being
// lowered to elementwise generics first. The rewrite that turns a dequantized
// contraction into integer arithmetic matches the named ops, so reaching it
// requires the quantization to still be named by the time global optimization
// runs. Anything this pass declines is left to ConvertTorchToLinalg, which
// lowers all four ops to generics.
//===----------------------------------------------------------------------===//

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTORCHQUANTIZATIONTOLINALGEXTPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace Torch = torch::Torch;
namespace TorchConversion = torch::TorchConversion;

namespace {

/// Everything the conversion needs, resolved from the torch types and the
/// constant operands. Matching produces this without creating any IR, because a
/// pattern that builds ops and then reports a match failure leaves them behind.
struct AffineQuantizationPlan {
  /// The torch operands, converted once matching has succeeded.
  /// `torchZeroPoint` is null for symmetric quantization.
  Value torchInput;
  Value torchScale;
  Value torchZeroPoint;
  /// The signless builtin types the torch tensors convert to. `scaleType` is
  /// null when the quantization parameters are scalars rather than tensors, in
  /// which case they come from `!torch.float` and `!torch.int` operands.
  RankedTensorType inputType;
  RankedTensorType resultType;
  RankedTensorType scaleType;
  RankedTensorType zeroPointType;
  /// Maps the index space of the quantized value to that of the quantization
  /// parameters: zero results for per-tensor, one for per-channel.
  AffineMap qparamMap;
  int64_t quantMin;
  int64_t quantMax;
  bool storageUnsigned;
};

/// Returns true for an unsigned integer torch dtype. Signedness is only present
/// on the torch types; the builtin types they convert to are signless.
static bool isUnsignedDtype(Type dtype) {
  auto intType = dyn_cast<IntegerType>(dtype);
  return intType && intType.isUnsigned();
}

/// The builtin tensor type `type` converts to, with integer element types made
/// signless. `toBuiltinTensor` keeps the torch signedness, which the rest of the
/// compiler does not carry, so it is dropped here the same way the Torch to
/// Linalg backend type conversion does. Both `to_builtin_tensor` and
/// `from_builtin_tensor` allow signedness to differ across the boundary.
static RankedTensorType
getSignlessBuiltinTensorType(Torch::ValueTensorType type) {
  auto builtinType = dyn_cast_or_null<RankedTensorType>(type.toBuiltinTensor());
  if (!builtinType) {
    return nullptr;
  }
  Type dtype = type.getDtype();
  if (!dtype.isInteger()) {
    return builtinType;
  }
  return cast<RankedTensorType>(builtinType.clone(
      IntegerType::get(type.getContext(), dtype.getIntOrFloatBitWidth(),
                       IntegerType::Signless)));
}

/// Returns the torch tensor type of `value` when it is ranked with a known
/// dtype, and null otherwise.
static Torch::ValueTensorType getRankedTypedTensorType(Value value) {
  auto type = dyn_cast<Torch::ValueTensorType>(value.getType());
  if (!type || !type.hasSizes() || !type.hasDtype()) {
    return nullptr;
  }
  return type;
}

/// Resolves the parts shared by all four ops. `storageIsResult` selects which
/// side of the op holds the quantized value, i.e. quantize versus dequantize.
static FailureOr<AffineQuantizationPlan>
matchCommon(PatternRewriter &rewriter, Operation *op, Value torchInput,
            Value torchQuantMin, Value torchQuantMax, bool storageIsResult) {
  Torch::ValueTensorType inputType = getRankedTypedTensorType(torchInput);
  Torch::ValueTensorType resultType =
      getRankedTypedTensorType(op->getResult(0));
  if (!inputType || !resultType) {
    return rewriter.notifyMatchFailure(
        op, "expected ranked input and result with known dtypes");
  }

  AffineQuantizationPlan plan;
  if (!matchPattern(torchQuantMin,
                    Torch::m_TorchConstantInt(&plan.quantMin)) ||
      !matchPattern(torchQuantMax,
                    Torch::m_TorchConstantInt(&plan.quantMax))) {
    return rewriter.notifyMatchFailure(
        op, "expected constant quant_min and quant_max");
  }

  Type storageDtype =
      storageIsResult ? resultType.getDtype() : inputType.getDtype();
  Type realDtype =
      storageIsResult ? inputType.getDtype() : resultType.getDtype();
  if (!isa<IntegerType>(storageDtype) || !isa<FloatType>(realDtype)) {
    return rewriter.notifyMatchFailure(
        op, "expected an integer storage type and a float real type");
  }
  plan.storageUnsigned = isUnsignedDtype(storageDtype);

  plan.inputType = getSignlessBuiltinTensorType(inputType);
  plan.resultType = getSignlessBuiltinTensorType(resultType);
  if (!plan.inputType || !plan.resultType) {
    return rewriter.notifyMatchFailure(op, "unsupported element type");
  }
  plan.torchInput = torchInput;
  return plan;
}

/// Resolves per-tensor quantization parameters, which stay scalars.
static void matchPerTensorQParams(PatternRewriter &rewriter, Value torchScale,
                                  Value torchZeroPoint,
                                  AffineQuantizationPlan &plan) {
  plan.torchScale = torchScale;
  plan.torchZeroPoint = torchZeroPoint;
  plan.qparamMap = AffineMap::get(plan.inputType.getRank(), /*symbolCount=*/0,
                                  rewriter.getContext());
}

/// Resolves per-channel quantization parameters, which are rank one tensors
/// indexed by `axis`. A None zero point means symmetric quantization.
static LogicalResult matchPerChannelQParams(PatternRewriter &rewriter,
                                            Operation *op, Value torchScales,
                                            Value torchZeroPoints,
                                            Value torchAxis,
                                            AffineQuantizationPlan &plan) {
  int64_t rank = plan.inputType.getRank();
  int64_t axis;
  if (!matchPattern(torchAxis, Torch::m_TorchConstantInt(&axis))) {
    return rewriter.notifyMatchFailure(op, "expected a constant axis");
  }
  if (axis < 0) {
    axis += rank;
  }
  if (axis < 0 || axis >= rank) {
    return rewriter.notifyMatchFailure(op, "axis out of range");
  }

  Torch::ValueTensorType scalesType = getRankedTypedTensorType(torchScales);
  if (!scalesType || scalesType.getSizes().size() != 1) {
    return rewriter.notifyMatchFailure(op, "expected rank one scales");
  }
  if (!isa<FloatType>(scalesType.getDtype())) {
    return rewriter.notifyMatchFailure(
        op, "expected a floating point scales element type");
  }
  // The scales element type is passed through rather than converted to the
  // real type: it is what the quantization op does its arithmetic in, and PT2E
  // emits f32 scales for an f16 or bf16 value precisely because it computes in
  // f32.
  plan.scaleType = getSignlessBuiltinTensorType(scalesType);
  plan.torchScale = torchScales;

  if (!isa<Torch::NoneType>(torchZeroPoints.getType())) {
    Torch::ValueTensorType zeroPointsType =
        getRankedTypedTensorType(torchZeroPoints);
    if (!zeroPointsType || zeroPointsType.getSizes().size() != 1) {
      return rewriter.notifyMatchFailure(op, "expected rank one zero points");
    }
    if (!isa<IntegerType>(zeroPointsType.getDtype())) {
      return rewriter.notifyMatchFailure(
          op, "expected an integer zero point element type");
    }
    plan.zeroPointType = getSignlessBuiltinTensorType(zeroPointsType);
    plan.torchZeroPoint = torchZeroPoints;
  }

  plan.qparamMap =
      AffineMap::get(rank, /*symbolCount=*/0, rewriter.getAffineDimExpr(axis));
  return success();
}

/// Builds the operands of an affine quantization op from `plan`, crossing from
/// the torch types to the builtin ones. `zeroPoint` is left null for symmetric
/// quantization.
static void buildOperands(OpBuilder &b, Location loc,
                          const AffineQuantizationPlan &plan, Value &input,
                          Value &scale, Value &zeroPoint, Value &init,
                          SmallVector<AffineMap> &indexingMaps) {
  input = TorchConversion::ToBuiltinTensorOp::create(b, loc, plan.inputType,
                                                     plan.torchInput);
  if (plan.scaleType) {
    scale = TorchConversion::ToBuiltinTensorOp::create(b, loc, plan.scaleType,
                                                       plan.torchScale);
  } else {
    // A `!torch.float` is f64, but the PT2E reference decompositions compute in
    // f32 whatever the real type is, so that is the width to keep.
    Value f64 = TorchConversion::ToF64Op::create(b, loc, plan.torchScale);
    scale = convertScalarToDtype(b, loc, f64, b.getF32Type(),
                                 /*isUnsignedCast=*/false);
  }
  if (plan.torchZeroPoint) {
    if (plan.zeroPointType) {
      zeroPoint = TorchConversion::ToBuiltinTensorOp::create(
          b, loc, plan.zeroPointType, plan.torchZeroPoint);
    } else {
      // The affine quantization ops accept a zero point wider than the storage
      // type and narrow it when lowering, so the natural torch width is fine.
      zeroPoint = TorchConversion::ToI64Op::create(b, loc, plan.torchZeroPoint);
    }
  }

  init = tensor::EmptyOp::create(b, loc, tensor::getMixedSizes(b, loc, input),
                                 plan.resultType.getElementType());

  AffineMap identity = b.getMultiDimIdentityMap(plan.inputType.getRank());
  indexingMaps = {identity, plan.qparamMap};
  if (zeroPoint) {
    indexingMaps.push_back(plan.qparamMap);
  }
  indexingMaps.push_back(identity);
}

/// Replaces `op` with `quantize_affine`.
static void replaceWithQuantize(PatternRewriter &rewriter, Operation *op,
                                const AffineQuantizationPlan &plan) {
  Location loc = op->getLoc();
  Value input, scale, zeroPoint, init;
  SmallVector<AffineMap> indexingMaps;
  buildOperands(rewriter, loc, plan, input, scale, zeroPoint, init,
                indexingMaps);
  Value quantized = IREE::LinalgExt::QuantizeAffineOp::create(
                        rewriter, loc, plan.resultType, input, scale, zeroPoint,
                        init, rewriter.getAffineMapArrayAttr(indexingMaps),
                        rewriter.getI64IntegerAttr(plan.quantMin),
                        rewriter.getI64IntegerAttr(plan.quantMax),
                        /*storage_unsigned=*/plan.storageUnsigned
                            ? rewriter.getUnitAttr()
                            : nullptr,
                        /*zp_unsigned=*/nullptr)
                        ->getResult(0);
  rewriter.replaceOpWithNewOp<TorchConversion::FromBuiltinTensorOp>(
      op, op->getResult(0).getType(), quantized);
}

static void replaceWithDequantize(PatternRewriter &rewriter, Operation *op,
                                  const AffineQuantizationPlan &plan) {
  Location loc = op->getLoc();
  Value input, scale, zeroPoint, init;
  SmallVector<AffineMap> indexingMaps;
  buildOperands(rewriter, loc, plan, input, scale, zeroPoint, init,
                indexingMaps);
  Value real = IREE::LinalgExt::DequantizeAffineOp::create(
                   rewriter, loc, plan.resultType, input, scale, zeroPoint,
                   init, rewriter.getAffineMapArrayAttr(indexingMaps),
                   rewriter.getI64IntegerAttr(plan.quantMin),
                   rewriter.getI64IntegerAttr(plan.quantMax),
                   /*input_unsigned=*/plan.storageUnsigned
                       ? rewriter.getUnitAttr()
                       : nullptr,
                   /*zp_unsigned=*/nullptr)
                   ->getResult(0);
  rewriter.replaceOpWithNewOp<TorchConversion::FromBuiltinTensorOp>(
      op, op->getResult(0).getType(), real);
}

struct QuantizePerTensorConversion
    : OpRewritePattern<Torch::QuantizedDecomposedQuantizePerTensorOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Torch::QuantizedDecomposedQuantizePerTensorOp op,
                  PatternRewriter &rewriter) const override {
    FailureOr<AffineQuantizationPlan> plan =
        matchCommon(rewriter, op, op.getInput(), op.getQuantMin(),
                    op.getQuantMax(), /*storageIsResult=*/true);
    if (failed(plan)) {
      return failure();
    }
    matchPerTensorQParams(rewriter, op.getScale(), op.getZeroPoint(), *plan);
    replaceWithQuantize(rewriter, op, *plan);
    return success();
  }
};

struct DequantizePerTensorConversion
    : OpRewritePattern<Torch::QuantizedDecomposedDequantizePerTensorOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Torch::QuantizedDecomposedDequantizePerTensorOp op,
                  PatternRewriter &rewriter) const override {
    FailureOr<AffineQuantizationPlan> plan =
        matchCommon(rewriter, op, op.getInput(), op.getQuantMin(),
                    op.getQuantMax(), /*storageIsResult=*/false);
    if (failed(plan)) {
      return failure();
    }
    matchPerTensorQParams(rewriter, op.getScale(), op.getZeroPoint(), *plan);
    replaceWithDequantize(rewriter, op, *plan);
    return success();
  }
};

struct QuantizePerChannelConversion
    : OpRewritePattern<Torch::QuantizedDecomposedQuantizePerChannelOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Torch::QuantizedDecomposedQuantizePerChannelOp op,
                  PatternRewriter &rewriter) const override {
    FailureOr<AffineQuantizationPlan> plan =
        matchCommon(rewriter, op, op.getInput(), op.getQuantMin(),
                    op.getQuantMax(), /*storageIsResult=*/true);
    if (failed(plan) ||
        failed(matchPerChannelQParams(rewriter, op, op.getScales(),
                                      op.getZeroPoints(), op.getAxis(),
                                      *plan))) {
      return failure();
    }
    replaceWithQuantize(rewriter, op, *plan);
    return success();
  }
};

struct DequantizePerChannelConversion
    : OpRewritePattern<Torch::QuantizedDecomposedDequantizePerChannelOp> {
  using Base::Base;
  LogicalResult
  matchAndRewrite(Torch::QuantizedDecomposedDequantizePerChannelOp op,
                  PatternRewriter &rewriter) const override {
    FailureOr<AffineQuantizationPlan> plan =
        matchCommon(rewriter, op, op.getInput(), op.getQuantMin(),
                    op.getQuantMax(), /*storageIsResult=*/false);
    if (failed(plan) ||
        failed(matchPerChannelQParams(rewriter, op, op.getScales(),
                                      op.getZeroPoints(), op.getAxis(),
                                      *plan))) {
      return failure();
    }
    replaceWithDequantize(rewriter, op, *plan);
    return success();
  }
};

class ConvertTorchQuantizationToLinalgExtPass final
    : public impl::ConvertTorchQuantizationToLinalgExtPassBase<
          ConvertTorchQuantizationToLinalgExtPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect,
                    TorchConversion::TorchConversionDialect,
                    tensor::TensorDialect, arith::ArithDialect,
                    func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<QuantizePerTensorConversion, DequantizePerTensorConversion,
                 QuantizePerChannelConversion, DequantizePerChannelConversion>(
        context);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler::TorchInput
