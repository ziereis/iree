// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/LoopDimDivisibilityAnalysis.h"
#include <numeric>
#include "llvm/Support/Debug.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"

#define DEBUG_TYPE "iree-codegen-loop-dim-divisibility-analysis"

namespace mlir::iree_compiler {

LoopDimDivisibilityAnalysis::LoopDimDivisibilityAnalysis(
    const TensorDynamicDimAnalysis &tensorDimAnalysis, Operation *rootOperation)
    : tensorDimAnalysis(tensorDimAnalysis), rootOperation(rootOperation) {}

LogicalResult LoopDimDivisibilityAnalysis::run() {
  rootOperation->walk([&](IndexingMapOpInterface op) {
    // Require all indexing maps to be projected permutations. In theory we
    // could still analyze loop dimensions that are only accessed through
    // projected permutation results and poison (set to no-info) only those
    // loop dimensions referenced by non-projected expressions. For now the
    // simpler per-op gate is sufficient since ops in IREE codegen have
    // projected permutation maps by the time this analysis runs.
    SmallVector<AffineMap> maps = op.getIndexingMapsArray();
    bool allProjectedPermutation = llvm::all_of(
        maps, [](AffineMap map) { return map.isProjectedPermutation(); });
    if (!allProjectedPermutation) {
      return;
    }

    for (OpOperand &operand : op->getOpOperands()) {
      auto tensorType = dyn_cast<RankedTensorType>(operand.get().getType());
      if (!tensorType) {
        continue;
      }

      AffineMap map = op.getMatchingIndexingMap(&operand);
      for (auto [tensorDim, expr] : llvm::enumerate(map.getResults())) {
        auto dimExpr = dyn_cast<AffineDimExpr>(expr);
        if (!dimExpr) {
          continue;
        }

        if (!tensorType.isDynamicDim(tensorDim)) {
          continue;
        }

        std::optional<IREE::Util::ConstantIntDivisibility> divInfo =
            tensorDimAnalysis.getDivisibilityInfo(operand.get(), tensorDim);
        if (!divInfo) {
          continue;
        }

        if (divInfo->udiv() <= 1 && divInfo->sdiv() <= 1) {
          continue;
        }

        unsigned loopDim = dimExpr.getPosition();
        auto key = std::make_pair(op.getOperation(), loopDim);
        auto it = loopDimDivisibility.find(key);
        if (it != loopDimDivisibility.end()) {
          it->second = IREE::Util::ConstantIntDivisibility(
              std::lcm(it->second.udiv(), divInfo->udiv()),
              std::lcm(it->second.sdiv(), divInfo->sdiv()));
        } else {
          loopDimDivisibility[key] = *divInfo;
        }
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Loop dim divisibility for op: ";
      op->print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n";
      for (auto &[key, divInfo] : loopDimDivisibility) {
        if (key.first != op.getOperation()) {
          continue;
        }
        llvm::dbgs() << "\tLoop dim " << key.second << " : " << divInfo << "\n";
      }
    });
  });
  return success();
}

std::optional<IREE::Util::ConstantIntDivisibility>
LoopDimDivisibilityAnalysis::getLoopDimDivisibility(
    Operation *op, unsigned loopDimIndex) const {
  auto it = loopDimDivisibility.find({op, loopDimIndex});
  if (it == loopDimDivisibility.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace mlir::iree_compiler
