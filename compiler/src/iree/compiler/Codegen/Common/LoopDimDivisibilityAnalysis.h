// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_COMMON_LOOPDIM_DIVISIBILITY_ANALYSIS_H_
#define IREE_COMPILER_CODEGEN_COMMON_LOOPDIM_DIVISIBILITY_ANALYSIS_H_

#include "iree/compiler/Codegen/Common/TensorDynamicDimAnalysis.h"

namespace mlir::iree_compiler {

/// Analysis that computes per-loop-dimension divisibility information for
/// operations implementing IndexingMapOpInterface.
///
/// Builds on TensorDynamicDimAnalysis by reconciling per-tensor-dim
/// divisibility into per-loop-dim divisibility using indexing maps.
/// For each loop dimension, the divisibility is the LCM across all
/// operand tensor dimensions that map to it, since all operands sharing
/// a loop dimension must have the same runtime size (both constraints
/// hold simultaneously).
///
/// Only ops whose indexing maps are all projected permutations are analyzed.
/// Ops with any non-projected map are skipped entirely to avoid
/// partial/inconsistent loop-dim state.
class LoopDimDivisibilityAnalysis {
public:
  explicit LoopDimDivisibilityAnalysis(
      const TensorDynamicDimAnalysis &tensorDimAnalysis,
      Operation *rootOperation);

  LogicalResult run();

  /// Get the divisibility info for a loop dimension of an op.
  /// Returns std::nullopt if no divisibility info is known.
  std::optional<IREE::Util::ConstantIntDivisibility>
  getLoopDimDivisibility(Operation *op, unsigned loopDimIndex) const;

private:
  const TensorDynamicDimAnalysis &tensorDimAnalysis;
  Operation *rootOperation;

  DenseMap<std::pair<Operation *, unsigned>,
           IREE::Util::ConstantIntDivisibility>
      loopDimDivisibility;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_LOOPDIM_DIVISIBILITY_ANALYSIS_H_
