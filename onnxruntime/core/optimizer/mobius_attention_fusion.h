// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/optimizer/pattern_rewriter.h"

namespace onnxruntime {
namespace rewrite {

// Opt-in FP32 Mobius attention inverses; never registered as default optimizers.
// PackedMHA has no CPU kernel. Enable its single-subsequence rule only for CUDA deployment.
Rules MakeMobiusAttentionRules(bool include_cuda_packed_attention = false);

}  // namespace rewrite
}  // namespace onnxruntime
