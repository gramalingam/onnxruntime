// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/optimizer/pattern_rewriter.h"

namespace onnxruntime::rewrite {

// Opt-in float32, channels-first 1D CausalConvWithState rules. Not registered by default.
Rules MakeMobiusStatefulRules();

}  // namespace onnxruntime::rewrite
