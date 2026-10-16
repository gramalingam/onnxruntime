// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Reference: per row, TopK(logits, k) (largest, sorted desc, ties -> smaller index) then
// softmax over the k selected logits.
void ComputeRouterTopKReference(const std::vector<float>& logits, int64_t rows, int64_t E, int64_t k,
                                std::vector<float>& weights, std::vector<int64_t>& indices) {
  weights.resize(static_cast<size_t>(rows * k));
  indices.resize(static_cast<size_t>(rows * k));
  for (int64_t r = 0; r < rows; ++r) {
    const float* row = logits.data() + r * E;
    std::vector<int64_t> order(static_cast<size_t>(E));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + static_cast<ptrdiff_t>(k), order.end(),
                      [&](int64_t a, int64_t b) {
                        if (row[a] != row[b]) return row[a] > row[b];
                        return a < b;
                      });
    const float max_logit = row[order[0]];
    float sum = 0.0f;
    std::vector<float> exps(static_cast<size_t>(k));
    for (int64_t j = 0; j < k; ++j) {
      const float e = std::exp(row[order[static_cast<size_t>(j)]] - max_logit);
      exps[static_cast<size_t>(j)] = e;
      sum += e;
      indices[static_cast<size_t>(r * k + j)] = order[static_cast<size_t>(j)];
    }
    for (int64_t j = 0; j < k; ++j) {
      weights[static_cast<size_t>(r * k + j)] = exps[static_cast<size_t>(j)] / sum;
    }
  }
}

void RunRouterTopKTest(int64_t rows, int64_t E, int64_t k, bool use_float16, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> logits(static_cast<size_t>(rows * E));
  for (auto& v : logits) v = dist(rng);

  // The kernel sees fp16-rounded logits; round the reference inputs the same way so that
  // top-k tie-breaking (by value then index) matches exactly.
  if (use_float16) {
    std::vector<MLFloat16> h = ToFloat16(logits);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = h[i].ToFloat();
  }

  std::vector<float> expected_w;
  std::vector<int64_t> expected_i;
  ComputeRouterTopKReference(logits, rows, E, k, expected_w, expected_i);

  std::vector<int64_t> in_dims = {rows, E};
  std::vector<int64_t> out_dims = {rows, k};

  auto run = [&](std::unique_ptr<IExecutionProvider> ep, bool fp16) {
    OpTester tester("RouterTopK", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("k", k);
    if (fp16) {
      tester.AddInput<MLFloat16>("logits", in_dims, ToFloat16(logits));
      tester.AddOutput<MLFloat16>("weights", out_dims, ToFloat16(expected_w));
      tester.SetOutputTolerance(0.02f);
    } else {
      tester.AddInput<float>("logits", in_dims, logits);
      tester.AddOutput<float>("weights", out_dims, expected_w);
      tester.SetOutputTolerance(1e-4f);
    }
    tester.AddOutput<int64_t>("indices", out_dims, expected_i);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  };

  run(DefaultCpuExecutionProvider(), use_float16);

#ifdef USE_CUDA
  if (HasCudaEnvironment(0)) {
    run(DefaultCudaExecutionProvider(), use_float16);
  }
#endif
}

}  // namespace

TEST(RouterTopKTest, Mixtral) {
  RunRouterTopKTest(/*rows*/ 16, /*E*/ 8, /*k*/ 2, /*use_float16*/ false, 1);
}

TEST(RouterTopKTest, DeepSeek) {
  RunRouterTopKTest(/*rows*/ 32, /*E*/ 64, /*k*/ 6, /*use_float16*/ false, 2);
}

TEST(RouterTopKTest, TopOne) {
  RunRouterTopKTest(/*rows*/ 10, /*E*/ 128, /*k*/ 1, /*use_float16*/ false, 3);
}

TEST(RouterTopKTest, KEqualsE) {
  // k == num_experts: selects (and softmaxes over) all experts.
  RunRouterTopKTest(/*rows*/ 8, /*E*/ 5, /*k*/ 5, /*use_float16*/ false, 4);
}

TEST(RouterTopKTest, Float16) {
  RunRouterTopKTest(/*rows*/ 16, /*E*/ 32, /*k*/ 4, /*use_float16*/ true, 5);
}

}  // namespace test
}  // namespace onnxruntime
