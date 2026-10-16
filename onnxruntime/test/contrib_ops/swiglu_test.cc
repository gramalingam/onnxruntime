// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Reference: out = g * sigmoid(alpha * g) * l.
std::vector<float> ComputeSwiGLUReference(const std::vector<float>& gate,
                                          const std::vector<float>& linear, float alpha) {
  std::vector<float> out(gate.size());
  for (size_t i = 0; i < gate.size(); ++i) {
    const float sig = 1.0f / (1.0f + std::exp(-alpha * gate[i]));
    out[i] = gate[i] * sig * linear[i];
  }
  return out;
}

void RunSwiGLUTest(const std::vector<int64_t>& dims, float alpha, bool use_float16, unsigned seed) {
  int64_t count = 1;
  for (auto d : dims) count *= d;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  std::vector<float> gate(static_cast<size_t>(count)), linear(static_cast<size_t>(count));
  for (auto& v : gate) v = dist(rng);
  for (auto& v : linear) v = dist(rng);

  std::vector<float> expected = ComputeSwiGLUReference(gate, linear, alpha);

  auto run = [&](std::unique_ptr<IExecutionProvider> ep, bool fp16) {
    OpTester tester("SwiGLU", 1, onnxruntime::kMSDomain);
    if (alpha != 1.0f) {
      tester.AddAttribute<float>("alpha", alpha);
    }
    if (fp16) {
      tester.AddInput<MLFloat16>("gate", dims, ToFloat16(gate));
      tester.AddInput<MLFloat16>("linear", dims, ToFloat16(linear));
      tester.AddOutput<MLFloat16>("output", dims, ToFloat16(expected));
      tester.SetOutputTolerance(0.02f);
    } else {
      tester.AddInput<float>("gate", dims, gate);
      tester.AddInput<float>("linear", dims, linear);
      tester.AddOutput<float>("output", dims, expected);
      tester.SetOutputTolerance(1e-4f);
    }
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

TEST(SwiGLUTest, DefaultAlpha2D) {
  RunSwiGLUTest({4, 8}, /*alpha*/ 1.0f, /*use_float16*/ false, 1);
}

TEST(SwiGLUTest, CustomAlpha2D) {
  RunSwiGLUTest({3, 16}, /*alpha*/ 1.702f, /*use_float16*/ false, 2);
}

TEST(SwiGLUTest, LargeMultiPassChunks) {
  // Exceeds the 4096-element per-task chunk size to exercise the parallel path.
  RunSwiGLUTest({8, 5000}, /*alpha*/ 1.0f, /*use_float16*/ false, 3);
}

TEST(SwiGLUTest, Rank3) {
  RunSwiGLUTest({2, 3, 5}, /*alpha*/ 1.0f, /*use_float16*/ false, 4);
}

TEST(SwiGLUTest, Float16) {
  RunSwiGLUTest({4, 32}, /*alpha*/ 1.0f, /*use_float16*/ true, 5);
}

TEST(SwiGLUTest, Float16CustomAlpha) {
  RunSwiGLUTest({4, 32}, /*alpha*/ 1.702f, /*use_float16*/ true, 6);
}

}  // namespace test
}  // namespace onnxruntime
