// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

// Reference implementation.
//   group_indices:   [M, k]
//   bias:            [num_groups, N] or empty
// Per-expert result r[i, j] = input[i] @ weights[g] (+ bias[g]) -> output [M, k, N].
std::vector<float> ComputeGroupedMatMulReference(const std::vector<float>& input,
                                                 const std::vector<float>& weights,
                                                 const std::vector<int64_t>& group_indices,
                                                 const std::vector<float>& bias,
                                                 int64_t M, int64_t k, int64_t K, int64_t N) {
  const bool has_bias = !bias.empty();

  std::vector<float> output(static_cast<size_t>(M * k * N), 0.0f);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < k; ++j) {
      const int64_t sel = i * k + j;
      const int64_t g = group_indices[static_cast<size_t>(sel)];
      const float* in_row = input.data() + i * K;
      const float* w = weights.data() + g * K * N;
      for (int64_t n = 0; n < N; ++n) {
        float acc = has_bias ? bias[static_cast<size_t>(g * N + n)] : 0.0f;
        for (int64_t kk = 0; kk < K; ++kk) {
          acc += in_row[kk] * w[kk * N + n];
        }
        output[static_cast<size_t>(sel * N + n)] = acc;
      }
    }
  }
  return output;
}

// Runs the op on CPU (always) and CUDA (when available), comparing against the reference.
//   group_indices: flattened [M, k].
void RunGroupedMatMulTest(int64_t M, int64_t k, int64_t K, int64_t num_groups, int64_t N,
                          const std::vector<int64_t>& group_indices,
                          bool with_bias, bool use_float16, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> input(static_cast<size_t>(M * K));
  for (auto& v : input) v = dist(rng);
  std::vector<float> weights(static_cast<size_t>(num_groups * K * N));
  for (auto& v : weights) v = dist(rng);
  std::vector<float> bias;
  if (with_bias) {
    bias.resize(static_cast<size_t>(num_groups * N));
    for (auto& v : bias) v = dist(rng);
  }

  std::vector<float> expected =
      ComputeGroupedMatMulReference(input, weights, group_indices, bias, M, k, K, N);

  std::vector<int64_t> input_dims = {M, K};
  std::vector<int64_t> weights_dims = {num_groups, K, N};
  std::vector<int64_t> indices_dims = {M, k};
  std::vector<int64_t> bias_dims = {num_groups, N};
  std::vector<int64_t> output_dims = {M, k, N};

  auto run = [&](std::unique_ptr<IExecutionProvider> ep, bool fp16) {
    OpTester tester("GroupedMatMul", 1, onnxruntime::kMSDomain);
    if (fp16) {
      tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(input));
      tester.AddInput<MLFloat16>("weights", weights_dims, ToFloat16(weights));
      tester.AddInput<int64_t>("group_indices", indices_dims, group_indices);
      if (with_bias) {
        tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias));
      }
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(expected));
      tester.SetOutputTolerance(0.05f);
    } else {
      tester.AddInput<float>("input", input_dims, input);
      tester.AddInput<float>("weights", weights_dims, weights);
      tester.AddInput<int64_t>("group_indices", indices_dims, group_indices);
      if (with_bias) {
        tester.AddInput<float>("bias", bias_dims, bias);
      }
      tester.AddOutput<float>("output", output_dims, expected);
      tester.SetOutputTolerance(1e-3f);
    }
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  };

  run(DefaultCpuExecutionProvider(), use_float16);

#ifdef USE_CUDA
  if (HasCudaEnvironment(600)) {
    run(DefaultCudaExecutionProvider(), use_float16);
  }
#endif
}

}  // namespace

// --- Dense (k == 1) cases: equivalent to the plain grouped MatMul. ---

TEST(GroupedMatMulTest, Dense2D) {
  // 4 tokens, K=3, 2 groups, N=2, k=1.
  std::vector<int64_t> group_indices = {0, 1, 0, 1};
  RunGroupedMatMulTest(/*M*/ 4, /*k*/ 1, /*K*/ 3, /*num_groups*/ 2, /*N*/ 2, group_indices,
                       /*with_bias*/ false, /*use_float16*/ false, 1);
}

TEST(GroupedMatMulTest, Dense2DWithBias) {
  std::vector<int64_t> group_indices = {2, 0, 1, 1, 2, 0};
  RunGroupedMatMulTest(/*M*/ 6, /*k*/ 1, /*K*/ 4, /*num_groups*/ 3, /*N*/ 5, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 2);
}

TEST(GroupedMatMulTest, EmptyGroup) {
  // Group 1 receives no tokens.
  std::vector<int64_t> group_indices = {0, 0, 2, 2};
  RunGroupedMatMulTest(/*M*/ 4, /*k*/ 1, /*K*/ 3, /*num_groups*/ 3, /*N*/ 3, group_indices,
                       /*with_bias*/ false, /*use_float16*/ false, 4);
}

TEST(GroupedMatMulTest, AllSameGroup) {
  // Degenerates to a standard MatMul against weights[1].
  std::vector<int64_t> group_indices = {1, 1, 1, 1, 1};
  RunGroupedMatMulTest(/*M*/ 5, /*k*/ 1, /*K*/ 4, /*num_groups*/ 2, /*N*/ 3, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 5);
}

TEST(GroupedMatMulTest, SingleGroup) {
  // num_groups == 1 is equivalent to MatMul(input, weights[0]).
  std::vector<int64_t> group_indices = {0, 0, 0};
  RunGroupedMatMulTest(/*M*/ 3, /*k*/ 1, /*K*/ 4, /*num_groups*/ 1, /*N*/ 5, group_indices,
                       /*with_bias*/ false, /*use_float16*/ false, 6);
}

// --- Top-k (k > 1) cases: output [M, k, N]. ---

TEST(GroupedMatMulTest, TopKNoBias) {
  // 3 tokens, k=2, each selects 2 experts.
  std::vector<int64_t> group_indices = {0, 2, 1, 3, 2, 0};
  RunGroupedMatMulTest(/*M*/ 3, /*k*/ 2, /*K*/ 4, /*num_groups*/ 4, /*N*/ 5, group_indices,
                       /*with_bias*/ false, /*use_float16*/ false, 11);
}

TEST(GroupedMatMulTest, TopKWithBias) {
  std::vector<int64_t> group_indices = {0, 1, 2, 0, 1, 3};
  RunGroupedMatMulTest(/*M*/ 3, /*k*/ 2, /*K*/ 4, /*num_groups*/ 4, /*N*/ 5, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 10);
}

// --- float16 ---

TEST(GroupedMatMulTest, Float16TopKWithBias) {
  std::vector<int64_t> group_indices = {0, 1, 2, 0, 1, 2};
  RunGroupedMatMulTest(/*M*/ 3, /*k*/ 2, /*K*/ 4, /*num_groups*/ 3, /*N*/ 4, group_indices,
                       /*with_bias*/ true, /*use_float16*/ true, 7);
}

TEST(GroupedMatMulTest, Float16TopKNoBias) {
  std::vector<int64_t> group_indices = {2, 0, 1, 1, 0, 2};
  RunGroupedMatMulTest(/*M*/ 3, /*k*/ 2, /*K*/ 4, /*num_groups*/ 3, /*N*/ 4, group_indices,
                       /*with_bias*/ false, /*use_float16*/ true, 8);
}

// --- Larger random MoE-like workload. ---

TEST(GroupedMatMulTest, LargerRandom) {
  const int64_t M = 64;
  const int64_t k = 2;
  const int64_t num_groups = 8;
  std::mt19937 rng(123);
  std::uniform_int_distribution<int64_t> gdist(0, num_groups - 1);
  std::vector<int64_t> group_indices(static_cast<size_t>(M * k));
  for (auto& g : group_indices) g = gdist(rng);
  RunGroupedMatMulTest(M, k, /*K*/ 32, num_groups, /*N*/ 48, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 9);
}

}  // namespace test
}  // namespace onnxruntime
