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

// Reference implementation: for each token, output[i] = input[i] @ weights[g] (+ bias[g]).
std::vector<float> ComputeGroupedMatMulReference(const std::vector<float>& input,
                                                 const std::vector<float>& weights,
                                                 const std::vector<int64_t>& group_indices,
                                                 const std::vector<float>& bias,
                                                 int64_t num_tokens, int64_t K, int64_t N) {
  std::vector<float> output(static_cast<size_t>(num_tokens * N), 0.0f);
  const bool has_bias = !bias.empty();
  for (int64_t i = 0; i < num_tokens; ++i) {
    const int64_t g = group_indices[static_cast<size_t>(i)];
    const float* in_row = input.data() + i * K;
    const float* w = weights.data() + g * K * N;
    float* out_row = output.data() + i * N;
    for (int64_t n = 0; n < N; ++n) {
      float acc = has_bias ? bias[static_cast<size_t>(g * N + n)] : 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        acc += in_row[k] * w[k * N + n];
      }
      out_row[n] = acc;
    }
  }
  return output;
}

// Runs the op on CPU (always) and CUDA (when available), comparing against the reference.
void RunGroupedMatMulTest(const std::vector<int64_t>& input_dims,  // {..., K}
                          int64_t num_groups, int64_t N,
                          const std::vector<int64_t>& group_indices,
                          bool with_bias, bool use_float16, unsigned seed) {
  int64_t K = input_dims.back();
  int64_t num_tokens = 1;
  for (size_t i = 0; i + 1 < input_dims.size(); ++i) {
    num_tokens *= input_dims[i];
  }

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> input(static_cast<size_t>(num_tokens * K));
  for (auto& v : input) v = dist(rng);
  std::vector<float> weights(static_cast<size_t>(num_groups * K * N));
  for (auto& v : weights) v = dist(rng);
  std::vector<float> bias;
  if (with_bias) {
    bias.resize(static_cast<size_t>(num_groups * N));
    for (auto& v : bias) v = dist(rng);
  }

  std::vector<int64_t> output_dims(input_dims);
  output_dims.back() = N;

  std::vector<float> expected =
      ComputeGroupedMatMulReference(input, weights, group_indices, bias, num_tokens, K, N);

  std::vector<int64_t> weights_dims = {num_groups, K, N};
  std::vector<int64_t> indices_dims(input_dims.begin(), input_dims.end() - 1);
  std::vector<int64_t> bias_dims = {num_groups, N};

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

TEST(GroupedMatMulTest, Basic2D) {
  // 4 tokens, K=3, 2 groups, N=2.
  std::vector<int64_t> input_dims = {4, 3};
  std::vector<int64_t> group_indices = {0, 1, 0, 1};
  RunGroupedMatMulTest(input_dims, /*num_groups*/ 2, /*N*/ 2, group_indices,
                       /*with_bias*/ false, /*use_float16*/ false, 1);
}

TEST(GroupedMatMulTest, Basic2DWithBias) {
  std::vector<int64_t> input_dims = {6, 4};
  std::vector<int64_t> group_indices = {2, 0, 1, 1, 2, 0};
  RunGroupedMatMulTest(input_dims, /*num_groups*/ 3, /*N*/ 5, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 2);
}

TEST(GroupedMatMulTest, Batched3D) {
  // [B=2, M=3, K=4] with 3 groups, N=6.
  std::vector<int64_t> input_dims = {2, 3, 4};
  std::vector<int64_t> group_indices = {0, 1, 2, 2, 1, 0};
  RunGroupedMatMulTest(input_dims, /*num_groups*/ 3, /*N*/ 6, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 3);
}

TEST(GroupedMatMulTest, EmptyGroup) {
  // Group 1 receives no tokens.
  std::vector<int64_t> input_dims = {4, 3};
  std::vector<int64_t> group_indices = {0, 0, 2, 2};
  RunGroupedMatMulTest(input_dims, /*num_groups*/ 3, /*N*/ 3, group_indices,
                       /*with_bias*/ false, /*use_float16*/ false, 4);
}

TEST(GroupedMatMulTest, AllSameGroup) {
  // Degenerates to a standard batched MatMul against weights[1].
  std::vector<int64_t> input_dims = {5, 4};
  std::vector<int64_t> group_indices = {1, 1, 1, 1, 1};
  RunGroupedMatMulTest(input_dims, /*num_groups*/ 2, /*N*/ 3, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 5);
}

TEST(GroupedMatMulTest, SingleGroup) {
  // num_groups == 1 is equivalent to MatMul(input, weights[0]).
  std::vector<int64_t> input_dims = {3, 4};
  std::vector<int64_t> group_indices = {0, 0, 0};
  RunGroupedMatMulTest(input_dims, /*num_groups*/ 1, /*N*/ 5, group_indices,
                       /*with_bias*/ false, /*use_float16*/ false, 6);
}

TEST(GroupedMatMulTest, Float16Basic) {
  std::vector<int64_t> input_dims = {6, 4};
  std::vector<int64_t> group_indices = {0, 1, 2, 0, 1, 2};
  RunGroupedMatMulTest(input_dims, /*num_groups*/ 3, /*N*/ 4, group_indices,
                       /*with_bias*/ true, /*use_float16*/ true, 7);
}

TEST(GroupedMatMulTest, LargerRandom) {
  // Larger sizes to exercise the per-group GEMM path.
  const int64_t num_tokens = 64;
  const int64_t num_groups = 8;
  std::vector<int64_t> input_dims = {num_tokens, 32};
  std::mt19937 rng(123);
  std::uniform_int_distribution<int64_t> gdist(0, num_groups - 1);
  std::vector<int64_t> group_indices(static_cast<size_t>(num_tokens));
  for (auto& g : group_indices) g = gdist(rng);
  RunGroupedMatMulTest(input_dims, num_groups, /*N*/ 48, group_indices,
                       /*with_bias*/ true, /*use_float16*/ false, 8);
}

}  // namespace test
}  // namespace onnxruntime
