// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Driver that runs the toy FeedForward "C++ model" (see feed_forward_model.h).
//
// It builds a session over feed_forward.onnx (a single FeedForward custom-op
// node), registers the C++ operator, runs it, and verifies the output against a
// plain-C++ reference implementation of the same feed-forward computation.
//
// Generate the model first:  python generate_model.py

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "feed_forward_model.h"
#include "onnxruntime_cxx_api.h"

namespace {

constexpr int64_t kK = 4;  // input features
constexpr int64_t kH = 8;  // hidden features
constexpr int64_t kN = 3;  // output features

// Deterministic weights, identical to the formulas in generate_model.py.
float W1(int64_t k, int64_t j) { return static_cast<float>(((k * kH + j) % 7 - 3) * 0.1); }
float B1(int64_t j) { return static_cast<float>(0.01 * j); }
float W2(int64_t h, int64_t n) { return static_cast<float>(((h * kN + n) % 5 - 2) * 0.1); }
float B2(int64_t n) { return static_cast<float>(0.02 * n); }

// Reference: Y = Relu(X @ W1 + b1) @ W2 + b2
std::vector<float> ReferenceFeedForward(const std::vector<float>& x, int64_t m) {
  std::vector<float> hidden(static_cast<size_t>(m * kH), 0.0f);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < kH; ++j) {
      float acc = B1(j);
      for (int64_t k = 0; k < kK; ++k) {
        acc += x[i * kK + k] * W1(k, j);
      }
      hidden[i * kH + j] = std::max(0.0f, acc);
    }
  }

  std::vector<float> y(static_cast<size_t>(m * kN), 0.0f);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t n = 0; n < kN; ++n) {
      float acc = B2(n);
      for (int64_t h = 0; h < kH; ++h) {
        acc += hidden[i * kH + h] * W2(h, n);
      }
      y[i * kN + n] = acc;
    }
  }
  return y;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "feed_forward_model");
    std::cout << "ONNX Runtime version: " << Ort::GetVersionString() << "\n";

    // Register the C++ FeedForward operator in the "com.example" domain.
    models::FeedForwardOp feed_forward_op{};
    Ort::CustomOpDomain domain{"com.example"};
    domain.Add(&feed_forward_op);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    // The custom op internally creates ORT kernels; keep graph optimizations
    // minimal so the single custom node is left intact.
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    session_options.Add(domain);

    const std::filesystem::path model_path = (argc > 1) ? argv[1] : "feed_forward.onnx";
    std::cout << "Loading model: " << model_path.string() << "\n";
    Ort::Session session(env, model_path.native().c_str(), session_options);

    // Prepare input X of shape [M, K].
    constexpr int64_t m = 2;
    std::array<int64_t, 2> x_shape = {m, kK};
    std::vector<float> x = {1.0f, -2.0f, 0.5f, 3.0f,
                            -1.0f, 0.25f, 2.0f, -0.5f};

    auto cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value x_tensor = Ort::Value::CreateTensor<float>(
        cpu, x.data(), x.size(), x_shape.data(), x_shape.size());

    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};

    std::cout << "Running FeedForward custom op...\n";
    auto outputs = session.Run(Ort::RunOptions{}, input_names, &x_tensor, 1, output_names, 1);

    if (outputs.empty() || !outputs[0].IsTensor()) {
      throw std::runtime_error("expected one tensor output");
    }
    const float* y = outputs[0].GetTensorData<float>();
    auto y_info = outputs[0].GetTensorTypeAndShapeInfo();
    const size_t y_count = y_info.GetElementCount();

    std::vector<float> expected = ReferenceFeedForward(x, m);
    if (y_count != expected.size()) {
      throw std::runtime_error("output element count mismatch");
    }

    std::cout << "\nY = FeedForward(X):\n";
    bool correct = true;
    for (int64_t i = 0; i < m; ++i) {
      std::cout << "  [";
      for (int64_t n = 0; n < kN; ++n) {
        const float got = y[i * kN + n];
        std::cout << (n ? ", " : "") << got;
        if (std::abs(got - expected[i * kN + n]) > 1e-4f) {
          correct = false;
        }
      }
      std::cout << "]\n";
    }

    std::cout << "\nResult: " << (correct ? "PASS" : "FAIL") << "\n";
    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    return EXIT_FAILURE;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }
}
