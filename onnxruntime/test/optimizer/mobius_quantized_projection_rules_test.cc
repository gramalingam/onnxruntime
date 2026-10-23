// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/pattern_rewriter.h"

#include <cmath>
#include <limits>
#include <sstream>

#include "core/graph/model.h"
#include "core/optimizer/initializer.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
#ifndef DISABLE_CONTRIB_OPS
namespace {
using namespace rewrite;

enum class Mutation {
  None,
  Mask,
  Shift,
  NibbleOrder,
  Direction,
  Transpose,
  Midpoint,
  ZeroMask,
  FloatZero,
  TrimK,
  GroupIndex,
  SharedIntermediate,
  Fp4Table,
  Fp4NegativeZero,
  Fp8Table,
  Fp8NaNEntry,
  NonfiniteScale,
  GlobalVector,
  GlobalNonfinite,
  DynamicWeight,
  ShapeStart,
};

struct FixtureOptions {
  bool packed_zero = false;
  bool bias = false;
  int64_t block = 16;
  int64_t blocks = 3;
  int64_t n = 5;
  int rank = 2;
  bool scalar_as_vector = false;
  Mutation mutation = Mutation::None;
};

// Faithful port of Mobius d69efe03a9f4c1ada744bc31dd01077a710b3c39,
// src/mobius/functions/{matmul_nbits,matmul_block_quantized_fp4_weight}.py.
// Graph::Resolve lifts Constant(value=tensor) to these same initializers.
struct ProjectionBody {
  ModelTestBuilder& b;
  Mutation mutation;

  NodeArg* Op(const char* name, std::vector<NodeArg*> inputs) {
    auto* result = b.MakeIntermediate();
    b.AddNode(name, inputs, {result});
    return result;
  }

  NodeArg* Integers(std::vector<int64_t> values) {
    return b.MakeInitializer<int64_t>({static_cast<int64_t>(values.size())}, values);
  }

  NodeArg* Reshape(NodeArg* input, std::vector<int64_t> shape) {
    return Op("Reshape", {input, Integers(std::move(shape))});
  }

  NodeArg* Cast(NodeArg* input, int64_t to) {
    auto* result = b.MakeIntermediate();
    b.AddNode("Cast", {input}, {result}).AddAttribute("to", to);
    return result;
  }

  NodeArg* Concat(NodeArg* low, NodeArg* high, int64_t axis) {
    auto* result = b.MakeIntermediate();
    b.AddNode("Concat", {low, high}, {result}).AddAttribute("axis", axis);
    return result;
  }

  NodeArg* Nibbles(NodeArg* packed, int64_t axis, bool zero = false) {
    const uint8_t mask = mutation == Mutation::Mask || (zero && mutation == Mutation::ZeroMask) ? 7 : 15;
    auto* low = Op("BitwiseAnd", {packed, b.MakeScalarInitializer<uint8_t>(mask)});
    auto* high = b.MakeIntermediate();
    b.AddNode("BitShift", {packed, b.MakeScalarInitializer<uint8_t>(mutation == Mutation::Shift ? 3 : 4)}, {high})
        .AddAttribute("direction", mutation == Mutation::Direction ? "LEFT" : "RIGHT");
    auto* axes = Integers({axis});
    low = Op("Unsqueeze", {low, axes});
    high = Op("Unsqueeze", {high, axes});
    return mutation == Mutation::NibbleOrder ? Concat(high, low, axis) : Concat(low, high, axis);
  }

  NodeArg* Shape(NodeArg* input) {
    auto* result = b.MakeIntermediate();
    auto& node = b.AddNode("Shape", {input}, {result});
    if (mutation == Mutation::ShapeStart) node.AddAttribute("start", int64_t{-100});
    return result;
  }

  NodeArg* Slice(NodeArg* input, int64_t start, int64_t end) {
    return Op("Slice", {input, Integers({start}), Integers({end})});
  }

  NodeArg* Gather(NodeArg* table, NodeArg* indices) {
    auto* result = b.MakeIntermediate();
    b.AddNode("Gather", {table, indices}, {result}).AddAttribute("axis", int64_t{0});
    return result;
  }

  template <typename T>
  NodeArg* Activation(int64_t k, int rank) {
    std::vector<int64_t> shape = rank == 1   ? std::vector<int64_t>{k}
                                 : rank == 2 ? std::vector<int64_t>{2, k}
                                             : std::vector<int64_t>{2, 3, k};
    int64_t size = k * (rank == 1 ? 1 : rank == 2 ? 2
                                                  : 6);
    std::vector<T> data;
    for (int64_t i = 0; i < size; ++i) data.push_back(T(std::sin(static_cast<float>(i * 7)) * 0.7f));
    return b.MakeInput<T>(shape, data);
  }

  NodeArg* Packed(std::vector<int64_t> shape) {
    size_t size = 1;
    for (int64_t dim : shape) size *= static_cast<size_t>(dim);
    std::vector<uint8_t> data;
    for (size_t i = 0; i < size; ++i) data.push_back(static_cast<uint8_t>(i * 73 + 19));
    return mutation == Mutation::DynamicWeight ? b.MakeInput<uint8_t>(shape, data)
                                               : b.MakeInitializer<uint8_t>(shape, data);
  }

  template <typename T>
  void Finish(NodeArg* a, NodeArg* weight, int64_t n, bool bias) {
    auto* transposed = b.MakeIntermediate();
    b.AddNode("Transpose", {weight}, {transposed})
        .AddAttribute("perm", mutation == Mutation::Transpose ? std::vector<int64_t>{0, 1}
                                                              : std::vector<int64_t>{1, 0});
    auto* output = bias ? b.MakeIntermediate() : b.MakeOutput();
    b.AddNode("MatMul", {a, transposed}, {output});
    if (bias) {
      std::vector<T> values;
      for (int64_t i = 0; i < n; ++i) values.push_back(T(static_cast<float>(i - 2) / 8));
      b.AddNode("Add", {output, b.MakeInitializer<T>({n}, values)}, {b.MakeOutput()});
    }
    if (mutation == Mutation::SharedIntermediate) b.AddNode("Identity", {weight}, {b.MakeOutput()});
  }
};

template <typename T>
void BuildInt4(ModelTestBuilder& b, FixtureOptions options = {}) {
  ProjectionBody f{b, options.mutation};
  const int64_t k = options.block * options.blocks;
  auto* a = f.Activation<T>(k - (options.mutation == Mutation::TrimK ? 1 : 0), options.rank);
  auto* packed = f.Packed({options.n, options.blocks, options.block / 2});
  std::vector<T> scales;
  for (int64_t i = 0; i < options.n * options.blocks; ++i)
    scales.push_back(T(static_cast<float>(i % 11 + 1) / 31));
  auto* s = b.MakeInitializer<T>({options.n, options.blocks}, scales);
  auto* codes = f.Cast(f.Reshape(f.Reshape(f.Nibbles(packed, 3), {0, 0, -1}), {0, -1}),
                       ONNX_NAMESPACE::TensorProto_DataType_UINT4);
  NodeArg* zero = nullptr;
  if (options.mutation == Mutation::FloatZero) {
    zero = b.MakeInitializer<float>({options.n, options.blocks}, 0.0f, 15.0f);
  } else if (options.packed_zero) {
    auto* z = f.Packed({options.n, (options.blocks + 1) / 2});
    auto* rows = f.Reshape(f.Nibbles(z, 2, true), {0, -1});
    auto* end = f.Slice(f.Shape(s), 1, 2);
    zero = f.Op("Slice", {rows, f.Integers({0}), end, f.Integers({1})});
  } else {
    ONNX_NAMESPACE::TensorProto midpoint;
    midpoint.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
    midpoint.add_dims(1);
    const uint8_t value = options.mutation == Mutation::Midpoint ? 7 : 8;
    if (options.rank == 3)
      midpoint.set_raw_data(&value, 1);
    else
      midpoint.add_int32_data(value);
    zero = b.MakeIntermediate();
    b.AddNode("ConstantOfShape", {f.Shape(s)}, {zero}).AddAttribute("value", midpoint);
  }
  zero = f.Cast(zero, ONNX_NAMESPACE::TensorProto_DataType_UINT4);
  auto* weight = b.MakeIntermediate();
  auto& dequant = b.AddNode("DequantizeLinear", {codes, s, zero}, {weight});
  dequant.AddAttribute("axis", int64_t{1});
  dequant.AddAttribute("block_size", options.block);
  if (options.mutation == Mutation::TrimK)
    weight = f.Op("Slice", {weight, f.Integers({0}), f.Integers({k - 1}), f.Integers({1})});
  if (options.mutation == Mutation::GroupIndex)
    weight = f.Gather(weight, f.Integers({4, 3, 2, 1, 0}));
  f.Finish<T>(a, weight, options.n, options.bias);
}

void BuildNvFp4(ModelTestBuilder& b, FixtureOptions options = {}) {
  ProjectionBody f{b, options.mutation};
  const int64_t k = options.blocks * options.block;
  auto* a = f.Activation<MLFloat16>(k, options.rank);
  auto* packed = f.Packed({options.n, k / 2});
  std::vector<uint8_t> scales;
  for (int64_t i = 0; i < options.n * options.blocks; ++i)
    scales.push_back(static_cast<uint8_t>((i * 17 + 1) % 127));
  if (options.mutation == Mutation::NonfiniteScale) scales[0] = 127;
  auto* s = b.MakeInitializer<uint8_t>({options.n, options.blocks}, scales);
  const float global = options.mutation == Mutation::GlobalNonfinite ? std::numeric_limits<float>::infinity() : 0.375f;
  auto* g = b.MakeInitializer<float>(options.mutation == Mutation::GlobalVector ? std::vector<int64_t>{options.n, 1}
                                     : options.scalar_as_vector                 ? std::vector<int64_t>{1}
                                                                                : std::vector<int64_t>{},
                                     std::vector<float>(options.mutation == Mutation::GlobalVector
                                                            ? static_cast<size_t>(options.n)
                                                            : 1,
                                                        global));
  std::vector<float> fp4{0.0f, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  if (options.mutation == Mutation::Fp4Table) fp4[5] = 2.5f;
  if (options.mutation == Mutation::Fp4NegativeZero) fp4[8] = 0.0f;
  // Independent arithmetic specification of ml_dtypes.float8_e4m3fn's byte view.
  std::vector<float> fp8;
  for (int byte = 0; byte < 256; ++byte) {
    int exponent = (byte & 127) / 8;
    int mantissa = byte & 7;
    float magnitude = exponent == 0 ? static_cast<float>(mantissa) / 512
                                    : (8.0f + static_cast<float>(mantissa)) * std::pow(2.0f, exponent - 10);
    if ((byte & 127) == 127) magnitude = std::numeric_limits<float>::quiet_NaN();
    fp8.push_back(byte < 128 ? magnitude : -magnitude);
  }
  if (options.mutation == Mutation::Fp8Table) fp8[56] = 2;
  if (options.mutation == Mutation::Fp8NaNEntry) fp8[127] = 448;
  auto* codes = f.Cast(f.Reshape(f.Nibbles(packed, 2), {0, -1}), ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto* values = f.Gather(b.MakeInitializer<float>({16}, fp4), codes);
  auto* decoded_scales = f.Gather(b.MakeInitializer<float>({256}, fp8), f.Cast(s, ONNX_NAMESPACE::TensorProto_DataType_INT64));
  auto* unsqueezed = f.Op("Unsqueeze", {decoded_scales, f.Integers({2})});
  auto* prefix = f.Slice(f.Shape(unsqueezed), 0, 2);
  auto* expanded_shape = f.Concat(prefix, f.Integers({options.block}), 0);
  auto* expanded = f.Op("Expand", {unsqueezed, expanded_shape});
  auto* block_scales = f.Reshape(expanded, {0, -1});
  auto* weight = f.Op("Mul", {f.Op("Mul", {values, block_scales}), g});
  weight = f.Op("CastLike", {weight, a});
  f.Finish<MLFloat16>(a, weight, options.n, options.bias);
}

struct TestModel {
  Model model{"mobius", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 24}, {kMSDomain, 1}}, {}, DefaultLoggingManager().DefaultLogger()};
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder{graph};
};

size_t Count(const Graph& graph, const char* op) {
  size_t result = 0;
  for (const auto& node : graph.Nodes()) result += node.OpType() == op;
  return result;
}

std::string Serialize(const Model& model) {
  return model.ToProto().SerializeAsString();
}

Status Execute(const std::string& model, const ModelTestBuilder& builder, std::vector<OrtValue>& outputs) {
  SessionOptions options;
  options.graph_optimization_level = TransformerLevel::Default;
  options.intra_op_param.thread_pool_size = 1;
  InferenceSessionWrapper session{options, GetEnvironment()};
  std::istringstream stream(model);
  ORT_RETURN_IF_ERROR(session.Load(stream));
  ORT_RETURN_IF_ERROR(session.Initialize());
  return session.Run(RunOptions{}, builder.feeds_, builder.output_names_, &outputs);
}

TEST(MobiusQuantizedProjectionTests, Int4StructureAndCpuNumerics) {
  for (bool indexed : {false, true}) {
    for (bool packed_zero : {false, true}) {
      for (int rank : {1, 2, 3}) {
        for (int64_t block : {16, 32, 64}) {
          SCOPED_TRACE(MakeString(indexed, " packed=", packed_zero, " rank=", rank, " block=", block));
          TestModel test;
          FixtureOptions fixture{.packed_zero = packed_zero, .bias = true, .block = block, .blocks = block == 16 ? 3 : block == 32 ? 2
                                                                                                                                   : 1,
                                 .rank = rank};
          BuildInt4<float>(test.builder, fixture);
          test.builder.SetGraphOutputs();
          ASSERT_STATUS_OK(test.graph.Resolve());
          const auto baseline = Serialize(test.model);
          const auto* output = test.graph.GetOutputs()[0];
          std::string diagnostics;
          Options options{.indexed = indexed};
          options.on_failure = [&](const MatchFailure& failure) {
            if (failure.stage != FailureStage::Structure) diagnostics += failure.reason + "\n";
          };
          PatternRewriter rewriter(MakeMobiusMatMulNBitsRules(), options);
          Statistics stats;
          ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
          ASSERT_EQ(stats.rewrites, 1u) << diagnostics;
          ASSERT_STATUS_OK(test.graph.Resolve());
          EXPECT_EQ(test.graph.GetOutputs()[0], output);
          EXPECT_EQ(Count(test.graph, "MatMulNBits"), 1u);
          EXPECT_EQ(Count(test.graph, "MatMul"), 0u);
          EXPECT_EQ(Count(test.graph, "Add"), 1u);
          for (const auto& node : test.graph.Nodes()) {
            if (node.OpType() != "MatMulNBits") continue;
            EXPECT_EQ(node.GetAttributes().at("K").i(), fixture.block * fixture.blocks);
            EXPECT_EQ(node.GetAttributes().at("N").i(), fixture.n);
            EXPECT_EQ(node.GetAttributes().at("block_size").i(), block);
            EXPECT_EQ(node.GetAttributes().at("bits").i(), 4);
            EXPECT_EQ(node.GetAttributes().at("accuracy_level").i(), 0);
            EXPECT_EQ(node.InputDefs().size(), packed_zero ? 4u : 3u);
          }
          std::vector<OrtValue> expected, actual;
          ASSERT_STATUS_OK(Execute(baseline, test.builder, expected));
          ASSERT_STATUS_OK(Execute(Serialize(test.model), test.builder, actual));
          ASSERT_EQ(expected.size(), 1u);
          ASSERT_EQ(actual.size(), 1u);
          const auto& x = expected[0].Get<Tensor>();
          const auto& y = actual[0].Get<Tensor>();
          ASSERT_EQ(x.Shape(), y.Shape());
          for (int64_t i = 0; i < x.Shape().Size(); ++i) {
            ASSERT_TRUE(std::isfinite(x.Data<float>()[i]));
            ASSERT_TRUE(std::isfinite(y.Data<float>()[i]));
            EXPECT_NEAR(x.Data<float>()[i], y.Data<float>()[i], 2e-5f + 2e-4f * std::abs(x.Data<float>()[i]));
          }
          ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
          EXPECT_EQ(stats.rewrites, 0u);
        }
      }
    }
  }
}

TEST(MobiusQuantizedProjectionTests, Int4Fp16Structure) {
  for (bool packed_zero : {false, true}) {
    TestModel test;
    BuildInt4<MLFloat16>(test.builder, {.packed_zero = packed_zero});
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(MakeMobiusMatMulNBitsRules()).Run(test.graph, stats));
    ASSERT_EQ(stats.rewrites, 1u);
    ASSERT_STATUS_OK(test.graph.Resolve());
    EXPECT_EQ(test.graph.GetOutputs()[0]->TypeAsProto()->tensor_type().elem_type(),
              ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  }
}

TEST(MobiusQuantizedProjectionTests, NvFp4StructureOnly) {
  for (bool indexed : {false, true}) {
    for (int rank : {1, 2, 3}) {
      for (int64_t blocks : {1, 2, 3}) {
        TestModel test;
        BuildNvFp4(test.builder, {.bias = true, .blocks = blocks, .rank = rank, .scalar_as_vector = rank == 3});
        test.builder.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        Statistics stats;
        PatternRewriter rewriter(MakeMobiusNvFp4Rules(), {.indexed = indexed});
        ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
        ASSERT_EQ(stats.rewrites, 1u);
        ASSERT_STATUS_OK(test.graph.Resolve());
        EXPECT_EQ(Count(test.graph, "MatMulBlockQuantizedFp4Weight"), 1u);
        EXPECT_EQ(Count(test.graph, "Gather"), 0u);
        EXPECT_EQ(Count(test.graph, "MatMul"), 0u);
        EXPECT_EQ(Count(test.graph, "Add"), 1u);
        for (const auto& node : test.graph.Nodes()) {
          if (node.OpType() != "MatMulBlockQuantizedFp4Weight") continue;
          EXPECT_EQ(node.InputDefs().size(), 4u);
          EXPECT_EQ(node.GetAttributes().at("block_size").i(), 16);
          const auto* g = test.graph.GetConstantInitializer(node.InputDefs()[3]->Name(), true);
          ASSERT_NE(g, nullptr);
          Initializer global(*g);
          EXPECT_FLOAT_EQ(global.data<float>()[0], 0.375f);
        }
        std::shared_ptr<Model> reloaded;
        ASSERT_STATUS_OK(Model::Load(test.model.ToProto(), reloaded, nullptr, DefaultLoggingManager().DefaultLogger()));
        EXPECT_EQ(Count(reloaded->MainGraph(), "MatMulBlockQuantizedFp4Weight"), 1u);
        ASSERT_STATUS_OK(rewriter.Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 0u);
      }
    }
  }
}

TEST(MobiusQuantizedProjectionTests, RejectChangedBodiesAndDeletionBoundaries) {
  const std::vector<Mutation> common{Mutation::Mask, Mutation::Shift, Mutation::NibbleOrder, Mutation::Direction,
                                     Mutation::Transpose,
                                     Mutation::SharedIntermediate, Mutation::DynamicWeight, Mutation::ShapeStart};
  for (bool indexed : {false, true}) {
    for (bool nvfp4 : {false, true}) {
      auto mutations = common;
      if (nvfp4) {
        for (auto mutation : {Mutation::Fp4Table, Mutation::Fp4NegativeZero, Mutation::Fp8Table,
                              Mutation::Fp8NaNEntry, Mutation::NonfiniteScale, Mutation::GlobalVector,
                              Mutation::GlobalNonfinite})
          mutations.push_back(mutation);
      } else {
        for (auto mutation : {Mutation::Midpoint, Mutation::ZeroMask, Mutation::FloatZero,
                              Mutation::TrimK, Mutation::GroupIndex})
          mutations.push_back(mutation);
      }
      for (auto mutation : mutations) {
        SCOPED_TRACE(MakeString("NVFP4=", nvfp4, " mutation=", static_cast<int>(mutation), " indexed=", indexed));
        TestModel test;
        FixtureOptions fixture{.packed_zero = mutation == Mutation::ZeroMask, .mutation = mutation};
        if (mutation == Mutation::Transpose) fixture.n = fixture.block * fixture.blocks;
        if (nvfp4)
          BuildNvFp4(test.builder, fixture);
        else
          BuildInt4<float>(test.builder, fixture);
        test.builder.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        const auto original = Serialize(test.model);
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter(nvfp4 ? MakeMobiusNvFp4Rules() : MakeMobiusMatMulNBitsRules(),
                                         {.indexed = indexed})
                             .Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, 0u);
        EXPECT_TRUE(Serialize(test.model) == original);
      }
    }
  }
}

TEST(MobiusQuantizedProjectionTests, ProviderAndOpsetBoundaries) {
  for (bool nvfp4 : {false, true}) {
    for (const char* provider : {kCpuExecutionProvider, kCudaExecutionProvider, kWebGpuExecutionProvider}) {
      for (bool allow : {false, true}) {
        TestModel test;
        if (nvfp4)
          BuildNvFp4(test.builder);
        else
          BuildInt4<float>(test.builder);
        test.builder.SetGraphOutputs();
        ASSERT_STATUS_OK(test.graph.Resolve());
        for (auto& node : test.graph.Nodes()) node.SetExecutionProviderType(provider);
        Options options;
        options.compatible_execution_providers.clear();
        if (allow) options.compatible_execution_providers.insert(provider);
        Statistics stats;
        ASSERT_STATUS_OK(PatternRewriter(nvfp4 ? MakeMobiusNvFp4Rules() : MakeMobiusMatMulNBitsRules(), options)
                             .Run(test.graph, stats));
        EXPECT_EQ(stats.rewrites, allow && (std::string_view(provider) == kCudaExecutionProvider ||
                                            (!nvfp4 && std::string_view(provider) == kCpuExecutionProvider))
                                      ? 1u
                                      : 0u);
      }
    }
    TestModel test;
    if (nvfp4)
      BuildNvFp4(test.builder);
    else
      BuildInt4<float>(test.builder);
    test.builder.SetGraphOutputs();
    ASSERT_STATUS_OK(test.graph.Resolve());
    auto proto = test.model.ToProto();
    for (auto& import : *proto.mutable_opset_import()) {
      if (import.domain().empty()) import.set_version(25);
    }
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(proto, model, nullptr, DefaultLoggingManager().DefaultLogger()));
    Statistics stats;
    ASSERT_STATUS_OK(PatternRewriter(nvfp4 ? MakeMobiusNvFp4Rules() : MakeMobiusMatMulNBitsRules())
                         .Run(model->MainGraph(), stats));
    EXPECT_EQ(stats.rewrites, 0u);
  }
}

TEST(MobiusQuantizedProjectionTests, RejectUnsupportedTypesGeometryAndOverridableWeights) {
  for (bool nvfp4 : {false, true}) {
    for (int scenario = 0; scenario < 3; ++scenario) {
      SCOPED_TRACE(MakeString("NVFP4=", nvfp4, " scenario=", scenario));
      TestModel test;
      FixtureOptions fixture;
      if (scenario == 0) fixture.block = nvfp4 ? 32 : 8;
      if (nvfp4)
        BuildNvFp4(test.builder, fixture);
      else
        BuildInt4<float>(test.builder, fixture);
      test.builder.SetGraphOutputs();
      ASSERT_STATUS_OK(test.graph.Resolve());
      auto proto = test.model.ToProto();
      if (scenario == 1) {
        // The standard bodies also resolve with BF16; these factories do not opt into it.
        proto.mutable_graph()->clear_value_info();
        for (auto& input : *proto.mutable_graph()->mutable_input())
          input.mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
        for (auto& output : *proto.mutable_graph()->mutable_output())
          output.mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16);
        if (!nvfp4) {
          for (auto& tensor : *proto.mutable_graph()->mutable_initializer()) {
            if (tensor.data_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) continue;
            Initializer data(tensor);
            auto converted = data.ToBFloat16(tensor.name());
            tensor = std::move(converted);
          }
        }
      } else if (scenario == 2) {
        const NodeArg* packed = nullptr;
        for (const auto& node : test.graph.Nodes()) {
          if (node.OpType() == "BitwiseAnd") {
            packed = node.InputDefs()[0];
            break;
          }
        }
        ASSERT_NE(packed, nullptr);
        auto* input = proto.mutable_graph()->add_input();
        input->set_name(packed->Name());
        *input->mutable_type() = *packed->TypeAsProto();
      }
      std::shared_ptr<Model> model;
      ASSERT_STATUS_OK(Model::Load(proto, model, nullptr, DefaultLoggingManager().DefaultLogger()));
      const auto original = Serialize(*model);
      Statistics stats;
      ASSERT_STATUS_OK(PatternRewriter(nvfp4 ? MakeMobiusNvFp4Rules() : MakeMobiusMatMulNBitsRules())
                           .Run(model->MainGraph(), stats));
      EXPECT_EQ(stats.rewrites, 0u);
      EXPECT_TRUE(Serialize(*model) == original);
    }
  }
}
}  // namespace
#endif  // DISABLE_CONTRIB_OPS
}  // namespace onnxruntime::test
