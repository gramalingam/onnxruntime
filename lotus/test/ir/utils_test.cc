#ifdef _MSC_VER
#pragma warning(push)
// 'identifier' : unreferenced formal parameter
#pragma warning(disable : 4100)
// 'type' : forcing value to bool 'true' or 'false' (performance warning)
#pragma warning(disable : 4800)
#endif
#include "google/protobuf/util/message_differencer.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "onnx/defs/data_type_utils.h"
#include "onnx/onnx_pb.h"
#include "gtest/gtest.h"

using google::protobuf::util::MessageDifferencer;
using onnx::Utils::DataTypeUtils;
using namespace onnx;

namespace LotusIR {
namespace Test {

TEST(OpUtilsTest, TestPTYPE) {
  DataType p1 = DataTypeUtils::ToType("tensor(int32)");
  DataType p2 = DataTypeUtils::ToType("tensor(int32)");
  DataType p3 = DataTypeUtils::ToType("tensor(int32)");
  EXPECT_EQ(p1, p2);
  EXPECT_EQ(p2, p3);
  EXPECT_EQ(p1, p3);
  DataType p4 = DataTypeUtils::ToType("seq(tensor(int32))");
  DataType p5 = DataTypeUtils::ToType("seq(tensor(int32))");
  DataType p6 = DataTypeUtils::ToType("seq(tensor(int32))");
  EXPECT_EQ(p4, p5);
  EXPECT_EQ(p5, p6);
  EXPECT_EQ(p4, p6);

  TypeProto t1 = DataTypeUtils::ToTypeProto(p1);
  EXPECT_TRUE(t1.has_tensor_type());
  EXPECT_TRUE(t1.tensor_type().has_elem_type());
  EXPECT_EQ(t1.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t2 = DataTypeUtils::ToTypeProto(p2);
  EXPECT_TRUE(t2.has_tensor_type());
  EXPECT_TRUE(t2.tensor_type().has_elem_type());
  EXPECT_EQ(t2.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t3 = DataTypeUtils::ToTypeProto(p3);
  EXPECT_TRUE(t3.has_tensor_type());
  EXPECT_TRUE(t3.tensor_type().has_elem_type());
  EXPECT_EQ(t3.tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t4 = Utils::DataTypeUtils::ToTypeProto(p4);
  EXPECT_TRUE(t4.has_sequence_type());
  EXPECT_TRUE(t4.sequence_type().has_elem_type());
  EXPECT_TRUE(t4.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t4.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t4.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t5 = Utils::DataTypeUtils::ToTypeProto(p5);
  EXPECT_TRUE(t5.has_sequence_type());
  EXPECT_TRUE(t5.sequence_type().has_elem_type());
  EXPECT_TRUE(t5.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t5.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t5.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
  TypeProto t6 = Utils::DataTypeUtils::ToTypeProto(p6);
  EXPECT_TRUE(t6.has_sequence_type());
  EXPECT_TRUE(t6.sequence_type().has_elem_type());
  EXPECT_TRUE(t6.sequence_type().elem_type().has_tensor_type());
  EXPECT_TRUE(t6.sequence_type().elem_type().tensor_type().has_elem_type());
  EXPECT_EQ(t6.sequence_type().elem_type().tensor_type().elem_type(), TensorProto_DataType::TensorProto_DataType_INT32);
}
}  // namespace Test
}  // namespace LotusIR
