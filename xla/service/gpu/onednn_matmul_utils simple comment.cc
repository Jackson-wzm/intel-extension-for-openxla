/* Copyright (c) 2023 Intel Corporation

Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

//  版权声明和许可证信息  然后导入了一些必要的库和头文件，包括onednn_matmul_utils.h、algorithm、cstdint等。
// 这些头文件提供了所需的功能和类型，包括算法、可选参数、元组、类型特征、实用工具、向量和XeTLA库。
#include "xla/service/gpu/onednn_matmul_utils.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <xetla.hpp>

#include "xla/service/gpu/xetla/gemm/gemm.h"
#include "xla/service/onednn_util.h"

// 命名空间和函数定义
namespace xla {
namespace gpu {

// SYCLGemm子命名空间
namespace SYCLGemm{
    
    //EpilogueCast函数  将一个字符串（string）转换为对应的 GemmBackendEpilogue 枚举类型
    absl::StatusOr<GemmBackendEpilogue> EpilogueCast(std::string& epilogue){
        if(epilogue == "DEFAULT"){
            return GemmBackendEpilogue::DEFAULT;
        }else if(epilogue == "RELU"){
            return GemmBackendEpilogue::RELU;
        }else if(epilogue == "GELU"){
            return GemmBackendEpilogue::GELU;
        }else if(epilogue == "BIAS"){
            return GemmBackendEpilogue::BIAS;
        }else if(epilogue == "BIAS_RELU"){
            return GemmBackendEpilogue::BIAS_RELU;
        }else if(epilogue == "BIAS_GELU"){
            return GemmBackendEpilogue::BIAS_GELU;
        }else if(epilogue == "GELU_AUX"){
            return GemmBackendEpilogue::GELU_AUX;
        }else if(epilogue == "BIAS_GELU_AUX"){
            return GemmBackendEpilogue::BIAS_GELU_AUX;
        }else{
            return Internal("Unknown Epilogue.");
        }
    }
    //EpilogueCast函数将 将 GemmBackendEpilogue 枚举值转换为对应字符串表示的函数 EpilogueCast。
    absl::StatusOr<std::string> EpilogueCast(GemmBackendEpilogue epilogue){
        if(epilogue == GemmBackendEpilogue::DEFAULT){
            return "DEFAULT";
        }else if(epilogue == GemmBackendEpilogue::RELU){
            return "RELU";
        }else if(epilogue == GemmBackendEpilogue::GELU){
            return "GELU";
        }else if(epilogue == GemmBackendEpilogue::BIAS){
            return "BIAS";
        }else if(epilogue == GemmBackendEpilogue::BIAS_RELU){
            return "BIAS_RELU";
        }else if(epilogue == GemmBackendEpilogue::BIAS_GELU){
            return "BIAS_GELU";
        }else if(epilogue == GemmBackendEpilogue::GELU_AUX){
            return "GELU_AUX";
        }else if(epilogue == GemmBackendEpilogue::BIAS_GELU_AUX){
            return "BIAS_GELU_AUX";
        }else{
            return Internal("Unknown Epilogue.");
        }
    }
    
    // EpilogueAddsVectorBias 函数根据GemmBackendEpilogue的值判断是否添加向量偏置
    absl::StatusOr<bool> EpilogueAddsVectorBias(GemmBackendEpilogue epilogue) {
        switch (epilogue) {
            case GemmBackendEpilogue::DEFAULT:
            case GemmBackendEpilogue::RELU:
            case GemmBackendEpilogue::GELU:
            case GemmBackendEpilogue::GELU_AUX:
                return false;
            case GemmBackendEpilogue::BIAS:
            case GemmBackendEpilogue::BIAS_RELU:
            case GemmBackendEpilogue::BIAS_GELU:
            case GemmBackendEpilogue::BIAS_GELU_AUX:
                return true;
            default:
                return Internal("Unknown Epilogue.");
        }
    }
    //EpilogueHasAuxiliaryOutput函数根据GemmBackendEpilogue的值判断是否具有辅助输出。
    absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendEpilogue epilogue) {
        switch (epilogue) {
            case GemmBackendEpilogue::DEFAULT:
            case GemmBackendEpilogue::RELU:
            case GemmBackendEpilogue::GELU:
            case GemmBackendEpilogue::BIAS:
            case GemmBackendEpilogue::BIAS_RELU:
            case GemmBackendEpilogue::BIAS_GELU:
                return false;
            case GemmBackendEpilogue::GELU_AUX:
            case GemmBackendEpilogue::BIAS_GELU_AUX:
                return true;
            default:
              return Internal("Unknown Epilogue.");
        }
    }
    // AsSYCLEpilogue函数根据GemmBackendConfig_Epilogue的值返回相应的GemmBackendEpilogue。
    absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(
        GemmBackendConfig_Epilogue epilogue) {
          switch (epilogue) {
            case GemmBackendConfig::DEFAULT:
              return GemmBackendEpilogue::DEFAULT;
            case GemmBackendConfig::RELU:
              return GemmBackendEpilogue::RELU;
            case GemmBackendConfig::GELU:
              return GemmBackendEpilogue::GELU;
            case GemmBackendConfig::GELU_AUX:
              return GemmBackendEpilogue::GELU_AUX;
            case GemmBackendConfig::BIAS:
              return GemmBackendEpilogue::BIAS;
            case GemmBackendConfig::BIAS_RELU:
              return GemmBackendEpilogue::BIAS_RELU;
            case GemmBackendConfig::BIAS_GELU:
              return GemmBackendEpilogue::BIAS_GELU;
            case GemmBackendConfig::BIAS_GELU_AUX:
              return GemmBackendEpilogue::BIAS_GELU_AUX;
            default:
              return Internal("Unsupported Epilogue.");
          }
    }
}


/*
PrimitiveTypeToXetlaNative 模板结构和类型定义
定义模板结构 PrimitiveTypeToXetlaNative，将 XLA的PrimitiveType转换为 相应的Xetla原生类型,数据结构。
1. 将 F32 类型映射为 float 类型、
2. 将 F16 类型映射为 sycl::half 类型
3. 将 BF16 类型映射为 ::gpu::xetla::bf16 类型。
4. 将 S8 类型映射为 int8_t 类型。
5. 将 S32 类型映射为 int32_t 类型。

通过这种模板特化，我们可以轻松地将某个 PrimitiveType 映射到相应的 Xetla 本地类型

如果 要使用 Xetla 进行计算， 那么就需要把 常规的数据转变为 符合 Xetla库 种张量操作数据格式，才能被计算
*/ 

// Returns the xetla native type (eg, float) corresponding to the given template
// parameter XLA primitive type (eg, F32).
template <PrimitiveType>
struct PrimitiveTypeToXetlaNative;

template <>
struct PrimitiveTypeToXetlaNative<F32> {
  using type = float;
};
template <>
struct PrimitiveTypeToXetlaNative<F16> {
  using type = sycl::half;
};
template <>
struct PrimitiveTypeToXetlaNative<BF16> {
  using type = ::gpu::xetla::bf16;
};
template <>
struct PrimitiveTypeToXetlaNative<S8> {
  using type = int8_t;
};
template <>
struct PrimitiveTypeToXetlaNative<S32> {
  using type = int32_t;
};

/// Return oneDNN data type (memory::data_type) for input type T
///
/// @input None
/// @return dnnl::memory::data_type corresponding to type T

/*

  定义了模板函数 OneDnnType<T>，并对一些特定的数据类型进行了模板特化。
  模板特化的主要目的是将不同的C++数据类型映射到 OneDNN 库中的相应类型。
  
  这些模板特化函数的作用是将不同的 C++ 数据类型映射到 OneDNN 库中相应的 dnnl::memory::data_type 枚举值。具体来说：
  float 映射到 dnnl::memory::data_type::f32
  double 映射到 dnnl::memory::data_type::f64
  sycl::half 映射到 dnnl::memory::data_type::f16
  int8_t 映射到 dnnl::memory::data_type::s8
  int32_t 映射到 dnnl::memory::data_type::s32
  ::gpu::xetla::bf16 映射到 dnnl::memory::data_type::bf16
  作用
  这些模板特化函数的主要作用是提供一个统一的接口，通过调用 OneDnnType<T>() 函数，可以将不同的C++数据类型转换为 OneDNN 库中相应的数据类型。
  为后续计算做好数据的转化和准备
      
*/

template <typename T>
inline dnnl::memory::data_type OneDnnType();

/// Instantiation for float type. Add similar instantiations for other
/// type if needed.
template <>
inline dnnl::memory::data_type OneDnnType<float>() {
  return dnnl::memory::data_type::f32;
}

template <>
inline dnnl::memory::data_type OneDnnType<double>() {
  return dnnl::memory::data_type::f64;
}

template <>
inline dnnl::memory::data_type OneDnnType<sycl::half>() {
  return dnnl::memory::data_type::f16;
}

template <>
inline dnnl::memory::data_type OneDnnType<int8_t>() {
  return dnnl::memory::data_type::s8;
}

template <>
inline dnnl::memory::data_type OneDnnType<int32_t>() {
  return dnnl::memory::data_type::s32;
}

template <>
inline dnnl::memory::data_type OneDnnType<::gpu::xetla::bf16>() {
  return dnnl::memory::data_type::bf16;
}


// 矩阵描述符和其他结构
namespace {

  /*
  获取矩阵描述符
  检查矩阵的布局顺序是否为列主序。如果是列主序，则需要转置矩阵
  包括：
  1. 直接传入的矩阵数据
  2. 根据是否需要转置设置相应的转置选项
  3. 如果转置，则行数为列数，否则为行数
  4. 如果转置，则列数为行数，否则为列数
  5. 批处理的步幅
  6. 主维度的步幅

  */
MatrixDescriptor GetMatrixDesc(const MatrixLayout& layout,
                               se::DeviceMemoryBase data) {
  bool transpose = layout.order == MatrixLayout::Order::kColumnMajor;
  return MatrixDescriptor{
      data,
      transpose ? se::blas::Transpose::kTranspose
                : se::blas::Transpose::kNoTranspose,
      transpose ? layout.num_cols : layout.num_rows,
      transpose ? layout.num_rows : layout.num_cols,
      layout.batch_stride,
      layout.leading_dim_stride,
  };
}

/*
创建 结构体 OneDnnMatMulParams 用于存储oneDNN矩阵乘法的参数，包括维度和步幅信息。
*/

struct OneDnnMatMulParams {
  dnnl::memory::dims a_dims;
  dnnl::memory::dims b_dims;
  dnnl::memory::dims c_dims;
  dnnl::memory::dims bias_dims;
  dnnl::memory::dims a_strides;
  dnnl::memory::dims b_strides;
  dnnl::memory::dims c_strides;
  dnnl::memory::dims bias_strides;

  OneDnnMatMulParams(dnnl::memory::dims a_dims, dnnl::memory::dims b_dims,
                     dnnl::memory::dims c_dims, dnnl::memory::dims bias_dims,
                     dnnl::memory::dims a_strides, dnnl::memory::dims b_strides,
                     dnnl::memory::dims c_strides,
                     dnnl::memory::dims bias_strides)
      : a_dims(std::move(a_dims)),
        b_dims(std::move(b_dims)),
        c_dims(std::move(c_dims)),
        bias_dims(std::move(bias_dims)),
        a_strides(std::move(a_strides)),
        b_strides(std::move(b_strides)),
        c_strides(std::move(c_strides)),
        bias_strides(std::move(bias_strides)) {}
};



//表示这是一个模板特化。参数 InputT 表示矩阵元素的类型
template <typename InputT>
std::enable_if_t<std::is_same_v<InputT, ::gpu::xetla::bf16> ||
                     std::is_same_v<InputT, sycl::half>,
                 absl::StatusOr<bool>>


// 定义模板函数 RunXetlaGemm，用于运行 Xetla GEMM（通用矩阵乘法）操作。这个函数基于模板类型 InputT 进行实例化，
// 并根据提供的后处理类型（epilogue）来配置和运行 GEMM 操作
/*  
  定义函数 RunXetlaGemm，参数包括：
  handle: GPU 流处理句柄。
  lhs, rhs, c, out: 矩阵描述符，分别表示左操作数、右操作数、中间结果和输出结果。
  bias: 偏置矩阵的数据。
  
  SYCLGemm： 上述定义好的结构体
    SYCLGemm::GemmBackendEpilogue  这是我们 首次 定义的RunXetlaGemm方法， 这个方法会调用之前定义好的  GemmBackendEpilogue 用于返回具体的数值
    epilogue: 后处理类型，指定 GEMM 操作后如何处理结果。
  beta: 一个浮点数，用于缩放矩阵 c。
  
  总体来说，这段代码通过模板和多种后处理策略，灵活地配置并运行 Xetla GEMM 操作。
  每个后处理策略通过 switch 语句选择，并根据 beta 值进行适当的处理。
*/ 

RunXetlaGemm(se::gpu::GpuStreamHandle handle, const MatrixDescriptor& lhs,
             const MatrixDescriptor& rhs, const MatrixDescriptor& c,
             const MatrixDescriptor& out, se::DeviceMemoryBase bias,
             SYCLGemm::GemmBackendEpilogue epilogue, float beta) {
  void* bias_data = const_cast<void*>(bias.opaque());
  void* c_data = const_cast<void*>(c.data.opaque());
  switch (epilogue) {
    case SYCLGemm::GemmBackendEpilogue::DEFAULT: {
      auto policy = ::gpu::xetla::XetlaGemmKernel<InputT>()
                        .add_matrix_c(out)
                        .add_matrix_a(lhs)
                        .add_matrix_b(rhs)
                        .build();
      if (fabs(beta) - 0.0f > 1e-6) {
        if (fabs(beta) - 1.0f < 1e-6) {
          policy
              .add_epilogue(
                  c_data,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::RES_ADD)
              .build();
        } else {
          return true;
        }
      }
      if (policy.fallback() == false) {
        return !policy.run(handle);
      }
      return policy.fallback();
    }
    case SYCLGemm::GemmBackendEpilogue::BIAS: {
      auto policy =
          ::gpu::xetla::XetlaGemmKernel<InputT>()
              .add_matrix_c(out)
              .add_matrix_a(lhs)
              .add_matrix_b(rhs)
              .add_epilogue(
                  bias_data,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::BIAS)
              .build();
      if (fabs(beta) - 0.0f > 1e-6) {
        policy
            .add_epilogue(
                c_data,
                ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::RES_ADD,
                beta)
            .build();
      }
      if (policy.fallback() == false) {
        return !policy.run(handle);
      }
      return policy.fallback();
    }
    case SYCLGemm::GemmBackendEpilogue::GELU: {
      auto policy =
          ::gpu::xetla::XetlaGemmKernel<InputT>()
              .add_matrix_c(out)
              .add_matrix_a(lhs)
              .add_matrix_b(rhs)
              .add_epilogue(
                  nullptr,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::GELU)
              .build();
      if (policy.fallback() == false) {
        return !policy.run(handle);
      }
      return policy.fallback();
    }
    case SYCLGemm::GemmBackendEpilogue::BIAS_GELU: {
      auto policy =
          ::gpu::xetla::XetlaGemmKernel<InputT>()
              .add_matrix_c(out)
              .add_matrix_a(lhs)
              .add_matrix_b(rhs)
              .add_epilogue(
                  bias_data,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::BIAS)
              .add_epilogue(
                  nullptr,
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::GELU)
              .build();
      if (policy.fallback() == false) {
        return !policy.run(handle);
      }
      return policy.fallback();
    }
    case SYCLGemm::GemmBackendEpilogue::RELU:
    case SYCLGemm::GemmBackendEpilogue::BIAS_RELU:
      return true;
    default:
      return Internal("Unsupported Activation mode");
  }
}
template <typename InputT>
std::enable_if_t<!std::is_same_v<InputT, ::gpu::xetla::bf16> &&
                     !std::is_same_v<InputT, sycl::half>,
                 absl::StatusOr<bool>>
RunXetlaGemm(se::gpu::GpuStreamHandle handle, const MatrixDescriptor& lhs,
             const MatrixDescriptor& rhs, const MatrixDescriptor& c,
             const MatrixDescriptor& out, se::DeviceMemoryBase bias,
             SYCLGemm::GemmBackendEpilogue epilogue, float beta) {
  return Internal("Unsupported Datatype in XeTLA");
}




 /*
          CreateMatMulParams  该函数用于创建矩阵乘法的参数，返回一个OneDnnMatMulParams的智能指针。
          输入包括批量大小、左矩阵、右矩阵和输出矩阵的描述符
          这段代码设计用于设置矩阵乘法操作的参数，特别是用于 OneDNN 库。它包括处理特殊情况，如矩阵转置和设置适当的内存步幅
          功能包括：
          1. 定义左矩阵、右矩阵和输出矩阵的维度
          2. 定义左矩阵、右矩阵和输出矩阵的步幅
          3. 如果左矩阵需要转置，交换其最后两个维度和相应的步幅
          4. 如果右矩阵需要转置，交换其最后两个维度和相应的步幅
          5. 定义偏置维度和步幅
          6. 返回一个新的OneDnnMatMulParams对象
          */
std::unique_ptr<OneDnnMatMulParams> CreateMatMulParams(
    int64_t batch_size, const MatrixDescriptor& lhs,
    const MatrixDescriptor& rhs, const MatrixDescriptor& out) {
  dnnl::memory::dims lhs_dims{batch_size, lhs.num_rows, lhs.num_cols};
  dnnl::memory::dims rhs_dims{batch_size, rhs.num_rows, rhs.num_cols};
  dnnl::memory::dims out_dims{batch_size, out.num_rows, out.num_cols};

  auto lhs_strides =
      dnnl::memory::dims{lhs.batch_stride, lhs.leading_dim_stride, 1};
  auto rhs_strides =
      dnnl::memory::dims{rhs.batch_stride, rhs.leading_dim_stride, 1};
  auto out_strides =
      dnnl::memory::dims{out.batch_stride, out.leading_dim_stride, 1};
  int idx_last = 2;
  int idx_2nd_last = 1;

  // dst(m,n) = \sigma{src(m,k) * weights(k, n)}
  // lhs_strides holds the strides for each dim, say {24, 12, 4, 1} for
  // src_tensor {1, 2, 3, 4} if adj_x_ is false.
  // If adj_x_ is true, swap the innermost two dims of lhs_strides
  // to {24, 12, 1, 4}, just like set memory::format_tag::abdc
  if (lhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs_dims[idx_last], lhs_dims[idx_2nd_last]);
    std::swap(lhs_strides[idx_last], lhs_strides[idx_2nd_last]);
  }
  if (rhs.transpose == se::blas::Transpose::kTranspose) {
    std::swap(rhs_dims[idx_last], rhs_dims[idx_2nd_last]);
    std::swap(rhs_strides[idx_last], rhs_strides[idx_2nd_last]);
  }

  dnnl::memory::dims bias_dims(rhs_dims.size(), 1);
  bias_dims[rhs_dims.size() - 1] = rhs_dims[rhs_dims.size() - 1];
  auto bias_strides = CalculateTFStrides(bias_dims);

  return absl::make_unique<OneDnnMatMulParams>(
      lhs_dims, rhs_dims, out_dims, bias_dims, lhs_strides, rhs_strides,
      out_strides, bias_strides);
}


// DoXetlaGemm函数 执行Xetla的GEMM操作。根据提供的矩阵描述符和其他参数，执行GEMM运算
template <typename InputT>
absl::Status DoXetlaGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                         const MatrixDescriptor& lhs,
                         const MatrixDescriptor& rhs, const MatrixDescriptor& c,
                         const MatrixDescriptor& output,
                         se::DeviceMemoryBase bias, float alpha, float beta,
                         SYCLGemm::GemmBackendEpilogue epilogue, se::Stream* stream,
                         std::optional<se::blas::AlgorithmType> algorithm,
                         se::ScratchAllocator* scratch_allocator,
                         se::blas::ComputePrecision compute_precision) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::gpu::GpuStreamHandle stream_handle =
      stream_executor::gpu::AsGpuStreamValue(stream);
  TF_ASSIGN_OR_RETURN(bool fallback,
                      RunXetlaGemm<InputT>(stream_handle, lhs, rhs, c, output,
                                           bias, epilogue, beta));
  if (!fallback) return OkStatus();
  VLOG(2) << "lhs: " << batch_size << " " << lhs.num_rows << " "
          << lhs.num_cols;
  VLOG(2) << "rhs: " << batch_size << " " << rhs.num_rows << " "
          << rhs.num_cols;
  VLOG(2) << "out: " << batch_size << " " << output.num_rows << " "
          << output.num_cols;
  return absl::InternalError("Anyway, something is wrong in DoXetlaGemm.");
}


/*
  这是 DoOnednnGemm函数的完整声明
  跟上述  DoXetlaGemm 函数的完整声明一样，返回类型是 absl::Status。
  参数列表包含多个矩阵描述符、计算参数和上下文信息。
  矩阵乘法（GEMM）操作的实现和优化上，分别通过Xetla和oneDNN 两个后端来执行这些操作
  
  具体执行包括：
  1. 创建矩阵乘法参数
  2. 创建oneDNN的内存描述符
  3. 创建oneDNN引擎和属性
  4. 设置fp32模式
  5. 创建矩阵乘法的primitive descriptor
*/  

template <typename InputT, typename OutputT>
absl::Status DoOnednnGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                          const MatrixDescriptor& lhs,
                          const MatrixDescriptor& rhs,
                          const MatrixDescriptor& c,
                          const MatrixDescriptor& output,
                          se::DeviceMemoryBase bias, float alpha, float beta,
                          SYCLGemm::GemmBackendEpilogue epilogue,
                          se::Stream* stream,
                          std::optional<se::blas::AlgorithmType> algorithm,
                          se::ScratchAllocator* scratch_allocator,
                          se::blas::ComputePrecision compute_precision) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::gpu::GpuStreamHandle stream_handle =
      stream_executor::gpu::AsGpuStreamValue(stream);
  void* lhs_data = const_cast<void*>(lhs.data.opaque());
  void* rhs_data = const_cast<void*>(rhs.data.opaque());
  void* c_data = const_cast<void*>(c.data.opaque());
  void* out_data = const_cast<void*>(output.data.opaque());
  void* bias_data = const_cast<void*>(bias.opaque());

  VLOG(2) << "lhs: " << batch_size << " " << lhs.num_rows << " "
          << lhs.num_cols;
  VLOG(2) << "rhs: " << batch_size << " " << rhs.num_rows << " "
          << rhs.num_cols;
  VLOG(2) << "out: " << batch_size << " " << output.num_rows << " "
          << output.num_cols;
  VLOG(2) << "lhs stride: " << lhs.batch_stride << " " << lhs.leading_dim_stride
          << " " << 1;
  VLOG(2) << "rhs stride: " << rhs.batch_stride << " " << rhs.leading_dim_stride
          << " " << 1;
  VLOG(2) << "out stride: " << output.batch_stride << " "
          << output.leading_dim_stride << " " << 1;
  VLOG(2) << "lhs trans: " << TransposeString(lhs.transpose);
  VLOG(2) << "rhs trans: " << TransposeString(rhs.transpose);

  auto params = CreateMatMulParams(batch_size, lhs, rhs, output);

  auto src_md = dnnl::memory::desc(params->a_dims, OneDnnType<InputT>(),
                                   params->a_strides);
  auto weights_md = dnnl::memory::desc(params->b_dims, OneDnnType<InputT>(),
                                       params->b_strides);
  auto dst_md = dnnl::memory::desc(params->c_dims, OneDnnType<OutputT>(),
                                   params->c_strides);
  auto bias_md =
      bias_data ? dnnl::memory::desc(params->bias_dims, OneDnnType<InputT>(),
                                     params->bias_strides)
                : dnnl::memory::desc();

  auto dnnl_engine = FindOrCreateEngine(stream_handle);
  dnnl::primitive_attr post_ops_attr;
  post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  // Set fp32 mode.
  dnnl::fpmath_mode fp32_math_mode = GetFP32MathMode();
  if (std::is_same<InputT, float>::value) {
    post_ops_attr.set_fpmath_mode(fp32_math_mode);
  }

  dnnl::post_ops post_ops = dnnl::post_ops();
  // C = activation(MatMul(x, w, bias) + beta * C)
  //   po.append_sum(beta)
  //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);
  CHECK(fabs(alpha - 1.0f) < 1e-6);
  if (c_data && fabs(beta - 0.0f) > 1e-6) post_ops.append_sum(beta);
  switch (epilogue) {
    case SYCLGemm::GemmBackendEpilogue::RELU:
    case SYCLGemm::GemmBackendEpilogue::BIAS_RELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
      break;
    case SYCLGemm::GemmBackendEpilogue::GELU:
    case SYCLGemm::GemmBackendEpilogue::BIAS_GELU:
      post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0, 0);
      break;
    case SYCLGemm::GemmBackendEpilogue::DEFAULT:
    case SYCLGemm::GemmBackendEpilogue::BIAS:
      break;
    default:
      return Internal("Unsupported Activation mode");
  }
  post_ops_attr.set_post_ops(post_ops);

  auto matmul_pd =
      bias_data
          ? std::make_shared<dnnl::matmul::primitive_desc>(
                dnnl_engine, src_md, weights_md, bias_md, dst_md, post_ops_attr)
          : std::make_shared<dnnl::matmul::primitive_desc>(
                dnnl_engine, src_md, weights_md, dst_md, post_ops_attr);
  std::unordered_map<int, dnnl::memory> fwd_primitive_args;

  size_t scratchpad_size = matmul_pd->scratchpad_desc().get_size();
  void* workspace;
  TF_RETURN_IF_ERROR(
      AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));

  auto scratchpad_mem =
      dnnl::memory(matmul_pd->scratchpad_desc(), dnnl_engine, workspace);

  auto matmul_primitive = dnnl::matmul(*matmul_pd);

  auto dnnl_stream = dnnl::sycl_interop::make_stream(
      dnnl_engine, *(stream_executor::gpu::AsGpuStreamValue(stream)));
  auto src_mem = CreateDnnlMemory(src_md, dnnl_engine, lhs_data);

  auto wei_mem = CreateDnnlMemory(weights_md, dnnl_engine, rhs_data);
  auto dst_mem = CreateDnnlMemory(dst_md, dnnl_engine, out_data);
  fwd_primitive_args.emplace(DNNL_ARG_SRC, src_mem);
  fwd_primitive_args.emplace(DNNL_ARG_WEIGHTS, wei_mem);
  fwd_primitive_args.emplace(DNNL_ARG_DST, dst_mem);
  fwd_primitive_args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem);
  if (bias_data) {
    auto bias_mem = CreateDnnlMemory(bias_md, dnnl_engine, bias_data);
    fwd_primitive_args.emplace(DNNL_ARG_BIAS, bias_mem);
  }
  matmul_primitive.execute(dnnl_stream, fwd_primitive_args);
  return absl::OkStatus();
}



/*
这是 DoGemm函数的完整声明
跟上述  DoOnednnGemm, DoXetlaGemm 函数的完整声明一样，返回类型是 absl::Status。
参数列表包含多个矩阵描述符、计算参数和上下文信。
用于执行GEMM（广义矩阵乘法）操作。根据输入的矩阵描述符、偏置、alpha和beta值等参数，选择使用Xetla或oneDNN后端来执行GEMM运算


根据算法选择来执行矩阵乘法计算的。它决定使用 Xetla 硬件加速的 GEMM（General Matrix Multiply）操作还是使用 oneDNN 库中的 GEMM 操作
  上面定义好的  两种矩阵执行操作，这里进行选择
  
  这段代码的核心功能是根据用户选择的算法类型，调用相应的函数来执行矩阵乘法计算：
  如果选择了 Xetla GEMM 算法，调用 DoXetlaGemm 函数。
  如果选择了其他算法（即默认使用 oneDNN），调用 DoOnednnGemm 函数。
  通过这种方式，代码能够根据不同的计算需求和硬件特性，灵活选择最合适的算法和实现方式，以优化矩阵乘法的性能和计算效率。
*/  

template <typename InputT, typename OutputT>
absl::Status DoGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                    const MatrixDescriptor& lhs, const MatrixDescriptor& rhs,
                    const MatrixDescriptor& c, const MatrixDescriptor& output,
                    se::DeviceMemoryBase bias, float alpha, float beta,
                    SYCLGemm::GemmBackendEpilogue epilogue, se::Stream* stream,
                    std::optional<se::blas::AlgorithmType> algorithm,
                    se::ScratchAllocator* scratch_allocator,
                    se::blas::ComputePrecision compute_precision) {
  if (algorithm == se::blas::kXetlaGemm) {
    VLOG(1) << "Run Xetla gemm kernel";
    return DoXetlaGemm<InputT>(batch_size, m, n, k, lhs, rhs, c, output, bias,
                               alpha, beta, epilogue, stream, algorithm,
                               scratch_allocator, compute_precision);
  } else {
    VLOG(1) << "Run OneDnn gemm kernel";
    return DoOnednnGemm<InputT, OutputT>(
        batch_size, m, n, k, lhs, rhs, c, output, bias, alpha, beta, epilogue,
        stream, algorithm, scratch_allocator, compute_precision);
  }
}



// 转置矩阵描述符
void TransposeMatrixDesc(MatrixDescriptor& matrix_desc) {
  matrix_desc.transpose =
      (matrix_desc.transpose == se::blas::Transpose::kNoTranspose)
          ? se::blas::Transpose::kTranspose
          : se::blas::Transpose::kNoTranspose;
}

// 这个函数确保矩阵描述符符合BLAS GEMM的要求，BLAS GEMM不支持转置的输出矩阵。通过使用数学恒等式 C^T = (A @ B)^T = B^T @ A^T，我们可以转换矩阵以满足要求。
void MakeBlasGemmCompatible(MatrixDescriptor& lhs, MatrixDescriptor& rhs,
                            MatrixDescriptor& output) {
  // BLAS GeMM doesn't support transposed output, but we can use the identity:
  // C^T = (A @ B)^T = B^T @ A^T.
  if (output.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs, rhs);
    TransposeMatrixDesc(lhs);
    TransposeMatrixDesc(rhs);
    TransposeMatrixDesc(output);
  }
}


void MakeBlasGemmCompatible(MatrixDescriptor& lhs, MatrixDescriptor& rhs,
                            MatrixDescriptor& c, MatrixDescriptor& output) {
  // BLAS GeMM doesn't support transposed output, but we can use the identity:
  // C^T = (A @ B)^T = B^T @ A^T.
  if (output.transpose == se::blas::Transpose::kTranspose) {
    std::swap(lhs, rhs);
    TransposeMatrixDesc(lhs);
    TransposeMatrixDesc(rhs);
    TransposeMatrixDesc(output);
    TransposeMatrixDesc(c);
  }
}
}  // namespace

/*
整个GEMM操作的入口点

这段代码定义了一个函数 RunGemm，用于运行矩阵乘法（GEMM，General Matrix Multiplication）。具体来说，它配置并执行矩阵乘法操作
RunGemm 函数根据提供的配置和输入参数，选择适当的矩阵乘法实现并执行操作。
函数首先初始化矩阵布局，提取矩阵维度和描述符，然后确保这些描述符与 BLAS 的 GEMM 操作兼容。
接着，通过检查操作数类型，选择并调用相应的 DoGemm 函数。如果没有匹配的类型组合，则返回错误状态
包括操作
1. 创建矩阵布局
2. 获取矩阵的维度
3. 获取矩阵描述符
4. 获取批量大小
5. 确保矩阵符合BLAS GEMM的要求
6. 获取操作数类型
7. 定义宏来处理不同类型的GEMM操作
8. 定义支持的类型组合
    TYPED_GEMM(BF16, BF16, BF16)
    TYPED_GEMM(F16, F16, F16)
    TYPED_GEMM(BF16, BF16, F32)
    TYPED_GEMM(F16, F16, F32)
    TYPED_GEMM(F32, F32, F32)
    TYPED_GEMM(S8, S8, S32)
9. 如果遇到不支持的类型组合，返回错误


*/
absl::Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
                     se::DeviceMemoryBase rhs_buffer,
                     se::DeviceMemoryBase c_buffer,
                     se::DeviceMemoryBase output_buffer,
                     se::DeviceMemoryBase bias_buffer, se::Stream* stream,
                     SYCLGemm::GemmBackendEpilogue epilogue,
                     se::ScratchAllocator* scratch_allocator) {
  VLOG(2) << "Executing a GemmThunk";

  auto lhs_layout = MatrixLayout{config.lhs_layout},
       rhs_layout = MatrixLayout{config.rhs_layout},
       output_layout = MatrixLayout{config.output_layout},
       c_layout = MatrixLayout{config.c_layout};

  int64_t m = output_layout.num_rows;
  int64_t n = output_layout.num_cols;
  int64_t k = lhs_layout.num_cols;
  MatrixDescriptor lhs = GetMatrixDesc(lhs_layout, lhs_buffer);
  MatrixDescriptor rhs = GetMatrixDesc(rhs_layout, rhs_buffer);
  MatrixDescriptor c = GetMatrixDesc(c_layout, c_buffer);
  MatrixDescriptor output = GetMatrixDesc(output_layout, output_buffer);
  int64_t batch_size = output_layout.batch_size;
  MakeBlasGemmCompatible(lhs, rhs, c, output);

  std::tuple operand_types{lhs_layout.dtype, rhs_layout.dtype,
                           output_layout.dtype};
#define TYPED_GEMM(ATYPE, BTYPE, CTYPE)                                       \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {                \
    using NativeAType = PrimitiveTypeToXetlaNative<ATYPE>::type;              \
    using NativeCType = PrimitiveTypeToXetlaNative<CTYPE>::type;              \
    return DoGemm<NativeAType, NativeCType>(                                  \
        batch_size, m, n, k, lhs, rhs, c, output, bias_buffer,                \
        config.alpha.real(), config.beta, epilogue, stream, config.algorithm, \
        scratch_allocator, config.compute_precision);                         \
  }

  TYPED_GEMM(BF16, BF16, BF16)
  TYPED_GEMM(F16, F16, F16)
  TYPED_GEMM(BF16, BF16, F32)
  TYPED_GEMM(F16, F16, F32)
  TYPED_GEMM(F32, F32, F32)
  TYPED_GEMM(S8, S8, S32)

#undef TYPED_GEMM
  return Internal(
      "Unexpected GEMM lhs type %s, rhs type %s and output type %s",
      primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
      primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype),
      primitive_util::LowercasePrimitiveTypeName(output_layout.dtype));
}

}  // namespace gpu
}  // namespace xla
