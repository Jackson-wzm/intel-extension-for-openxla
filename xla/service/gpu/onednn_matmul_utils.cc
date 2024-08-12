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

/*
API文档  Onednn: https://oneapi-src.github.io/oneDNN/struct_dnnl_convolution_backward_data_primitive_desc.html

example，告诉如何使用这些API的,看具体如何使用 : OneDNN code：
https://github.com/oneapi-src/oneDNN/blob/main/examples/primitives/matmul.cpp

onednn实现的matmul的方法要看initialize的时候选用了什么算法，然后不同的算法会call不同的实现，所有的实现方式都在 onednn repo 里面

下面解析 OneDNN api 文档的内容
struct dnnl::matmul::primitive_desc



*/
/*  
常规概念
1. dnnl::memory

    dnnl::memory 是 DNNL（Deep Neural Network Library，现在称为 oneDNN） 中一个重要的类，它用于表示和管理内存资源。
    oneDNN 是一个高性能的深度学习加速库，广泛应用于深度学习框架和应用程序中。 dnnl::memory 类的主要用途是描述内存布局和管理内存数据。
    以下是 dnnl::memory 的详细解释和使用方法：
    dnnl::memory 基本概念
    内存描述符（Memory Descriptor）：描述内存的布局，包括数据类型、维度、步长等。
    内存对象（Memory Object）：表示实际的内存资源，可以通过内存描述符进行创建。
    
    dnnl::memory 的主要成员函数和用法
    1. 构造函数
    创建一个内存对象，通常需要一个引擎（engine）和一个内存描述符。

    // 定义一个引擎，指定设备类型（CPU 或 GPU）
    dnnl::engine eng(dnnl::engine::kind::cpu, 0);

    // 定义一个内存描述符
    dnnl::memory::dims dimensions = {batch, channels, height, width};
    dnnl::memory::desc mem_desc(dimensions, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);

    // 创建一个内存对象
    dnnl::memory mem(mem_desc, eng);

    2. 访问内存数据
    可以通过 get_data_handle 和 set_data_handle 函数访问和设置内存数据的指针。
    // 获取数据指针
    float* data_ptr = static_cast<float*>(mem.get_data_handle());

    // 设置数据指针
    mem.set_data_handle(data_ptr);

    3. 内存描述符（Memory Descriptor）
    内存描述符是 dnnl::memory 的一个重要组成部分，它描述了内存的布局，包括数据类型、维度和步长。
    // 定义内存描述符
    dnnl::memory::desc mem_desc(dimensions, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);

    示例代码
    以下是一个完整的示例代码，展示了如何创建一个内存对象并访问其数据：
    #include <iostream>
    #include <dnnl.hpp>

    int main() {
        // 定义引擎和流
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream s(eng);

        // 定义内存描述符和内存对象
        dnnl::memory::dims dimensions = {1, 3, 224, 224}; // {batch, channels, height, width}
        dnnl::memory::desc mem_desc(dimensions, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw);
        dnnl::memory mem(mem_desc, eng);

        // 获取和设置数据指针
        float* data_ptr = static_cast<float*>(mem.get_data_handle());
        for (int i = 0; i < 1 * 3 * 224 * 224; ++i) {
            data_ptr[i] = static_cast<float>(i);
        }
        mem.set_data_handle(data_ptr);

        // 打印部分数据以验证
        for (int i = 0; i < 10; ++i) {
            std::cout << data_ptr[i] << " ";
        }
        std::cout << std::endl;

        return 0;
    }

    总结
    dnnl::memory 是 oneDNN 库中的一个核心类，用于表示和管理内存资源。通过内存描述符可以定义内存布局，
    并通过内存对象创建和访问实际的内存数据。使用 dnnl::memory 可以方便地在深度学习应用中管理和操作数据，提高计算性能。


*/

namespace xla {
  namespace gpu {

    namespace SYCLGemm{

        //这段代码的功能是将一个字符串（epilogue）转换为对应的 GemmBackendEpilogue 枚举类型。如果字符串匹配某个枚举值，则返回对应的枚举值；
        //如果不匹配，则返回一个错误状态

        absl::StatusOr<GemmBackendEpilogue> EpilogueCast(std::string& epilogue){
          absl::StatusOr<GemmBackendEpilogue>:    //函数返回类型。它表示返回值可能是 GemmBackendEpilogue 类型的值，或者是一个错误状态（absl::Status）。
          std::string& epilogue:                  //参数类型。表示传入的是一个字符串的引用，用于传递待转换的字符串。
          std::string& epilogue:                  // 参数类型。表示传入的是一个字符串的引用，用于传递待转换的字符串。
          absl::StatusOr                          //用于函数返回值的错误处理，确保函数能够优雅地处理错误情况。
          /*
          条件判断 用于将输入字符串转换为相应的枚举值。
          错误处理 确保当输入字符串不匹配任何已知值时，返回一个描述错误的状态

          absl::StatusOr 是 Google 的 Abseil 库中的一个模板类，用于返回一个状态或者一个值。
          这种模式常用于函数的返回值，以便在出错时不仅能返回错误状态，还能携带相应的错误信息，同时在成功时返回期望的值。

          类定义与使用
          absl::StatusOr<T> 可以被看作是一个联合类型，要么包含一个 absl::Status 对象，要么包含一个类型为 T 的值。
          这样设计的好处是让函数可以返回详细的错误信息，而不需要依赖异常。

          */
          
          //条件判断和返回枚举值:

            if(epilogue == "DEFAULT"){
                return GemmBackendEpilogue::DEFAULT;   //如果 epilogue 字符串等于 "DEFAULT"，则返回枚举值 GemmBackendEpilogue::DEFAULT。
            }else if(epilogue == "RELU"){
                return GemmBackendEpilogue::RELU;      //如果 epilogue 字符串等于 "RELU"，则返回枚举值 GemmBackendEpilogue::RELU。
            }else if(epilogue == "GELU"){
                return GemmBackendEpilogue::GELU;      //这些条件判断确保了输入的字符串能够匹配相应的枚举值，并返回正确的 GemmBackendEpilogue 枚举类型。
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
                return Internal("Unknown Epilogue.");  //如果传入的 epilogue 字符串不匹配任何已知的枚举值，函数将返回一个错误状态，表示内部错误，并包含错误信息 "Unknown Epilogue."。
            }
        }

        absl::StatusOr<std::string> EpilogueCast(GemmBackendEpilogue epilogue){
              /*
              这段代码实现了一个将 GemmBackendEpilogue 枚举值转换为对应字符串表示的函数 EpilogueCast。
              该函数使用 absl::StatusOr 作为返回类型，
              以确保可以处理错误情况
              absl::StatusOr<std::string>: 返回类型，表示函数的返回值可能是一个 std::string 或一个错误状态。
              GemmBackendEpilogue epilogue: 参数类型，表示传入的 GemmBackendEpilogue 枚举值

              absl::StatusOr<std::string>:
                StatusOr 是一种用于返回值或错误状态的类型。它要么包含一个有效的 std::string 值，要么包含一个错误状态。
                这有助于在函数返回时处理错误，而不是使用异常或错误代码。

              if-else 语句:

                if (epilogue == GemmBackendEpilogue::DEFAULT): 检查 epilogue 是否等于 GemmBackendEpilogue::DEFAULT，如果是，则返回字符串 "DEFAULT"。
                其他 else if 分支类似，根据 epilogue 的值返回对应的字符串表示。GemmBackendEpilogue 包含的返回值类型的实现 在 onednn_matmul_utils.h 文件里面进行定义。
              
              absl::StatusOr 用于返回值的错误处理，确保函数能够优雅地处理错误情况。
              if-else 语句 用于将输入的 GemmBackendEpilogue 枚举值映射到对应的字符串表示。
              错误处理 确保当输入的枚举值不匹配任何已知值时，返回一个描述错误的状态。
            */

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

      absl::StatusOr<bool> EpilogueAddsVectorBias(GemmBackendEpilogue epilogue) {  //函数签名
        /*这段代码定义了一个函数 EpilogueAddsVectorBias，用于确定给定的 GemmBackendEpilogue 枚举值是否会添加向量偏置（vector bias）。
        //该函数返回一个 absl::StatusOr<bool> 类型，以便能够处理潜在的错误情况
        //通过这种方式，函数 EpilogueAddsVectorBias 可以根据 GemmBackendEpilogue 枚举值确定是否添加向量偏置，并处理未知的枚举值错误。
        
        为什么要 添加向量偏置？
            向量偏置（vector bias）在机器学习和深度学习中起着至关重要的作用。它主要用于神经网络中的神经元或单元，以帮助模型更好地拟合数据。
            以下是添加向量偏置的一些关键原因和作用：
            1. 增加模型的表达能力
            偏置项允许模型在没有输入信号的情况下产生非零输出。没有偏置，模型的输出始终与输入的线性组合成比例，限制了模型的表达能力。
            偏置项相当于在输入数据中增加一个维度，这个维度总是等于1，使得模型能够更好地拟合数据。

            2. 解决数据偏移问题
            许多实际数据集并不以零为中心，有偏置项可以帮助模型在初始状态下更好地拟合数据的实际分布。例如，如果所有输入特征的值都很大，
            那么没有偏置项的模型可能无法正确捕捉到数据的趋势。

            3. 增加模型的灵活性
            偏置项允许神经元激活函数在更广泛的输入值范围内有效工作。它使得神经元的激活函数可以平移，从而在模型训练过程中更容易找到最佳的参数设置。

            4. 改善梯度下降的性能
            在模型训练过程中，偏置项能够加快收敛速度并改善梯度下降的性能。它使得模型在权重调整过程中更加灵活，从而更快地找到全局最优解。
        
        函数签名
          absl::StatusOr<bool>: 返回类型，表示函数的返回值可能是一个布尔值 bool 或一个错误状态。这有助于在函数返回时处理错误，而不是使用异常或错误代码
          GemmBackendEpilogue epilogue: 参数类型，表示传入的 GemmBackendEpilogue 枚举值。
        */

        // 函数具体实现
          switch (epilogue) {                        //switch (epilogue) { ... } 用于根据 epilogue 的值执行不同的代码块。
              case GemmBackendEpilogue::DEFAULT:
              case GemmBackendEpilogue::RELU:
              case GemmBackendEpilogue::GELU:
              case GemmBackendEpilogue::GELU_AUX:
                  return false;
                                                      //case GemmBackendEpilogue::DEFAULT::
                                                      //处理 GemmBackendEpilogue::DEFAULT 枚举值。
                                                      //继续处理 case GemmBackendEpilogue::RELU: 等。
                                                      //如果 epilogue 是 DEFAULT, RELU, GELU, 或 GELU_AUX，返回 false。
              case GemmBackendEpilogue::BIAS:
              case GemmBackendEpilogue::BIAS_RELU:
              case GemmBackendEpilogue::BIAS_GELU:
              case GemmBackendEpilogue::BIAS_GELU_AUX:
                  return true;
                                                      // case GemmBackendEpilogue::BIAS::
                                                      //处理 GemmBackendEpilogue::BIAS 枚举值。
                                                      // 继续处理 case GemmBackendEpilogue::BIAS_RELU: 等。
                                                      // 如果 epilogue 是 BIAS, BIAS_RELU, BIAS_GELU, 或 BIAS_GELU_AUX，返回 true。

              //错误处理
              default:
                  return Internal("Unknown Epilogue.");
                  //如果 epilogue 的值不属于任何已知的 GemmBackendEpilogue 枚举值，返回一个错误状态。
                  //使用 absl::InternalError("Unknown Epilogue.") 创建错误状态。
          }
      }

      //函数签名
      absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendEpilogue epilogue) {
          /*
          定义了一个函数 EpilogueHasAuxiliaryOutput，用于确定给定的 GemmBackendEpilogue 枚举值是否具有辅助输出。
          该函数返回一个 absl::StatusOr<bool> 类型，以便能够处理潜在的错误情况
          absl::StatusOr<bool>: 返回类型，表示函数的返回值可能是一个布尔值 bool 或一个错误状态。这有助于在函数返回时处理错误，而不是使用异常或错误代码。
          GemmBackendEpilogue epilogue: 参数类型，表示传入的 GemmBackendEpilogue 枚举值

          absl::StatusOr 用于返回值的错误处理，确保函数能够优雅地处理错误情况。
          switch 语句 用于根据 GemmBackendEpilogue 枚举值决定是否返回 true 或 false。
          错误处理 确保当输入的枚举值不匹配任何已知值时，返回一个描述错误的状态。
          
          */
          


      //函数具体实现
          switch (epilogue) {                      //switch (epilogue) { ... } 用于根据 epilogue 的值执行不同的代码块。
              case GemmBackendEpilogue::DEFAULT:
              case GemmBackendEpilogue::RELU:
              case GemmBackendEpilogue::GELU:
              case GemmBackendEpilogue::BIAS:
              case GemmBackendEpilogue::BIAS_RELU:
              case GemmBackendEpilogue::BIAS_GELU:
                  return false;
                                                  //case GemmBackendEpilogue::DEFAULT::
                                                  //处理 GemmBackendEpilogue::DEFAULT 枚举值。
                                                  //继续处理 case GemmBackendEpilogue::RELU: 等。
                                                  //如果 epilogue 是 DEFAULT, RELU, GELU, BIAS, BIAS_RELU, 或 BIAS_GELU，返回 false。

              case GemmBackendEpilogue::GELU_AUX:
              case GemmBackendEpilogue::BIAS_GELU_AUX:
                  return true;
                                                  //处理 GemmBackendEpilogue::GELU_AUX 枚举值。
                                                  //继续处理 case GemmBackendEpilogue::BIAS_GELU_AUX:。
                                                  //如果 epilogue 是 GELU_AUX 或 BIAS_GELU_AUX，返回 true。
              default:
                return Internal("Unknown Epilogue.");
                                  default::
                                  //如果 epilogue 的值不属于任何已知的 GemmBackendEpilogue 枚举值，返回一个错误状态。
                                  //使用 absl::InternalError("Unknown Epilogue.") 创建错误状态。
          }
      }

      absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(GemmBackendConfig_Epilogue epilogue) {
        /*
        定义了一个名为 AsSYCLEpilogue 的函数，用于将 GemmBackendConfig_Epilogue 枚举值转换为 GemmBackendEpilogue 枚举值。
        该函数返回一个 absl::StatusOr<GemmBackendEpilogue> 类型，以便能够处理潜在的错误情况

          在 .h 文件里面定义的是 
          enum class GemmBackendEpilogue{      枚举类 GemmBackendEpilogue 定义了一系列与矩阵乘法后处理 (epilogue) 相关的选项，比如默认处理、RELU、GELU、偏置等。
            DEFAULT,
            RELU,
            GELU,
            BIAS,  偏差
            BIAS_RELU,
            BIAS_GELU,
            GELU_AUX,
            BIAS_GELU_AUX,
        };

        */

        //absl::StatusOr<GemmBackendEpilogue>: 返回类型，表示函数的返回值可能是一个 GemmBackendEpilogue 枚举值或一个错误状态
        //GemmBackendConfig_Epilogue epilogue: 参数类型，表示传入的 GemmBackendConfig_Epilogue 枚举值。

            switch (epilogue) {
              case GemmBackendConfig::DEFAULT:           处理 GemmBackendConfig::DEFAULT 枚举值。
                return GemmBackendEpilogue::DEFAULT;     返回 GemmBackendEpilogue::DEFAULT。
              case GemmBackendConfig::RELU:              处理 GemmBackendConfig::RELU 枚举值。
                return GemmBackendEpilogue::RELU;        返回 GemmBackendEpilogue::RELU
              case GemmBackendConfig::GELU:              处理 GemmBackendConfig::GELU 枚举值
                return GemmBackendEpilogue::GELU;        返回 GemmBackendEpilogue::GELU。
              case GemmBackendConfig::GELU_AUX:          处理 GemmBackendConfig::GELU_AUX 枚举值。
                return GemmBackendEpilogue::GELU_AUX;    返回 GemmBackendEpilogue::GELU_AUX。
              case GemmBackendConfig::BIAS:
                return GemmBackendEpilogue::BIAS;
              case GemmBackendConfig::BIAS_RELU:
                return GemmBackendEpilogue::BIAS_RELU;
              case GemmBackendConfig::BIAS_GELU:
                return GemmBackendEpilogue::BIAS_GELU;
              case GemmBackendConfig::BIAS_GELU_AUX:
                return GemmBackendEpilogue::BIAS_GELU_AUX;
              default:
                return Internal("Unsupported Epilogue.");   如果 epilogue 的值不属于任何已知的 GemmBackendConfig_Epilogue 枚举值，返回一个错误状态。
            }
      }
    } // namespace SYCLGemm



    // Returns the xetla native type (eg, float) corresponding to 相应的 the given template parameter 模板参数 XLA primitive type 原始类型 (eg, F32).
    
    /*
    模板结构和类型定义
    定义模板结构 PrimitiveTypeToXetlaNative，根据不同的PrimitiveType返回相应的Xetla原生类型,数据结构。
    1. 将 F32 类型映射为 float 类型、
    2. 将 F16 类型映射为 sycl::half 类型
    3. 将 BF16 类型映射为 ::gpu::xetla::bf16 类型。
    4. 将 S8 类型映射为 int8_t 类型。
    5. 将 S32 类型映射为 int32_t 类型。

    通过这种模板特化，我们可以轻松地将某个 PrimitiveType 映射到相应的 Xetla 本地类型

    如果 要使用 Xetla 进行计算， 那么就需要把 常规的数据转变为 符合 Xetla库 种张量操作数据格式，才能被计算
    */ 

    基础模板声明
    template <PrimitiveType>  声明了一个模板，其中 PrimitiveType 是模板参数。
    struct PrimitiveTypeToXetlaNative;  定义了一个结构模板 PrimitiveTypeToXetlaNative，此结构将被具体化（特化）以处理不同的 PrimitiveType


    特化模板
    接下来是对 PrimitiveTypeToXetlaNative 进行特化处理，以定义不同的 PrimitiveType 映射到的 Xetla 本地类型

    特化模板：F32
    template <>                                         表示这是一个特化模板。
    struct PrimitiveTypeToXetlaNative<F32> {            特化模板用于 F32 类型
      using type = float;                               将 F32 类型映射为 float 类型
    };

    特化模板：F16
    template <>
    struct PrimitiveTypeToXetlaNative<F16> {
      using type = sycl::half;                     将 F16 类型映射为 sycl::half 类型
    };

    特化模板：BF16
    template <>
    struct PrimitiveTypeToXetlaNative<BF16> {
      using type = ::gpu::xetla::bf16;            将 BF16 类型映射为 ::gpu::xetla::bf16 类型。
    }; 

    特化模板：S8
    template <>
    struct PrimitiveTypeToXetlaNative<S8> {       将 S8 类型映射为 int8_t 类型。
      using type = int8_t;
    };

    特化模板：S32
    template <>
    struct PrimitiveTypeToXetlaNative<S32> {      将 S32 类型映射为 int32_t 类型。
      using type = int32_t;
    };

    //示例使用
    // 通过这种模板特化，我们可以轻松地将某个 PrimitiveType 映射到相应的 Xetla 本地类型。例如：
    PrimitiveTypeToXetlaNative<F32>::type myFloat;  // myFloat 是 float 类型
    PrimitiveTypeToXetlaNative<F16>::type myHalf;   // myHalf 是 sycl::half 类型
    PrimitiveTypeToXetlaNative<S8>::type myInt8;    // myInt8 是 int8_t 类型

    总结
    模板声明: 定义了一个通用的模板结构 PrimitiveTypeToXetlaNative。
    模板特化: 为每种 PrimitiveType 提供特化定义，将其映射到具体的 Xetla 本地类型。
    使用方便: 通过这种映射，可以在编译期确定具体的类型，从而简化类型处理逻辑


    /// Return oneDNN data type (memory::data_type) for input type T
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

    //模板函数声明
    template <typename T>   这行代码表明 OneDnnType 是一个模板函数，其中 T 是一个模板参数，可以是任何类型。
    inline dnnl::memory::data_type OneDnnType();

      /*
      inline: 关键字 inline 提示编译器在调用这个函数时将其内联展开，从而避免函数调用的开销。不过，具体是否内联由编译器决定。
      dnnl::memory::data_type: 这是函数的返回类型。它表示这个函数将返回一个 dnnl::memory::data_type 类型的值。
      dnnl::memory::data_type 是 OneDNN 库中表示数据类型的枚举类型。
      OneDnnType();: 这是函数的名称  OneDnnType，后面跟着空的参数列表，表示这个函数没有参数。

      OneDnnType 模板函数的主要目的是将 C++ 类型 T 映射到 OneDNN 库中对应的 data_type
      
      Instantiation for float type. Add similar instantiations for other type if needed.
      浮点类型的实例。如果需要，为其他类型添加类似的实例。
      */

    //模板特化: 模板特化为特定的数据类型提供了具体的实现。以下是对几种特定类型的模板特化
    template <>
    inline dnnl::memory::data_type OneDnnType<float>() {
      return dnnl::memory::data_type::f32;
    }
      /*
      template <>: 表示这是一个模板特化。
      inline dnnl::memory::data_type OneDnnType<float>(): 特化版本的 OneDnnType 函数，专门用于 float 类型。
      return dnnl::memory::data_type::f32: 返回 OneDNN 库中与 float 对应的数据类型 f32。
      内联函数: inline 关键字提示编译器尽可能内联展开函数，以减少函数调用开销
      特化实现: 通过为特定类型（如 float、int8_t 等）特化这个模板函数，可以实现从 C++ 基本类型到 OneDNN 数据类型的映射。
      */
    template <>
    inline dnnl::memory::data_type OneDnnType<double>() {   // 对应 double 类型，返回 OneDNN 中的 f64
      return dnnl::memory::data_type::f64;
    }

    //函数定义: inline dnnl::memory::data_type OneDnnType<double>() 定义了一个特化的模板函数，专门用于 double 类型。
    //返回值: return dnnl::memory::data_type::f64; 返回 dnnl::memory::data_type::f64，表示 double 类型在 OneDNN 中对应的类型是 f64。

    template <>
    inline dnnl::memory::data_type OneDnnType<sycl::half>() {   // 对应 sycl::half 类型，返回 OneDNN 中的 f16。
      return dnnl::memory::data_type::f16;
    }

    template <>
    inline dnnl::memory::data_type OneDnnType<int8_t>() {       // 对应 int8_t 类型，返回 OneDNN 中的 s8。
      return dnnl::memory::data_type::s8;
    }

    template <>
    inline dnnl::memory::data_type OneDnnType<int32_t>() {      // 对应 int32_t 类型，返回 OneDNN 中的 s32。
      return dnnl::memory::data_type::s32;
    }

    template <>
    inline dnnl::memory::data_type OneDnnType<::gpu::xetla::bf16>() {   // 对应 ::gpu::xetla::bf16 类型，返回 OneDNN 中的 bf16。
      return dnnl::memory::data_type::bf16;
    }

    
    namespace {
          /*
          在C++中，namespace用于声明一个命名空间，它主要用于防止命名冲突，将标识符（如变量、函数、类等）组织在不同的命名空间中，使代码更加清晰、模块化和可维护。
          命名空间的作用
          防止命名冲突：在大型项目中，不同的模块或库可能会使用相同的标识符。如果不使用命名空间，就可能导致命名冲突。命名空间通过将标识符组织在不同的作用域中，避免了这种冲突。
          代码组织：命名空间有助于将相关的代码组织在一起，使代码结构更加清晰。例如，可以将所有与数学相关的函数放在一个math命名空间中，
          将所有与图形相关的函数放在一个graphics命名空间中。
          提升代码可读性：使用命名空间可以使代码的意图更加明确，读者可以通过命名空间名称快速理解代码的功能和用途。
                    
          */

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
              // layout: 这是一个 MatrixLayout 类型的对象，描述了矩阵的布局。
              // data: 这是一个 se::DeviceMemoryBase 类型的对象，表示设备内存中矩阵的数据。                                  

            bool transpose = layout.order == MatrixLayout::Order::kColumnMajor;  检查矩阵的布局顺序是否为列主序。如果是列主序，则需要转置矩阵。
            return MatrixDescriptor{
                data,                                              直接传入的矩阵数据。
                transpose ? se::blas::Transpose::kTranspose        
                          : se::blas::Transpose::kNoTranspose,     根据是否需要转置设置相应的转置选项
                transpose ? layout.num_cols : layout.num_rows,     如果转置，则行数为列数，否则为行数。
                transpose ? layout.num_rows : layout.num_cols,     如果转置，则列数为行数，否则为列数。
                layout.batch_stride,                               批处理的步幅
                layout.leading_dim_stride,                         主维度的步幅
            }; 
          }


            //结构体 OneDnnMatMulParams
          struct OneDnnMatMulParams {           
            /*
            定义一个名为 OneDnnMatMulParams 的结构体，用于存储矩阵乘法的参数。包含了矩阵A、B、C和偏置的维度和步幅信息
            这个结构体 OneDnnMatMulParams 主要用于存储矩阵乘法操作所需的各种参数，包括矩阵的维度和步长。构造函数通过参数初始化列表来高效地初始化这些成员变量。

            dnnl::memory 是 DNNL（Deep Neural Network Library，现在称为 oneDNN） 中一个重要的类，它用于表示和管理内存资源。
            oneDNN 是一个高性能的深度学习加速库，广泛应用于深度学习框架和应用程序中。
            dnnl::memory 类的主要用途是描述内存布局和管理内存数据。以下是 dnnl::memory 的详细解释和使用方法：
            dnnl::memory 基本概念
            内存描述符（Memory Descriptor）：描述内存的布局，包括数据类型、维度、步长等。
            内存对象（Memory Object）：表示实际的内存资源，可以通过内存描述符进行创建。
            
            */

        //定义一个成员变量，用于存储
          dnnl::memory::dims a_dims;          // 矩阵 A 的维度  定义一个成员变量 a_dims，类型为 dnnl::memory::dims，用于存储矩阵 A 的维度
          dnnl::memory::dims b_dims;          // 矩阵 B 的维度
          dnnl::memory::dims c_dims;          // 矩阵 C 的维度
          dnnl::memory::dims bias_dims;       // 偏置矩阵的维度
          dnnl::memory::dims a_strides;       // 矩阵 A 的步长
          dnnl::memory::dims b_strides;       // 矩阵 B 的步长
          dnnl::memory::dims c_strides;       // 矩阵 C 的步长
          dnnl::memory::dims bias_strides;    // 偏置矩阵的步长

          // 构造函数，初始化上述所有成员变量。定义一个构造函数，接受多个 dnnl::memory::dims 类型的参数，用于初始化结构体的成员变量。
          OneDnnMatMulParams(dnnl::memory::dims a_dims, dnnl::memory::dims b_dims,
                            dnnl::memory::dims c_dims, dnnl::memory::dims bias_dims,
                            dnnl::memory::dims a_strides, dnnl::memory::dims b_strides,
                            dnnl::memory::dims c_strides,
                            dnnl::memory::dims bias_strides)
              : a_dims(std::move(a_dims)),        // 使用传入的参数 a_dims 初始化成员变量 a_dims。这里使用了 std::move 来避免不必要的拷贝。
                b_dims(std::move(b_dims)),        // 使使用传入的参数 b_dims 初始化成员变量 b_dims。
                c_dims(std::move(c_dims)),        // 使用传入的参数初始化 c_dims
                bias_dims(std::move(bias_dims)),  // 使用传入的参数初始化 bias_dims
                a_strides(std::move(a_strides)),  // 使用传入的参数初始化 a_strides
                b_strides(std::move(b_strides)),  // 使用传入的参数初始化 b_strides
                c_strides(std::move(c_strides)),  // 使用传入的参数初始化 c_strides
                bias_strides(std::move(bias_strides)) {}  // 使用传入的参数初始化 bias_strides
            };



          /*  
          这段代码定义模板函数 RunXetlaGemm，用于运行 Xetla GEMM（通用矩阵乘法）操作。这个函数基于模板类型 InputT 进行实例化，
          并根据提供的后处理类型（epilogue）来配置和运行 GEMM 操作。
          
          */

          template <typename InputT>      //表示这是一个模板特化。参数 InputT 表示矩阵元素的类型

          std::enable_if_t<std::is_same_v<InputT, ::gpu::xetla::bf16> ||
                              std::is_same_v<InputT, sycl::half>,
                          absl::StatusOr<bool>>
          /*
          std::enable_if_t 是 C++11 引入的一个特性，用于在模板编程中进行条件编译。它的作用是启用或禁用模板的实例化，具体取决于一个布尔条件。
          std::enable_if_t 是 std::enable_if 的一种简化形式，它使用起来更方便。
          std::enable_if 是在 <type_traits> 头文件中定义的，它有两个模板参数：
          一个布尔条件 B。一个类型 T，默认值是 void。
          如果 B 为 true，std::enable_if 会定义一个类型 type，它等同于 T。如果 B 为 false，则 std::enable_if 不会定义 type。
          这种机制可以用于在特定条件下启用或禁用某些模板。

          std::enable_if_t 是 C++14 引入的 std::enable_if 的简化版本，它等同于 typename std::enable_if<B, T>::type，但更简洁。
          条件启用模板函数:

          std::enable_if_t<std::is_integral_v<T>, void> 作为函数返回类型。这意味着只有当 T 是整数类型时，std::enable_if_t 才会定义一个类型 void，从而使得这个函数模板有效。
          如果 T 不是整数类型，std::enable_if_t 将不定义任何类型，导致编译错误，从而禁用这个模板实例化。
          条件启用成员函数:

          同样的原理被应用于类模板的成员函数，使得某些成员函数只有在特定条件下才会被实例化。
          std::enable_if_t<std::is_floating_point_v<T>, void> 和 std::enable_if_t<std::is_integral_v<T>, void> 用于有条件地实例化 doSomething 函数。

          上述代码

          使用 std::enable_if_t 限制模板参数 InputT，只能是 ::gpu::xetla::bf16 或 sycl::half。
                    返回类型是 absl::StatusOr<bool>，表示函数要么返回一个状态对象（用于错误处理），要么返回一个布尔值
          

          */
          问题： 支持的数据类型 是不是就是从这里 控制的？ 只能是 ::gpu::xetla::bf16 或 sycl::half。


        RunXetlaGemm(se::gpu::GpuStreamHandle handle, const MatrixDescriptor& lhs,
                  const MatrixDescriptor& rhs, const MatrixDescriptor& c,
                  const MatrixDescriptor& out, se::DeviceMemoryBase bias,
                  SYCLGemm::GemmBackendEpilogue epilogue, float beta) 
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
                  
        {
        void* bias_data = const_cast<void*>(bias.opaque());   // 将 bias 和 c 的数据指针转换为非 const 的 void* 指针，以便在后续操作中使用。
        void* c_data = const_cast<void*>(c.data.opaque());


        //根据 返回的 epilogue 类型选择不同的后处理策略， 
        switch (epilogue) {

          // 如果 epilogue 为 DEFAULT，则不进行额外的后处理
          case SYCLGemm::GemmBackendEpilogue::DEFAULT: {  

            auto policy = ::gpu::xetla::XetlaGemmKernel<InputT>()
                              .add_matrix_c(out)
                              .add_matrix_a(lhs)
                              .add_matrix_b(rhs)
                              .build();
            /*  
            创建一个 XetlaGemmKernel 实例，并配置 out, lhs 和 rhs 矩阵。
            使用一个名为 XetlaGemmKernel 的模板类来构建一个称为 policy 的对象。这个对象将用于执行通用矩阵乘法（GEMM）操作
            
            模板类 XetlaGemmKernel<InputT>:
            XetlaGemmKernel 是一个模板类，它使用模板参数 InputT。这个模板参数 InputT， 上面定义好了  指定矩阵元素的数据类型只能是 float、sycl::half 等。
            
            调用构造函数 XetlaGemmKernel<InputT>():
            这部分代码调用了 XetlaGemmKernel 类的默认构造函数，创建一个 XetlaGemmKernel<InputT> 类型的临时对象。
            
            方法 add_matrix_c(out):
            这是一个链式调用的方法。add_matrix_c 方法将输出矩阵 out 添加到 XetlaGemmKernel 对象中。out 是一个 MatrixDescriptor 类型的对象，描述了输出矩阵的布局和数据。
            
            方法 add_matrix_a(lhs):
            另一个链式调用的方法。add_matrix_a 方法将左操作数矩阵 lhs 添加到 XetlaGemmKernel 对象中。lhs 也是一个 MatrixDescriptor 类型的对象，描述了左操作数矩阵的布局和数据。
            
            方法 add_matrix_b(rhs):
            另一个链式调用的方法。add_matrix_b 方法将右操作数矩阵 rhs 添加到 XetlaGemmKernel 对象中。rhs 同样是一个 MatrixDescriptor 类型的对象，描述了右操作数矩阵的布局和数据。
            
            方法 build():
            最后一个链式调用的方法。build 方法用于完成配置并返回一个最终的 XetlaGemmKernel 对象，这个对象包含了所有配置好的信息（矩阵 C、A 和 B）。
            
            自动类型推导 auto:
            auto 关键字用于自动推导 policy 对象的类型。policy 的类型将是 XetlaGemmKernel<InputT>，即带有所有添加的矩阵配置的 XetlaGemmKernel 对象。
            综合起来，这行代码创建并配置了一个 XetlaGemmKernel 对象，该对象将用于执行矩阵乘法操作。具体地，它将 out 作为输出矩阵，lhs 作为左操作数矩阵，rhs 作为右操作数矩阵，最后调用 build 方法来完成配置并生成最终的 policy 对象。
            
            
            代码分析总结
            模板类 XetlaGemmKernel<InputT>:
                负责处理矩阵乘法的类。
                InputT 是矩阵元素的类型。
            
            链式方法调用:
                add_matrix_c(out): 添加输出矩阵 out。
                add_matrix_a(lhs): 添加左操作数矩阵 lhs。
                add_matrix_b(rhs): 添加右操作数矩阵 rhs。
            
            方法 build():生成并返回一个配置完毕的 XetlaGemmKernel 对象。
            
            自动类型推导 auto: 自动推导出 policy 的类型。
            这行代码的目的是创建一个配置好的 XetlaGemmKernel 对象，用于执行矩阵乘法操作
            
            */

            if (fabs(beta) - 0.0f > 1e-6) {     //这行代码判断 beta 是否为接近于 0 的数值。如果 beta 不等于 0，则需要考虑 beta 对 c 矩阵的影响。
              if (fabs(beta) - 1.0f < 1e-6) {   // 如果 beta 等于 1，则添加一个加和后处理（RES_ADD）
                policy                          // 如果 beta 等于 1，调用 add_epilogue 方法，并传入 c_data 和 RES_ADD 类型，最后构建 policy 对象。
                    .add_epilogue(
                        c_data,
                        ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::RES_ADD)
                    .build();  
              } else {
                return true;  // 如果 beta 不是 0 或 1，直接返回 true，表示不支持这种情况。
              }
            }

            if (policy.fallback() == false) { // 检查是否有回退策略（fallback）
              return !policy.run(handle);  // 如果没有回退（fallback），则运行 policy，返回 !policy.run(handle) 的结果。
            }
            return policy.fallback();  //如果发生回退，返回 policy.fallback() 的结果。
          }
          

          case SYCLGemm::GemmBackendEpilogue::BIAS: {  // 如果 epilogue 为 BIAS，则配置偏置矩阵
            
            auto policy =
                ::gpu::xetla::XetlaGemmKernel<InputT>()
                    .add_matrix_c(out)
                    .add_matrix_a(lhs)
                    .add_matrix_b(rhs)
                    .add_epilogue(
                        bias_data,
                        ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::BIAS)
                    .build();
            /*  
            创建 XetlaGemmKernel 实例，并添加 bias 矩阵作为后处理。
            
            */
           
            if (fabs(beta) - 0.0f > 1e-6) {  //如果 beta 不等于 0，添加加和后处理（RES_ADD）。
              policy
                  .add_epilogue(
                      c_data,
                      ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::RES_ADD,
                      beta)
                  .build();

                  /*
                  1. add_epilogue(...)
                  add_epilogue 是一个方法调用，通常出现在一个具有链式接口的对象上。它表示在主操作（如矩阵乘法）之后添加一个尾随操作（epilogue）。
                  在深度学习和数值计算中，epilogue常用于在矩阵乘法之后添加一些操作，比如激活函数、加法、缩放等。
                  该方法通常属于一个构建器对象（builder object）或某种支持链式操作的对象。
                  2. c_data
                  c_data 是传递给 add_epilogue 的第一个参数，可能是结果矩阵或缓冲区的数据指针或引用。
                  这个参数通常是操作的目标，即epilogue操作会对c_data进行修改或在其上执行计算。
                  3. ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::RES_ADD
                  这是传递给 add_epilogue 的第二个参数。
                  ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType 表示 XetlaGemmKernel 类的一个嵌套类型 EpilogueType，可能是一个枚举或结构体。
                  RES_ADD 是 EpilogueType 中的一个成员，通常表示一种特定的epilogue操作类型，这里可能是“加法与缩放”（Residual Addition with Scaling）。
                  使用 RES_ADD 表示选择特定的epilogue操作，这种操作将在主操作完成后应用于结果数据。
                  4. beta
                  beta 是传递给 add_epilogue 的第三个参数，可能是一个缩放因子或权重值，用于控制 RES_ADD 操作的强度。
                  在很多线性代数库中，beta 通常用于在加法或缩放操作中指定一个比例因子，例如执行C = alpha * A * B + beta * C时，beta就是这个比例因子。
                  5. .build()
                  .build() 通常是构建器模式中的一个方法，用来完成对象或操作的构建过程。
                  在调用 add_epilogue 方法并传递所有需要的参数之后，调用 .build() 会最终生成一个对象或执行一系列的计算操作。
                  build() 可能会返回一个最终的对象或触发整个计算图的执行。
                  */
            }
            if (policy.fallback() == false) { 
              return !policy.run(handle);   // 如果没有回退（fallback），则运行 policy，返回 !policy.run(handle) 的结果。
            }
            return policy.fallback();  //如果发生回退，返回 policy.fallback() 的结果。
          }


          case SYCLGemm::GemmBackendEpilogue::GELU: {   // 如果 epilogue 为 GELU，则配置 GELU 后处理。
            auto policy =
                ::gpu::xetla::XetlaGemmKernel<InputT>()  // 创建 XetlaGemmKernel 实例，并添加 GELU 后处理。
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


          case SYCLGemm::GemmBackendEpilogue::BIAS_GELU: {  //如果 epilogue 为 BIAS_GELU，则配置偏置和 GELU 后处理。
            auto policy =                                   // 创建一个 XetlaGemmKernel 实例，该实例参数化为类型 InputT。
                ::gpu::xetla::XetlaGemmKernel<InputT>()     // 创建 XetlaGemmKernel 实例，并添加偏置和 GELU 后处理。
                    .add_matrix_c(out)                     // 将输出矩阵 out 添加到 XetlaGemmKernel 实例中
                    .add_matrix_a(lhs)                     // 将左乘矩阵 lhs 添加到 XetlaGemmKernel 实例中。
                    .add_matrix_b(rhs)                     // 将右乘矩阵 rhs 添加到 XetlaGemmKernel 实例中。
                    .add_epilogue(
                        bias_data,
                        ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::BIAS)
                    .add_epilogue(        // 添加一个 GELU 类型的后处理 epilogue 到 XetlaGemmKernel 实例中。这里的 nullptr 表示没有额外的数据输入到 epilogue 中。
                        nullptr,
                        ::gpu::xetla::XetlaGemmKernel<InputT>::EpilogueType::GELU) 
                    .build();     // 构建并完成 XetlaGemmKernel 实例的配置。
            
            if (policy.fallback() == false) {  // 检查 policy 是否有回退（fallback）策略，如果没有回退策略（即 policy.fallback() 返回 false），则执行以下代码。
              return !policy.run(handle);      // 运行 policy，并返回其结果的反值。如果 policy.run(handle) 返回 true，则表示运行成功，返回 false；反之亦然。
            }
            return policy.fallback();          // 如果有回退策略，返回 policy.fallback() 的结果。
          }


          case SYCLGemm::GemmBackendEpilogue::RELU:      //如果 epilogue 为 RELU 或 BIAS_RELU，直接返回 true，表示暂时不支持这两种情况。
          case SYCLGemm::GemmBackendEpilogue::BIAS_RELU:
            return true;
          default:
            return Internal("Unsupported Activation mode");  //如果 epilogue 是其他未定义的值，返回一个错误状态，指示不支持的激活模式。
        }

        /*
         背景和细节
          XetlaGemmKernel：这是一个通用矩阵乘法（GEMM）内核模板类。它使用泛型 InputT 进行参数化，以支持不同的数据类型。
          add_matrix_c, add_matrix_a, add_matrix_b：这些方法将输出矩阵 out、左乘矩阵 lhs 和右乘矩阵 rhs 添加到内核实例中。
          add_epilogue：该方法将后处理 epilogue 添加到内核实例中。在这个例子中，添加了一个 GELU 类型的 epilogue。
          build：这是一个构建器模式的方法，用于完成 XetlaGemmKernel 实例的配置并返回实例。
          fallback：这是一个检查方法，用于确定内核是否有回退策略。回退策略用于在主要算法无法执行时提供替代方案。
          run：这是一个运行方法，用于执行配置好的 GEMM 内核。如果执行成功则返回 true，否则返回 false。   
        */
        }


      
          template <typename InputT>  // 定义一个模板函数 RunXetlaGemm，参数化类型为 InputT。

          // 第一部分代码
          std::enable_if_t<!std::is_same_v<InputT, ::gpu::xetla::bf16> &&
                              !std::is_same_v<InputT, sycl::half>,
                          absl::StatusOr<bool>>

                /*
                std::enable_if_t: 用于在编译期条件判断。用于在条件为真时启用此函数
                !std::is_same_v<InputT, ::gpu::xetla::bf16>: 确保 InputT 不是 ::gpu::xetla::bf16 类型。
                !std::is_same_v<InputT, sycl::half>:         确保 InputT 不是 sycl::half 类型。
                如果 InputT 既不是 ::gpu::xetla::bf16 也不是 sycl::half，则函数返回类型为 absl::StatusOr<bool>。
                            
                */
          
          RunXetlaGemm(se::gpu::GpuStreamHandle handle, const MatrixDescriptor& lhs,
                      const MatrixDescriptor& rhs, const MatrixDescriptor& c,
                      const MatrixDescriptor& out, se::DeviceMemoryBase bias,
                      SYCLGemm::GemmBackendEpilogue epilogue, float beta) 
            /*
            定义 RunXetlaGemm 函数，接受多个参数
            se::gpu::GpuStreamHandle handle：GPU 流句柄。
            const MatrixDescriptor& lhs, const MatrixDescriptor& rhs, const MatrixDescriptor& c, const MatrixDescriptor& out：描述矩阵的结构体。
            se::DeviceMemoryBase bias：偏置的设备内存基址。
            SYCLGemm::GemmBackendEpilogue epilogue：枚举类型，指定后处理操作。
            float beta：矩阵 C 的标量乘数。
            */ 


          {
            return Internal("Unsupported Datatype in XeTLA");  // 如果上述 所有的数据类型 都不支持的话，返回一个错误信息，表示不支持该数据类型。
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
              int64_t batch_size,          //矩阵的批处理大小
              const MatrixDescriptor& lhs, //描述输入矩阵和输出矩阵的结构体
              const MatrixDescriptor& rhs, 
              const MatrixDescriptor& out) 
              {
            dnnl::memory::dims lhs_dims{batch_size, lhs.num_rows, lhs.num_cols};  //初始化 lhs 矩阵的维度
                /*
                实际上是在使用 DNNL（Deep Neural Network Library，现在称为 oneDNN）的 C++ API。
                命名空间：dnnl::memory::dims 是 DNNL 提供的用于表示张量维度的数据类型。dnnl 是 DNNL 命名空间（现在称为 oneDNN）。
                对象声明：lhs_dims 是一个对象，用于存储张量（左矩阵）的维度信息。
                初始化方式：{batch_size, lhs.num_rows, lhs.num_cols} 是通过列表初始化方式初始化 lhs_dims 对象。
                具体含义：
                  batch_size：表示矩阵的批处理大小，即同时处理的矩阵个数。
                  lhs.num_rows：表示左手边矩阵的行数。
                  lhs.num_cols：表示左手边矩阵的列数。

                在深度学习框架中，如 DNNL/oneDNN，矩阵和张量的操作需要明确定义它们的维度和大小。这些信息通常在算法和内存操作中使用，以确保正确的数据布局和计算。
                如果 batch_size = 4，lhs.num_rows = 3，lhs.num_cols = 2，那么lhs_dims 将会是 {4, 3, 2}，表示一个批次中包含了四个大小为 3x2 的矩阵。

                dnnl::memory::dims lhs_dims{batch_size, lhs.num_rows, lhs.num_cols} 这行代码初始化了一个存储矩阵维度信息的对象 lhs_dims，
                其中 batch_size 是批处理大小，lhs.num_rows 和 lhs.num_cols 是左手边矩阵的行数和列数

                */
            dnnl::memory::dims rhs_dims{batch_size, rhs.num_rows, rhs.num_cols}; // 解释 同上
            dnnl::memory::dims out_dims{batch_size, out.num_rows, out.num_cols}; // 解释 同上

            auto lhs_strides =
                dnnl::memory::dims{lhs.batch_stride, lhs.leading_dim_stride, 1}; //初始化 lhs 矩阵的步幅
              /*  
              这段代码用于初始化存储 左 矩阵或张量步幅（stride）的对象
              这行代码初始化了一个存储矩阵或张量步幅信息的对象 lhs_strides，其中 lhs.batch_stride 和 lhs.leading_dim_stride 分别指定了矩阵在批处理和主维度上的步幅

              auto 关键字：auto 在这里用于自动推导变量类型，根据右侧表达式的类型来确定 lhs_strides 的类型。
              
              命名空间和对象声明：dnnl::memory::dims 是 DNNL 提供的用于表示张量或矩阵步幅（stride）的数据类型。
                lhs_strides 是一个对象，用于存储张量或矩阵的步幅信息。

              {lhs.batch_stride, lhs.leading_dim_stride, 1} 是通过列表初始化方式初始化 lhs_strides 对象。
                  具体含义：
                    lhs.batch_stride：表示矩阵在批处理中的步幅，即一个批次中相邻矩阵在内存中的偏移量。
                    lhs.leading_dim_stride：表示矩阵的主维度（通常是行或列）的步幅，即在主维度中相邻元素在内存中的偏移量。
                    1：通常表示次要维度（非主维度）的步幅，用于张量中的额外维度。
              
              用途：步幅是指定如何在内存中布置矩阵或张量数据的重要参数。它们确保数据在计算机内存中的正确布局，以便高效地进行计算和访问
              示例：
                如果 lhs.batch_stride = 24，lhs.leading_dim_stride = 12，那么 lhs_strides 将会是 {24, 12, 1}，这表示在批处理中的步幅是 24，主维度（例如行或列）的步幅是 12，而额外维度的步幅为 1。
                总结来说，auto lhs_strides = dnnl::memory::dims{lhs.batch_stride, lhs.leading_dim_stride, 1}; 这行代码初始化了一个存储矩阵或张量步幅信息的对象 lhs_strides，其中 lhs.batch_stride 和 lhs.leading_dim_stride 分别指定了矩阵在批处理和主维度上的步幅
              
              问题： Onednn 中 为什么要指定矩阵在批处理和主维度上的步幅， 比如批处理中的步幅是 24，主维度（例如行或列）的步幅是 12，而额外维度的步幅为 1。 
                    这些步幅在 计算中的作用是什么？

                    在使用 OneDNN（以前称为 DNNL）或其他深度学习框架时，指定矩阵在批处理和主维度上的步幅（stride）是为了确保数据在内存中的正确布局，并能够有效地进行计算和访问
                    批处理中的步幅和主维度的步幅
                    
                    批处理中的步幅：
                    批处理中的步幅指定了相邻矩阵在内存中的偏移量。例如，如果批处理中的步幅是 24，那么在内存中，每个相邻矩阵的数据在存储时相隔 24 个元素的位置。这确保了在处理批次数据时，能够正确地定位和访问每个矩阵的数据。
                    
                    主维度的步幅：
                    主维度通常是矩阵的行或列。主维度的步幅定义了在主维度中相邻元素在内存中的偏移量。例如，如果主维度是行，步幅是 12，则每个相邻行的数据在内存中相隔 12 个元素的位置。
                    这对于矩阵乘法等操作非常重要，因为它确保了在主维度上连续访问数据，从而提高了数据访问的效率。
                    
                    额外维度的步幅：
                    如果矩阵或张量有额外的维度（如高维张量），额外维度的步幅通常设置为 1。这表示在额外维度中相邻元素在内存中的偏移量为 1，即相邻元素存储在连续的内存位置。
                    
                    作用和优化：数据布局优化：通过设置正确的步幅，可以优化数据的存储布局，使得计算机可以高效地访问和处理数据，减少内存访问的开销。

                    计算效率：步幅的正确设置有助于并行计算和向量化操作，提高了矩阵乘法、卷积等计算操作的效率。

                    内存操作优化：在深度学习任务中，大量的计算涉及到大规模的数据操作，优化数据布局和内存访问模式可以显著提升整体的计算速度和效率。

                    综上所述，指定矩阵在批处理和主维度上的步幅是为了优化数据存储布局，以便能够高效地进行深度学习计算，减少内存访问的开销，并提高计算操作的效率和并行性。
              
              */

            auto rhs_strides =
                dnnl::memory::dims{rhs.batch_stride, rhs.leading_dim_stride, 1};  // 初始化存储 左 矩阵或张量步幅（stride）的对象 参考上述代码
            auto out_strides =
                dnnl::memory::dims{out.batch_stride, out.leading_dim_stride, 1};  // 初始化存储 输出 矩阵或张量步幅（stride）的对象 参考上述代码
            
            
            int idx_last = 2;
            int idx_2nd_last = 1;
            /*
            它们通常用于标识矩阵或张量维度的索引位置。让我们逐步解释这两行代码的含义

            int 类型声明：int 是 C++ 中的整数类型，用于声明变量。
            
            变量名和初始化：
              idx_last 和 idx_2nd_last 是两个变量名，用于存储索引值。
              idx_last = 2;：将 idx_last 初始化为 2。通常，这里的 2 可以理解为数组或张量中的最后一个维度的索引。
              idx_2nd_last = 1;：将 idx_2nd_last 初始化为 1。这里的 1 可以理解为数组或张量中的倒数第二个维度的索引。
              
            具体含义：
              在处理多维数组或张量时，索引值表示了每个维度的位置。例如，对于一个三维张量，可以使用 idx_last 和 idx_2nd_last 来表示最后一个维度和倒数第二个维度的索引位置。
              idx_last 可以用于访问数组或张量中的最内层维度，而 idx_2nd_last 则用于访问倒数第二层维度。
            
            应用场景：
            在机器学习和深度学习中，处理多维数据结构时，索引变量非常有用。例如，对于张量的转置操作，可以使用这些索引变量来交换维度的顺序。
            示例：

            如果你有一个三维张量，那么 idx_last 可能对应于最内层的维度，例如深度维度；而 idx_2nd_last 可能对应于倒数第二层的维度，例如高度或宽度维度。
            综上所述，int idx_last = 2; 和 int idx_2nd_last = 1; 这两行代码用于声明和初始化用于索引多维数据结构中特定维度的变量，
            通常用于在处理和操作数组或张量时确定维度的位置。
            */




            // dst(m,n) = \sigma{src(m,k) * weights(k, n)}
            // lhs_strides holds the strides for each dim, say {24, 12, 4, 1} for src_tensor {1, 2, 3, 4} if adj_x_ is false.
              // lhs_strides 保存每个维度的步幅，例如如果 adj_x_ 为 false，则为 {24, 12, 4, 1} 对应于 src_tensor {1, 2, 3, 4}。
            // If adj_x_ is true, swap the innermost two dims of lhs_strides to {24, 12, 1, 4}, just like set memory::format_tag::abdc
              // 如果 adj_x_ 为 true，则交换 lhs_strides 的最内两个维度为 {24, 12, 1, 4}，就像设置 memory::format_tag::abdc 一样。

            // 转置检查和调整
            if (lhs.transpose == se::blas::Transpose::kTranspose) {
              std::swap(lhs_dims[idx_last], lhs_dims[idx_2nd_last]);
              std::swap(lhs_strides[idx_last], lhs_strides[idx_2nd_last]);
            }
            /*
            条件语句： if (lhs.transpose == se::blas::Transpose::kTranspose)：这是一个条件语句，它检查变量 lhs.transpose 是否
            等于枚举类型 se::blas::Transpose::kTranspose。这个条件通常用于判断是否需要执行矩阵转置操作。
            代码块：{} 内的代码是在条件成立时执行的代码块。

            std::swap 函数：std::swap 函数，它用来交换两个变量或数组元素的值。在这里，它交换了 lhs_dims 数组中索引为 idx_last 和 idx_2nd_last 处的元素
            std::swap (lhs_strides[idx_last], lhs_strides[idx_2nd_last]);：类似地，这行代码交换了 lhs_strides 数组中索引为 idx_last 和 idx_2nd_last 处的元素。
            具体作用：当 lhs.transpose 表示需要转置时，通过交换 lhs_dims 和 lhs_strides 数组中对应索引的元素，实现了矩阵维度的交换操作。
            这通常在矩阵乘法等数学运算中很常见，以确保正确的数据布局和计算顺序。
            
            示例理解
            如果 lhs.transpose 的值为 se::blas::Transpose::kTranspose，意味着需要对 lhs_dims 和 lhs_strides 执行转置操作。
            例如，如果 idx_last = 2，idx_2nd_last = 1，则上述代码将交换 lhs_dims 中第 2 和第 1 个元素，以及 lhs_strides 中第 2 和第 1 个元素。
            
            这段代码的主要作用是根据 lhs.transpose 的值来判断是否需要对 lhs_dims 和 lhs_strides 进行转置操作，从而确保数据在矩阵运算中的正确布局和计算顺序

            */

            if (rhs.transpose == se::blas::Transpose::kTranspose) {
              std::swap(rhs_dims[idx_last], rhs_dims[idx_2nd_last]);
              std::swap(rhs_strides[idx_last], rhs_strides[idx_2nd_last]);
            }

            // 偏置维度和步幅
            dnnl::memory::dims bias_dims(rhs_dims.size(), 1); // 初始化偏置的维度
            bias_dims[rhs_dims.size() - 1] = rhs_dims[rhs_dims.size() - 1];  // 将偏置的最后一个维度设置为与 rhs 矩阵的最后一个维度相同。
            auto bias_strides = CalculateTFStrides(bias_dims); // 计算偏置维度的步幅

            return absl::make_unique<OneDnnMatMulParams>(
                lhs_dims, rhs_dims, out_dims, bias_dims, lhs_strides, rhs_strides,
                out_strides, bias_strides); 
                //返回一个指向新创建的 OneDnnMatMulParams 对象的独占指针，该对象初始化了计算得到的维度和步幅。
          }

          /*
          在 JAX 框架中，上述代码逻辑可以通过定义自定义操作和使用 JAX 的 numpy 接口来实现。
          这段代码主要用于创建矩阵乘法的参数，并在特定条件下处理数据的转置和步幅。

          以下是一个简化示例，用于展示如何在 JAX 中实现类似的逻辑

          import jax.numpy as jnp

          def create_matmul_params(batch_size, lhs, rhs, out):
              lhs_dims = (batch_size, lhs.shape[0], lhs.shape[1])
              rhs_dims = (batch_size, rhs.shape[0], rhs.shape[1])
              out_dims = (batch_size, out.shape[0], out.shape[1])

              lhs_strides = (lhs.strides[0], lhs.strides[1], 1)
              rhs_strides = (rhs.strides[0], rhs.strides[1], 1)
              out_strides = (out.strides[0], out.strides[1], 1)

              if lhs.transpose:
                  lhs_dims = (batch_size, lhs.shape[1], lhs.shape[0])
                  lhs_strides = (lhs.strides[1], lhs.strides[0], 1)

              if rhs.transpose:
                  rhs_dims = (batch_size, rhs.shape[1], rhs.shape[0])
                  rhs_strides = (rhs.strides[1], rhs.strides[0], 1)

              bias_dims = (rhs.shape[1],)
              bias_strides = (1,)

              return {
                  'lhs_dims': lhs_dims,
                  'rhs_dims': rhs_dims,
                  'out_dims': out_dims,
                  'bias_dims': bias_dims,
                  'lhs_strides': lhs_strides,
                  'rhs_strides': rhs_strides,
                  'out_strides': out_strides,
                  'bias_strides': bias_strides,
              }

          # 示例调用
          lhs = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
          rhs = jnp.array([[5, 6], [7, 8]], dtype=jnp.float32)
          out = jnp.array([[9, 10], [11, 12]], dtype=jnp.float32)
          batch_size = 1

          params = create_matmul_params(batch_size, lhs, rhs, out)
          */

        /*
        DoXetlaGemm函数 执行Xetla的GEMM操作。根据提供的矩阵描述符和其他参数，执行GEMM运算


        */


          template <typename InputT>   
          /*
          这是一个模板声明，表示这个函数是一个模板函数，可以接受任意类型的 InputT。
          该函数首先检查 output 矩阵没有被转置。
          然后将流转换为 GPU 流句柄并调用 RunXetlaGemm 函数。
          如果 RunXetlaGemm 成功，直接返回成功状态。
          否则，记录一些调试信息，并返回一个内部错误状态。
          这个函数的设计目的是执行一次通用矩阵乘法（GEMM）操作，并根据条件执行不同的处理步骤。
          */

          // 函数声明
          absl::Status DoXetlaGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                                  const MatrixDescriptor& lhs,
                                  const MatrixDescriptor& rhs, const MatrixDescriptor& c,
                                  const MatrixDescriptor& output,
                                  se::DeviceMemoryBase bias, float alpha, float beta,
                                  SYCLGemm::GemmBackendEpilogue epilogue, se::Stream* stream,
                                  std::optional<se::blas::AlgorithmType> algorithm,
                                  se::ScratchAllocator* scratch_allocator,
                                  se::blas::ComputePrecision compute_precision) 
            /*
            这是 DoXetlaGemm 函数的完整声明，返回类型是 absl::Status。
            参数列表包含多个矩阵描述符、计算参数和上下文信息。
            
            函数返回类型absl::Status:
                这是函数的返回类型，用于表示操作的结果状态。absl::Status 可以表示成功或失败，并包含错误信息。
            
            函数参数
                int64_t batch_size:
                表示批处理的大小，即同时处理的矩阵对数。在深度学习中，批处理大小通常用于加速训练和推理过程。
                
                int64_t m:
                表示矩阵 lhs 和 output 的行数。在矩阵乘法中，这通常是结果矩阵的行数。
                
                int64_t n:
                表示矩阵 rhs 和 output 的列数。在矩阵乘法中，这通常是结果矩阵的列数。
                
                int64_t k:
                表示矩阵 lhs 的列数和矩阵 rhs 的行数。在矩阵乘法中，这是相乘的两个矩阵的公共维度。
                
                const MatrixDescriptor& lhs:表示左乘矩阵的描述符。
                MatrixDescriptor 通常包含矩阵的维度、步幅、数据类型等信息。
                const MatrixDescriptor& rhs:表示右乘矩阵的描述符。类似于 lhs，包含矩阵的详细信息。
                const MatrixDescriptor& c:
                表示一个额外的矩阵 c 的描述符，可能用于矩阵乘法结果的偏置或其他操作。
                这在一些高级的矩阵运算中会用到，比如GEMM操作中的C矩阵。
                
                const MatrixDescriptor& output:表示输出矩阵的描述符。包含输出矩阵的维度、步幅、数据类型等信息。
                
                se::DeviceMemoryBase bias:表示偏置矩阵的设备内存基地址。在深度学习中，偏置项通常用于增加模型的表现力。
                float alpha: GEMM运算中的比例因子 alpha。用于缩放矩阵乘法的结果。
                float beta: GEMM运算中的比例因子 beta。用于缩放加到结果中的矩阵 c。
                
                SYCLGemm::GemmBackendEpilogue epilogue:表示GEMM运算的后处理选项。这可能包括加上偏置、激活函数等操作。
                se::Stream* stream:表示执行操作的流。流用于在设备上排队和同步操作，常用于GPU计算中。
                std::optional<se::blas::AlgorithmType> algorithm:表示可选的BLAS算法类型。提供不同的矩阵乘法实现，优化性能。
                se::ScratchAllocator* scratch_allocator:表示临时内存分配器。用于在计算过程中分配临时内存。
                se::blas::ComputePrecision compute_precision:表示计算精度。可以指定计算时使用的精度类型，如浮点数或双精度浮点数。
            
            */
                                  
                                  {
            CHECK(output.transpose == se::blas::Transpose::kNoTranspose); 
            //这是一个检查宏，用于确保 output 矩阵没有被转置。如果 output.transpose 不等于 se::blas::Transpose::kNoTranspose，程序将会中止
            /*
            功能：这是一个检查宏，用于确保 output 矩阵没有被转置。
            作用：如果 output.transpose 不等于 se::blas::Transpose::kNoTranspose，即如果 output 矩阵被转置了，程序将会中止。
            上下文：这种检查通常用于在关键的前提条件不满足时快速发现错误，从而避免后续操作在错误的前提下进行。
            */

            se::gpu::GpuStreamHandle stream_handle = stream_executor::gpu::AsGpuStreamValue(stream); 
            // 这行代码将 stream 转换为 GPU 流句柄 stream_handle，用于后续的 GPU 操作。
            //功能：将 stream 转换为 GPU 流句柄 stream_handle。
            //作用：将高层次的流对象 stream 转换为底层的 GPU 流句柄 stream_handle，用于后续的 GPU 操作
            // 上下文：流句柄用于在 GPU 上排队和管理异步操作，是进行 GPU 计算的基础。

            TF_ASSIGN_OR_RETURN(bool fallback,
                                RunXetlaGemm<InputT>(stream_handle, lhs, rhs, c, output,  // 这行代码调用 RunXetlaGemm 函数，并将其返回值赋给 fallback。
                                                    bias, epilogue, beta));              // 如果 RunXetlaGemm 函数调用失败，则直接返回错误状态
            /*

            功能：调用 RunXetlaGemm 函数，并将其返回值赋给 fallback。
            作用：RunXetlaGemm 是一个模板函数，用于执行特定的数据类型 InputT 的矩阵乘法操作。
            stream_handle：GPU 流句柄，用于管理 GPU 操作。
            lhs：左乘矩阵的描述符。
            rhs：右乘矩阵的描述符。
            c：额外矩阵的描述符，可能用于偏置或其他操作。
            output：输出矩阵的描述符。
            bias：偏置矩阵的设备内存基地址。
            epilogue：GEMM 运算的后处理选项。
            beta：GEMM 运算中的比例因子。
            上下文：
            TF_ASSIGN_OR_RETURN 宏：尝试执行 RunXetlaGemm 函数，如果失败，则直接返回错误状态。如果成功，则将返回值赋给 fallback 变量。
            fallback：一个布尔值，表示是否需要回退到其他实现（根据 RunXetlaGemm 的返回值来决定）。
            这行代码确保在调用 RunXetlaGemm 时处理可能的错误，并在失败时立即返回错误状态，避免继续执行后续代码。
            
            总结
              通过 CHECK 宏确保 output 矩阵没有被转置，从而避免潜在错误。
              将高层次的 stream 对象转换为底层的 stream_handle，用于 GPU 操作。
              调用 RunXetlaGemm 函数，并通过 TF_ASSIGN_OR_RETURN 宏处理其返回值，确保在错误发生时立即返回错误状态。
            */
            if (!fallback) return OkStatus();          // 如果 fallback 为 false，则表示 RunXetlaGemm 成功，直接返回 OkStatus() 表示成功状态。

            VLOG(2) << "lhs: " << batch_size << " " << lhs.num_rows << " "
                    << lhs.num_cols;
            VLOG(2) << "rhs: " << batch_size << " " << rhs.num_rows << " "
                    << rhs.num_cols;
            VLOG(2) << "out: " << batch_size << " " << output.num_rows << " "
                    << output.num_cols;
            /*
            这几行代码使用 VLOG 记录矩阵 lhs、rhs 和 output 的维度信息，日志级别为 2。
              1. if (!fallback) return OkStatus();
              功能：检查 fallback 变量，如果 fallback 为 false，则返回 OkStatus()。
              作用：
              fallback 是之前调用 RunXetlaGemm 函数的返回值，表示是否需要回退到其他实现。
              如果 fallback 为 false，表示 RunXetlaGemm 成功完成任务，不需要回退。
              在这种情况下，函数直接返回 OkStatus()，表示操作成功。
              上下文：这行代码用于在不需要回退的情况下提前结束函数执行，从而避免不必要的计算和操作。
              
              2. VLOG(2) << "lhs: " << batch_size << " " << lhs.num_rows << " " << lhs.num_cols;
              功能：记录左乘矩阵（lhs）的相关信息。
              作用：
              VLOG(2) 是一个日志宏，用于在日志级别 2 记录调试信息。
              记录的信息包括批处理大小 (batch_size)、lhs 矩阵的行数 (lhs.num_rows) 和列数 (lhs.num_cols)。
              上下文：用于调试和验证矩阵的维度信息，帮助开发人员确认输入数据的正确性。
              
              3. VLOG(2) << "rhs: " << batch_size << " " << rhs.num_rows << " " << rhs.num_cols;
              功能：记录右乘矩阵（rhs）的相关信息。
              作用：
              同样使用 VLOG(2) 记录调试信息。
              记录的信息包括批处理大小 (batch_size)、rhs 矩阵的行数 (rhs.num_rows) 和列数 (rhs.num_cols)。
              上下文：用于调试和验证矩阵的维度信息，帮助开发人员确认输入数据的正确性。
              
              4. VLOG(2) << "out: " << batch_size << " " << output.num_rows << " " << output.num_cols;
              功能：记录输出矩阵（output）的相关信息。
              作用：
              继续使用 VLOG(2) 记录调试信息。
              记录的信息包括批处理大小 (batch_size)、output 矩阵的行数 (output.num_rows) 和列数 (output.num_cols)。
              上下文：用于调试和验证输出矩阵的维度信息，帮助开发人员确认计算结果的正确性。
              
              总结
              if (!fallback) return OkStatus();：

              检查 fallback 变量，如果为 false，则返回成功状态 OkStatus()，表示 RunXetlaGemm 已成功完成任务，无需回退到其他实现。
              VLOG(2) << "lhs: " << batch_size << " " << lhs.num_rows << " " << lhs.num_cols;：

              记录左乘矩阵 lhs 的批处理大小、行数和列数，供调试使用。
              VLOG(2) << "rhs: " << batch_size << " " << rhs.num_rows << " " << rhs.num_cols;：

              记录右乘矩阵 rhs 的批处理大小、行数和列数，供调试使用。
              VLOG(2) << "out: " << batch_size << " " << output.num_rows << " " << output.num_cols;：

              记录输出矩阵 output 的批处理大小、行数和列数，供调试使用。
              通过这些步骤，代码确保了在 RunXetlaGemm 成功时提前返回，并在需要调试时提供了矩阵维度信息。


            */

            // 如果程序执行到这里，表示出现了某种错误，函数返回一个内部错误状态       
            return absl::InternalError("Anyway, something is wrong in DoXetlaGemm."); 
          }

          

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
            /*
            这是 DoOnednnGemm函数的完整声明
            跟上述  DoXetlaGemm 函数的完整声明一样，返回类型是 absl::Status。
            参数列表包含多个矩阵描述符、计算参数和上下文信息。
            矩阵乘法（GEMM）操作的实现和优化上，分别通过Xetla和oneDNN两个后端来执行这些操作
            
            具体执行包括：
            1. 创建矩阵乘法参数
            2. 创建oneDNN的内存描述符
            3. 创建oneDNN引擎和属性
            4. 设置fp32模式
            5. 创建矩阵乘法的primitive descriptor
            */                         

            CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
            //这是一个检查宏，用于确保 output 矩阵没有被转置。如果 output.transpose 不等于 se::blas::Transpose::kNoTranspose，程序将会中止
            /*
            功能：这是一个检查宏，用于确保 output 矩阵没有被转置。
            作用：如果 output.transpose 不等于 se::blas::Transpose::kNoTranspose，即如果 output 矩阵被转置了，程序将会中止。
            上下文：这种检查通常用于在关键的前提条件不满足时快速发现错误，从而避免后续操作在错误的前提下进行。
            */
            se::gpu::GpuStreamHandle stream_handle =
                stream_executor::gpu::AsGpuStreamValue(stream);
    
            // 这行代码将 stream 转换为 GPU 流句柄 stream_handle，用于后续的 GPU 操作。
            //功能：将 stream 转换为 GPU 流句柄 stream_handle。
            //作用：将高层次的流对象 stream 转换为底层的 GPU 流句柄 stream_handle，用于后续的 GPU 操作
            // 上下文：流句柄用于在 GPU 上排队和管理异步操作，是进行 GPU 计算的基础。

            void* lhs_data = const_cast<void*>(lhs.data.opaque());
              /*
              功能：获取左乘矩阵 lhs 数据的可修改指针。
              作用：
              lhs.data.opaque()：返回一个指向 lhs 数据的不可变（const）指针。
              const_cast<void*>：移除指针的常量性，得到一个可修改的 void* 指针。
              上下文：需要一个可修改的数据指针以便在后续操作中使用，即使原数据是不可变的。
              通过上述代码，我们将不可变的指针转换为可修改的指针，以便在后续的操作中对数据进行修改。以下是详细的总结：

              void* lhs_data = const_cast<void*>(lhs.data.opaque());：

              将左乘矩阵 lhs 数据的不可变指针转换为可修改指针 lhs_data。
              void* rhs_data = const_cast<void*>(rhs.data.opaque());：

              将右乘矩阵 rhs 数据的不可变指针转换为可修改指针 rhs_data。
              void* c_data = const_cast<void*>(c.data.opaque());：

              将矩阵 c 数据的不可变指针转换为可修改指针 c_data。
              void* out_data = const_cast<void*>(output.data.opaque());：

              将输出矩阵 output 数据的不可变指针转换为可修改指针 out_data。
              void* bias_data = const_cast<void*>(bias.opaque());：

              将偏置数据 bias 的不可变指针转换为可修改指针 bias_data。
              这些转换确保在后续操作中，程序可以对这些数据进行修改，即使原数据是不可变的。
              
              */
            void* rhs_data = const_cast<void*>(rhs.data.opaque());  // 功能：获取右乘矩阵 lhs 数据的可修改指针。
            void* c_data = const_cast<void*>(c.data.opaque());      // 功能：获取矩阵 c 数据的可修改指针。
            void* out_data = const_cast<void*>(output.data.opaque()); // 功能：获取输出矩阵的 数据的可修改指针
            void* bias_data = const_cast<void*>(bias.opaque());      // 功能：获取偏置 矩阵的 数据的可修改指针


            VLOG(2) << "lhs: " << batch_size << " " << lhs.num_rows << " "
                    << lhs.num_cols;

            /*
            功能：记录左乘矩阵（lhs）的相关信息。
              作用：
              VLOG(2) 是一个日志宏，用于在日志级别 2 记录调试信息。
              记录的信息包括批处理大小 (batch_size)、lhs 矩阵的行数 (lhs.num_rows) 和列数 (lhs.num_cols)。
              上下文：用于调试和验证矩阵的维度信息，帮助开发人员确认输入数据的正确性。
            */
              
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



            //这些步骤为矩阵乘法操作准备了详细的内存描述符，以便 oneDNN 库在内存中正确存储和访问这些矩阵数据。
            auto params = CreateMatMulParams(batch_size, lhs, rhs, output);
              /*
              功能：调用 CreateMatMulParams 函数，生成矩阵乘法操作的参数。
              作用：
              CreateMatMulParams 函数根据输入的矩阵描述符 lhs、rhs 和 output 以及 batch_size，创建并返回一个 OneDnnMatMulParams 的智能指针（unique_ptr）。
              这些参数包括矩阵的维度和步幅等信息。
              上下文：在进行矩阵乘法操作之前，需要生成并准备好相关参数。
              
              */
            auto src_md = dnnl::memory::desc(params->a_dims, OneDnnType<InputT>(),
                                            params->a_strides);

              /*
              功能：创建源矩阵（lhs）的内存描述符。
              作用：
              dnnl::memory::desc：构造一个内存描述符，包含矩阵的维度、数据类型和步幅。
              params->a_dims：源矩阵的维度。
              OneDnnType<InputT>()：源矩阵的数据类型。
              params->a_strides：源矩阵的步幅。
              上下文：内存描述符用于指定在内存中如何存储和访问矩阵数据，是 oneDNN 库操作的重要部分。
              */
            auto weights_md = dnnl::memory::desc(params->b_dims, OneDnnType<InputT>(),
                                                params->b_strides);
              /*
              功能：创建权重矩阵（rhs）的内存描述符。
              作用：
              dnnl::memory::desc：构造一个内存描述符，包含矩阵的维度、数据类型和步幅。
              params->b_dims：权重矩阵的维度。
              OneDnnType<InputT>()：权重矩阵的数据类型。
              params->b_strides：权重矩阵的步幅。
              上下文：内存描述符用于指定在内存中如何存储和访问矩阵数据，是 oneDNN 库操作的重要部分。
              
              */
            auto dst_md = dnnl::memory::desc(params->c_dims, OneDnnType<OutputT>(),
                                            params->c_strides);
              /*
              功能：创建目标矩阵（output）的内存描述符。
              作用：
              dnnl::memory::desc：构造一个内存描述符，包含矩阵的维度、数据类型和步幅。
              params->c_dims：目标矩阵的维度。
              OneDnnType<OutputT>()：目标矩阵的数据类型。
              params->c_strides：目标矩阵的步幅。
              上下文：内存描述符用于指定在内存中如何存储和访问矩阵数据，是 oneDNN 库操作的重要部分。
              
              */
            auto bias_md =
                bias_data ? dnnl::memory::desc(params->bias_dims, OneDnnType<InputT>(),
                                              params->bias_strides)
                          : dnnl::memory::desc();
              /*
              功能：根据偏置数据的存在与否创建偏置矩阵的内存描述符。
              作用：
              检查 bias_data 是否存在（非空）。
              如果 bias_data 存在，创建包含偏置矩阵维度、数据类型和步幅的内存描述符。
              如果 bias_data 不存在，创建一个默认的空内存描述符。
              上下文：偏置矩阵是可选的，如果存在则需要相应的内存描述符来指定其存储和访问方式。
              */

            auto dnnl_engine = FindOrCreateEngine(stream_handle);  //为后续的计算操作获取一个 DNNL 引擎。
              /*
              auto dnnl_engine
              功能：使用自动类型推断（auto）来声明变量 dnnl_engine，类型由右侧表达式决定。
              上下文：C++11 引入的自动类型推断功能，可以减少冗长的类型声明，使代码更简洁。
              2. FindOrCreateEngine(stream_handle)
              功能：调用 FindOrCreateEngine 函数，传入 stream_handle 参数，返回一个 dnnl::engine 对象。
              作用：
              stream_handle：GPU 流句柄，用于在 GPU 上执行操作。
              FindOrCreateEngine：查找现有的 DNNL 引擎（如果存在），或者创建一个新的引擎，并返回它。
              上下文：在使用 oneDNN 库进行计算之前，需要一个 DNNL 引擎对象，该引擎管理和协调计算资源。
              上下文和背景
              DNNL 引擎（dnnl::engine）：DNNL 引擎是 oneDNN 库的核心组件之一，它代表了执行计算操作的设备和上下文。一个 DNNL 引擎可以是 CPU 或 GPU 引擎，用于执行深度学习操作。
              GPU 流句柄（stream_handle）：GPU 流是一种异步计算的机制，它允许多个计算操作在 GPU 上并行执行。stream_handle 是一个指向这种流的指针或句柄。
              
              目的：为后续的计算操作获取一个 DNNL 引擎。
              细节：
              auto 关键字：自动推断 dnnl_engine 的类型。
              FindOrCreateEngine 函数：根据提供的 stream_handle，查找或创建一个 DNNL 引擎。
              stream_handle 参数：指定 GPU 流句柄，用于在 GPU 上执行计算。
              通过这行代码，我们确保在执行 oneDNN 库的操作之前，有一个有效的 DNNL 引擎来管理和协调计算资源。
              */
            
            //通过这些步骤，我们配置了计算原语的属性和后处理操作，确保在计算过程中使用用户提供的临时内存，并根据输入数据类型选择合适的数学计算模式
            
            dnnl::primitive_attr post_ops_attr;
              /*
              功能：声明一个 dnnl::primitive_attr 对象 post_ops_attr。
              作用：dnnl::primitive_attr 是 oneDNN 库中的一个类，用于描述和配置计算原语（例如卷积、矩阵乘法等）的属性和行为。post_ops_attr 用于设置后续操作的属性。
              上下文：在进行复杂计算（如矩阵乘法、卷积等）时，可能需要配置一些额外的属性，例如使用用户提供的 scratchpad 内存或设置数学计算模式。
              */
            post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
              /*
              功能：设置 post_ops_attr 对象的 scratchpad 模式为 user 模式。
              作用：dnnl::scratchpad_mode::user 表示用户提供 scratchpad 内存，而不是由 oneDNN 自动管理。
              scratchpad 内存：临时工作空间，用于在计算过程中存储中间结果和辅助数据。
              通过设置为 user 模式，用户可以更好地控制内存管理，可能提升性能或减少内存占用。
              上下文：在高性能计算中，控制内存的分配和管理有时是必要的，以提高效率。
              
              */
            // Set fp32 mode.
            dnnl::fpmath_mode fp32_math_mode = GetFP32MathMode();
              /*
              功能：调用 GetFP32MathMode 函数，获取 FP32（32位浮点）数学计算模式，并将其赋值给 fp32_math_mode 变量。
              作用：
              dnnl::fpmath_mode 是 oneDNN 库中的一个枚举类型，用于指定浮点数学计算模式。
              GetFP32MathMode 函数的具体实现没有提供，但通常会返回一个合适的计算模式，例如高精度或高性能模式。
              上下文：不同的计算模式可以平衡精度和性能，根据实际需求选择合适的模式。
              
              */

            if (std::is_same<InputT, float>::value) {
              post_ops_attr.set_fpmath_mode(fp32_math_mode);
            }

            /*
            功能：检查 InputT 类型是否为 float，如果是，则设置 post_ops_attr 对象的浮点数学计算模式。
            作用：
            std::is_same<InputT, float>::value：这是一个编译时类型检查，如果 InputT 类型为 float，则返回 true。
            post_ops_attr.set_fpmath_mode(fp32_math_mode)：将前面获取的 FP32 计算模式设置到 post_ops_attr 对象中。
            上下文：只有在输入类型为 float 时，才需要设置 FP32 计算模式。这是因为其他数据类型可能有不同的计算模式或不需要设置。
            
            */

            dnnl::post_ops post_ops = dnnl::post_ops();
            /*
            功能：创建一个 dnnl::post_ops 对象 post_ops。
            作用：
            dnnl::post_ops 是 oneDNN 库中的一个类，用于描述计算原语执行后的操作，例如加法、乘法等。
            初始化一个 post_ops 对象，后续可以在这个对象中添加需要的后处理操作。
            上下文：在进行矩阵乘法等计算操作后，可能需要一些额外的后处理步骤，这些步骤可以通过 post_ops 对象来描述和配置。
                    
            */


            // C = activation(MatMul(x, w, bias) + beta * C)
            //   po.append_sum(beta)
            //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);

            // 这段代码展示了在执行计算前配置和应用后处理操作的过程。首先，通过 CHECK 宏检查 alpha 是否接近于 1.0f，
            // 然后根据 beta 的值决定是否执行加和操作。接着，根据 epilogue 的不同取值，配置适当的后处理操作，
            // 最后将配置好的后处理操作应用到计算原语的属性中，以确保在计算执行后进行必要的数据处理和转换。
            CHECK(fabs(alpha - 1.0f) < 1e-6);
              /*
              功能：使用 CHECK 宏来检查条件 fabs(alpha - 1.0f) < 1e-6 是否成立。
              作用：CHECK 宏通常用于运行时断言，用于确保条件为真，否则会中止程序执行并输出错误消息。
              fabs(alpha - 1.0f)：计算 alpha 与 1.0f 的绝对差值。
              < 1e-6：检查绝对差值是否小于 1e-6（即 0.000001）。
              上下文：这行代码的作用是确保 alpha 的值接近于 1.0f，通常用于验证参数或状态的正确性。
              */
            if (c_data && fabs(beta - 0.0f) > 1e-6) post_ops.append_sum(beta);
              /*
              功能：条件语句检查 c_data 不为空且 fabs(beta - 0.0f) > 1e-6，如果成立则执行 post_ops.append_sum(beta)。
              作用：
              c_data：表示 c 的数据，如果存在则为真。
              fabs(beta - 0.0f) > 1e-6：检查 beta 的绝对值是否大于 1e-6（即 0.000001），即 beta 不接近于 0.0f。
              post_ops.append_sum(beta)：将 beta 添加到 post_ops 中，用于后续计算的加和操作。
              上下文：根据条件，如果 c_data 存在且 beta 不接近于 0.0f，则将 beta 添加到后处理操作中。
              */

            switch (epilogue) {
              /*
              功能：根据 epilogue 的值进行不同的处理。 根据后处理的类型，选择适当的激活函数或不进行任何操作
              作用：
              epilogue 是 SYCLGemm::GemmBackendEpilogue 枚举类型，表示后处理的类型或模式。
              switch 语句根据 epilogue 的不同取值执行相应的操作
              */
              case SYCLGemm::GemmBackendEpilogue::RELU:
              case SYCLGemm::GemmBackendEpilogue::BIAS_RELU:
                post_ops.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);  //如果 epilogue 是这些值，则在 post_ops 中追加 ReLU 激活函数操作。
                break;
              case SYCLGemm::GemmBackendEpilogue::GELU:
              case SYCLGemm::GemmBackendEpilogue::BIAS_GELU:
                post_ops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 0, 0);  // 如果 epilogue 是这些值，则在 post_ops 中追加 GELU（Gaussian Error Linear Unit）激活函数操作
                break;
              case SYCLGemm::GemmBackendEpilogue::DEFAULT: //如果 epilogue 是这些值，则不执行任何后处理操作
              case SYCLGemm::GemmBackendEpilogue::BIAS:   // 如果 epilogue 是这些值，则不执行任何后处理操作
                break;
              default:
                return Internal("Unsupported Activation mode");  // 对于其他未定义的 epilogue 值，返回一个内部错误消息，表示不支持的激活模式。
            }
            post_ops_attr.set_post_ops(post_ops);
              /*
              功能：将配置好的 post_ops 对象设置到 post_ops_attr 中。
              作用：post_ops_attr 是 dnnl::primitive_attr 类型的对象，用于描述计算原语执行后的操作。
              post_ops 是 dnnl::post_ops 类型的对象，包含了后处理操作，如加和、激活函数等。
              上下文：通过 set_post_ops 方法，将配置好的后处理操作应用到计算原语属性中，确保在执行计算时进行适当的后处理操作。
              */


            auto matmul_pd =  
                bias_data
                    ? std::make_shared<dnnl::matmul::primitive_desc>(
                          dnnl_engine, src_md, weights_md, bias_md, dst_md, post_ops_attr)
                    : std::make_shared<dnnl::matmul::primitive_desc>(
                          dnnl_engine, src_md, weights_md, dst_md, post_ops_attr);
                      
              /*
              功能：根据 bias_data 的存在与否，创建一个 dnnl::matmul::primitive_desc 对象 matmul_pd。
              作用：
              dnnl::matmul::primitive_desc 是 oneDNN 中描述矩阵乘法计算原语的对象，用于配置和描述矩阵乘法的计算。
              详细位置：https://oneapi-src.github.io/oneDNN/struct_dnnl_matmul_primitive_desc.html
               
              根据是否有偏置数据 (bias_data)，选择不同的构造函数：
              如果有 bias_data，则使用包含偏置的构造函数，该构造函数需要传递引擎 (dnnl_engine)、输入矩阵描述 (src_md 和 weights_md)、
              偏置描述 (bias_md)、输出矩阵描述 (dst_md) 和后处理属性 (post_ops_attr)。
              
              如果没有 bias_data，则使用不包含偏置的构造函数，只传递引擎、输入矩阵描述、输出矩阵描述和后处理属性。
              上下文：此行代码用于创建一个矩阵乘法的计算原语描述对象，根据是否有偏置来选择合适的构造方式。
              */
            
            std::unordered_map<int, dnnl::memory> fwd_primitive_args;
              /*
              功能：声明一个空的无序映射 fwd_primitive_args，键为整数类型，值为 dnnl::memory 类型。
              作用：
              std::unordered_map 是 C++ 中的容器，用于存储键值对，并支持快速查找。
              在这里，用于存储用于执行矩阵乘法计算的原语的参数。
              上下文：准备用于执行矩阵乘法计算的参数映射，后续可能用于存储输入数据、输出数据等信息。

              这段代码展示了如何根据输入条件选择不同的构造函数来创建矩阵乘法计算原语的描述对象 matmul_pd，
              并声明一个空的映射 fwd_primitive_args 用于存储执行计算所需的参数。
              通过这些步骤，准备好执行后续的矩阵乘法计算操作，并配置了可能需要的参数和属性
              */

            size_t scratchpad_size = matmul_pd->scratchpad_desc().get_size();
            void* workspace;
            TF_RETURN_IF_ERROR(
                AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));

              /*
              这段代码的作用是分配用于矩阵乘法计算的临时工作空间
              1. size_t scratchpad_size = matmul_pd->scratchpad_desc().get_size();
              功能：获取矩阵乘法计算原语描述对象 matmul_pd 的临时工作空间大小。
              作用：
              matmul_pd->scratchpad_desc()：获取用于存储计算过程中临时数据的描述对象。
              .get_size()：获取该临时空间的大小。
              scratchpad_size：保存获取到的临时空间大小。
              上下文：计算矩阵乘法过程中可能需要的临时空间大小，以便后续分配和使用。
              
              2. void* workspace;
              功能：声明一个指向 void 类型的指针 workspace，用于存储临时工作空间的地址。
              作用：
              准备一个指针，稍后用于分配临时工作空间的内存。
              上下文：在这里，声明但尚未分配任何具体内存，为分配临时空间做准备。
              
              3. TF_RETURN_IF_ERROR(AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));
              功能：调用 AllocateWorkspace 函数来分配临时工作空间，并检查分配是否成功。
              作用：
              AllocateWorkspace：可能是一个自定义函数或库函数，用于分配临时工作空间。
              &workspace：传递指向 workspace 指针的引用，函数将在此处分配临时空间的地址。
              scratch_allocator：可能是一个分配器对象或接口，用于管理临时空间的分配和释放。
              scratchpad_size：之前计算得到的临时空间大小。
              TF_RETURN_IF_ERROR：如果 AllocateWorkspace 函数返回错误状态，则立即返回该错误状态。
              
              上下文：这行代码的作用是确保在分配临时空间时不发生错误，并将分配的地址存储在 workspace 中，以便后续使用。
              总结
              这段代码用于准备并分配执行矩阵乘法计算所需的临时工作空间。首先，通过 matmul_pd 获取计算原语的临时空间大小，
              然后声明一个指针 workspace 来接收分配的地址。最后，调用 AllocateWorkspace 函数来实际分配临时空间，
              并使用 TF_RETURN_IF_ERROR 宏来确保分配操作不出错，从而保证后续计算过程的顺利执行。
                        
              */
            
            // 这段代码是在配置和准备执行矩阵乘法计算的过程中使用的一些操作
            auto scratchpad_mem =
                dnnl::memory(matmul_pd->scratchpad_desc(), dnnl_engine, workspace);
              /*
              功能：创建一个 oneDNN 内存对象 scratchpad_mem，用于存储矩阵乘法计算过程中的临时数据。
              作用：
              matmul_pd->scratchpad_desc()：获取矩阵乘法计算原语描述对象 matmul_pd 的临时数据描述。
              dnnl_engine：oneDNN 引擎，用于管理和执行计算。
              workspace：之前分配的临时工作空间的地址。
              dnnl::memory 构造函数：用于创建一个 oneDNN 内存对象，需要传递内存描述、引擎和数据所在的地址。
              上下文：创建用于存储临时数据的 oneDNN 内存对象，以便后续将其作为参数传递给矩阵乘法计算原语。
                        
              */
            auto matmul_primitive = dnnl::matmul(*matmul_pd);
              /*
              功能：创建一个矩阵乘法计算的 oneDNN 原语对象 matmul_primitive。
              作用：
              dnnl::matmul 构造函数：根据传入的矩阵乘法计算原语描述对象 matmul_pd 创建一个矩阵乘法计算原语对象。
              *matmul_pd：解引用 matmul_pd 指针，获取其指向的描述对象。
              上下文：准备用于执行矩阵乘法计算的原语对象，后续可以使用该对象来执行计算。
              
              */

            auto dnnl_stream = dnnl::sycl_interop::make_stream(
                dnnl_engine, *(stream_executor::gpu::AsGpuStreamValue(stream)));
                /*
                功能：创建一个与 SYCL 流相关联的 oneDNN 流对象 dnnl_stream。
                作用：
                dnnl_engine：oneDNN 引擎，用于管理和执行计算。
                stream_executor::gpu::AsGpuStreamValue(stream)：将输入的 stream 转换为 GPU 流句柄，用于后续的 GPU 操作。
                dnnl::sycl_interop::make_stream：创建一个与 SYCL 流相关联的 oneDNN 流对象。
                上下文：准备一个用于与 SYCL 流进行交互的 oneDNN 流对象，以便后续在 GPU 上执行计算。
                */

            auto src_mem = CreateDnnlMemory(src_md, dnnl_engine, lhs_data);
                /*
                功能：创建一个用于存储输入数据的 oneDNN 内存对象 src_mem。
                作用：
                CreateDnnlMemory：可能是一个自定义函数或库函数，用于创建 oneDNN 内存对象。
                src_md：描述输入数据的内存描述对象。
                dnnl_engine：oneDNN 引擎，用于管理和执行计算。
                lhs_data：输入数据的指针。
                上下文：准备用于存储输入数据的 oneDNN 内存对象，以便后续将其作为参数传递给矩阵乘法计算。
                */
            auto wei_mem = CreateDnnlMemory(weights_md, dnnl_engine, rhs_data);
                /*
            
                */
            auto dst_mem = CreateDnnlMemory(dst_md, dnnl_engine, out_data);
            
            fwd_primitive_args.emplace(DNNL_ARG_SRC, src_mem);
              /*
              功能：将上面创建的内存对象添加到用于执行矩阵乘法计算的参数映射 fwd_primitive_args 中。
              作用：
              fwd_primitive_args：之前声明的 std::unordered_map，用于存储执行计算所需的参数。
              DNNL_ARG_SRC、DNNL_ARG_WEIGHTS、DNNL_ARG_DST、DNNL_ARG_SCRATCHPAD：这些是 oneDNN 定义的常量，用于指定不同类型的参数。
              src_mem、wei_mem、dst_mem、scratchpad_mem：分别是存储输入数据、权重数据、输出数据和临时数据的内存对象。
              上下文：将创建的内存对象添加到参数映射中，以便后续执行矩阵乘法计算时使用。
              
              
              这段代码的主要作用是准备执行矩阵乘法计算所需的对象和参数。
              首先，创建了用于存储计算过程中临时数据的内存对象 scratchpad_mem，并创建了矩阵乘法计算的原语对象 matmul_primitive。
              然后，准备了与 SYCL 流相关联的流对象 dnnl_stream，并创建了用于存储输入、权重、输出数据的内存对象，并将它们添加到参数映射 fwd_primitive_args 中。
              这些准备工作为后续在 GPU 上执行矩阵乘法计算打下了基础。
              */
            fwd_primitive_args.emplace(DNNL_ARG_WEIGHTS, wei_mem);
            fwd_primitive_args.emplace(DNNL_ARG_DST, dst_mem);
            fwd_primitive_args.emplace(DNNL_ARG_SCRATCHPAD, scratchpad_mem);
            

            if (bias_data) {
                /*
                功能：检查是否存在偏置数据。
                作用：
                bias_data 是一个指向偏置数据的指针，如果该指针非空（即存在偏置数据），则执行条件内的代码块。
                上下文：判断是否需要将偏置数据添加到执行矩阵乘法计算的参数映射中。
                */
              auto bias_mem = CreateDnnlMemory(bias_md, dnnl_engine, bias_data);
                /*
                功能：创建一个用于存储偏置数据的 oneDNN 内存对象 bias_mem。
                作用：
                CreateDnnlMemory：可能是一个自定义函数或库函数，用于创建 oneDNN 内存对象。
                bias_md：描述偏置数据的内存描述对象。
                dnnl_engine：oneDNN 引擎，用于管理和执行计算。
                bias_data：偏置数据的指针。
                上下文：准备用于存储偏置数据的 oneDNN 内存对象，以便后续将其作为参数传递给矩阵乘法计算。
                
                */

              fwd_primitive_args.emplace(DNNL_ARG_BIAS, bias_mem);
                /*
                功能：将偏置数据的内存对象 bias_mem 添加到执行矩阵乘法计算的参数映射 fwd_primitive_args 中。
                作用：
                fwd_primitive_args：之前声明的 std::unordered_map，用于存储执行计算所需的参数。
                DNNL_ARG_BIAS：oneDNN 定义的常量，用于指定偏置数据的参数类型。
                bias_mem：存储偏置数据的内存对象。
                */

            }

            matmul_primitive.execute(dnnl_stream, fwd_primitive_args);
              /*
              功能：执行矩阵乘法计算。
              作用：
              matmul_primitive：之前创建的矩阵乘法计算的原语对象。
              dnnl_stream：与 SYCL 流相关联的 oneDNN 流对象，用于在 GPU 上执行计算。
              fwd_primitive_args：存储了执行计算所需的所有参数，包括输入数据、权重数据、输出数据和偏置数据。
              上下文：通过调用 execute 方法执行矩阵乘法计算，使用给定的流对象和参数。
              
              */
            
            return absl::OkStatus();
              /*
              功能：返回一个成功的状态。
              作用：
              absl::OkStatus()：表示执行成功的状态对象。
              上下文：在成功执行矩阵乘法计算后，返回一个成功状态，表明计算过程没有出现错误。
              */


            /*
            这段代码的主要作用是执行矩阵乘法计算。首先，检查是否存在偏置数据，如果有，则创建对应的内存对象并将其添加到参数映射中。
            然后，调用矩阵乘法计算原语对象的 execute 方法，在给定的流对象上执行计算，并返回成功的状态。
            这些步骤确保了矩阵乘法计算能够在正确的环境和参数下顺利完成。
            */
          }




          template <typename InputT, typename OutputT>
          absl::Status DoGemm(int64_t batch_size, int64_t m, int64_t n, int64_t k,
                              const MatrixDescriptor& lhs, const MatrixDescriptor& rhs,
                              const MatrixDescriptor& c, const MatrixDescriptor& output,
                              se::DeviceMemoryBase bias, float alpha, float beta,
                              SYCLGemm::GemmBackendEpilogue epilogue, se::Stream* stream,
                              std::optional<se::blas::AlgorithmType> algorithm,
                              se::ScratchAllocator* scratch_allocator,
                              se::blas::ComputePrecision compute_precision) 
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
                              
                              
            {
              /* 
              根据算法选择来执行矩阵乘法计算的。它决定使用 Xetla 硬件加速的 GEMM（General Matrix Multiply）操作还是使用 oneDNN 库中的 GEMM 操作
              上面定义好的  两种矩阵执行操作，这里进行选择
              
              这段代码的核心功能是根据用户选择的算法类型，调用相应的函数来执行矩阵乘法计算：
              如果选择了 Xetla GEMM 算法，调用 DoXetlaGemm 函数。
              如果选择了其他算法（即默认使用 oneDNN），调用 DoOnednnGemm 函数。
              通过这种方式，代码能够根据不同的计算需求和硬件特性，灵活选择最合适的算法和实现方式，以优化矩阵乘法的性能和计算效率。
              
              */
            if (algorithm == se::blas::kXetlaGemm) {  // 检查是否选择了 Xetla GEMM 算法， algorithm 是一个可选的枚举值，代表要使用的计算算法。se::blas::kXetlaGemm 是一个枚举值，表示使用 Xetla GEMM 算法
              VLOG(1) << "Run Xetla gemm kernel";  // 记录日志，显示即将运行 Xetla GEMM 内核。VLOG 是一个日志记录函数，1 表示日志的优先级，优先级较低，通常用于调试信息
              return DoXetlaGemm<InputT>(batch_size, m, n, k, lhs, rhs, c, output, bias,
                                        alpha, beta, epilogue, stream, algorithm,
                                        scratch_allocator, compute_precision);
                    /*
                    调用 DoXetlaGemm 函数来执行 Xetla GEMM 操作
                    DoXetlaGemm 是一个函数模板，用于执行 Xetla 硬件加速的 GEMM 操作
                    各参数传递给 DoXetlaGemm 函数，具体包括矩阵尺寸和描述符、偏置数据、标量参数（alpha、beta）、后处理操作（epilogue）、计
                    算流（stream）、算法类型（algorithm）、内存分配器（scratch_allocator）和计算精度（compute_precision）。
                    */ 
            } else {   // 如果算法不是 Xetla GEMM，则执行以下代码块，当选择的算法不是 Xetla GEMM 时，执行 oneDNN GEMM 操作
              VLOG(1) << "Run OneDnn gemm kernel"; //运行 oneDNN GEMM 内核
              return DoOnednnGemm<InputT, OutputT>(
                  batch_size, m, n, k, lhs, rhs, c, output, bias, alpha, beta, epilogue,
                  stream, algorithm, scratch_allocator, compute_precision);
                  /*
                  调用 DoOnednnGemm 函数来执行 oneDNN GEMM 操作
                  DoOnednnGemm 是一个函数模板，用于执行 oneDNN 库中的 GEMM 操作
                  各参数传递给 DoOnednnGemm 函数，具体包括矩阵尺寸和描述符、偏置数据、标量参数（alpha、beta）、后处理操作（epilogue）、计算流（stream）、
                  算法类型（algorithm）、内存分配器（scratch_allocator）和计算精度（compute_precision）。
                  */
            }
          }


          // 下面这段代码定义了三个函数，用于处理矩阵描述符，使其与 BLAS 的 GEMM 操作兼容。具体来说，它们处理了矩阵转置的问题，因为 BLAS GEMM 操作不支持转置输出矩阵
          void TransposeMatrixDesc(MatrixDescriptor& matrix_desc) {
            matrix_desc.transpose =
                (matrix_desc.transpose == se::blas::Transpose::kNoTranspose)
                    ? se::blas::Transpose::kTranspose
                    : se::blas::Transpose::kNoTranspose;
          }
          /*
          功能：转置一个矩阵描述符。
          作用：
          检查 matrix_desc.transpose 是否为 se::blas::Transpose::kNoTranspose。
          如果是，将其设置为 se::blas::Transpose::kTranspose。
          否则，将其设置回 se::blas::Transpose::kNoTranspose。
          上下文：用于在需要时将矩阵描述符标记为转置或不转置。
          
          */



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
            /*
            功能：使矩阵描述符与 BLAS GEMM 操作兼容。
            作用：
            检查 output.transpose 是否为 se::blas::Transpose::kTranspose。
            如果是，使用矩阵乘法的一个等式来调整矩阵描述符，使其符合 BLAS GEMM 的要求：
            C T=(A@B) T=B T @ A T
            具体步骤：
            交换 lhs 和 rhs。
            转置 lhs、rhs 和 output。
            上下文：调整矩阵描述符，使其符合 BLAS GEMM 不支持转置输出的限制。
            */
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
              /*
              功能：使矩阵描述符与 BLAS GEMM 操作兼容，并且处理偏置矩阵 c。
                作用：
                检查 output.transpose 是否为 se::blas::Transpose::kTranspose。
                如果是，使用矩阵乘法的一个等式来调整矩阵描述符，使其符合 BLAS GEMM 的要求：
                𝐶𝑇=(𝐴 @ 𝐵)𝑇=𝐵𝑇 @ 𝐴𝑇 

                具体步骤：
                交换 lhs 和 rhs。
                转置 lhs、rhs、output 和 c。
                上下文：调整矩阵描述符，使其符合 BLAS GEMM 不支持转置输出的限制，同时处理偏置矩阵 c。
              
              */

        }
      }
    }  // namespace




    absl::Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
                        se::DeviceMemoryBase rhs_buffer,
                        se::DeviceMemoryBase c_buffer,
                        se::DeviceMemoryBase output_buffer,
                        se::DeviceMemoryBase bias_buffer, se::Stream* stream,
                        SYCLGemm::GemmBackendEpilogue epilogue,
                        se::ScratchAllocator* scratch_allocator)
                /*
                整个GEMM操作的入口点

                这段代码定义了一个函数 RunGemm，用于运行矩阵乘法（GEMM，General Matrix Multiplication）。具体来说，它配置并执行矩阵乘法操作
                RunGemm 函数根据提供的配置和输入参数，选择适当的矩阵乘法实现并执行操作。
                函数首先初始化矩阵布局，提取矩阵维度和描述符，然后确保这些描述符与 BLAS 的 GEMM 操作兼容。接着，通过检查操作数类型，选择并调用相应的 DoGemm 函数。如果没有匹配的类型组合，则返回错误状态
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


                参数详解
                const GemmConfig& config：矩阵乘法配置，包括维度、数据类型、算法等信息。
                se::DeviceMemoryBase lhs_buffer：左手边矩阵的设备内存。
                se::DeviceMemoryBase rhs_buffer：右手边矩阵的设备内存。
                se::DeviceMemoryBase c_buffer：可选的中间矩阵的设备内存。
                se::DeviceMemoryBase output_buffer：输出矩阵的设备内存。
                se::DeviceMemoryBase bias_buffer：可选的偏置矩阵的设备内存。
                se::Stream stream*：指向用于执行操作的计算流的指针。
                SYCLGemm::GemmBackendEpilogue epilogue：后处理操作，如ReLU、GELU等。
                se::ScratchAllocator scratch_allocator*：用于分配临时工作空间的分配器。
                */         
      
      {       
            // 日志记录 使用 VLOG 记录执行 GEMM 操作的日志，用于调试和跟踪
            VLOG(2) << "Executing a GemmThunk";
            
            // 矩阵布局初始化  初始化左手边矩阵、右手边矩阵、输出矩阵和中间矩阵的布局。
            auto lhs_layout = MatrixLayout{config.lhs_layout},      // 获取左手边矩阵布局 (lhs_layout),从 config.lhs_layout 中获取左手边矩阵的布局信息，
                output_layout = MatrixLayout{config.output_layout}, // 并使用这些信息初始化一个 MatrixLayout 对象。MatrixLayout 是一个结构体或类，用于描述矩阵的布局信息，如维度、数据类型、步幅等
                rhs_layout = MatrixLayout{config.rhs_layout},
                c_layout = MatrixLayout{config.c_layout};
                /*
                背景信息
                config：这是一个包含矩阵乘法（GEMM）配置的对象或结构体，包含了各个矩阵的布局信息。
                MatrixLayout：这是一个结构体或类，用于描述矩阵的布局信息。通常包括矩阵的维度（行数、列数、批处理大小）、
                数据类型（如 float32、int8）、步幅（strides），以及其他相关的元数据。

                目的
                初始化矩阵布局对象，以便在后续的 GEMM 操作中使用这些布局信息。这些布局信息将用于配置和执行矩阵乘法操作，
                确保矩阵的维度和数据类型正确匹配。
                                
                */



            // 提取矩阵维度  提取输出矩阵的行数 m、列数 n 以及左手边矩阵的列数 k。
            int64_t m = output_layout.num_rows;
            int64_t n = output_layout.num_cols;
            int64_t k = lhs_layout.num_cols;
            
            // 获取矩阵描述符, 使用 GetMatrixDesc 函数获取矩阵描述符，这些描述符包含矩阵的具体信息（如数据类型、步幅等）。
            // 从矩阵布局和设备内存缓冲区中创建 MatrixDescriptor 对象，并获取批处理大小
            MatrixDescriptor lhs = GetMatrixDesc(lhs_layout, lhs_buffer);
                /*
                获取左手边矩阵描述符 (lhs)
                调用 GetMatrixDesc 函数，将 lhs_layout 和 lhs_buffer 作为参数，创建并返回一个 MatrixDescriptor 对象。
                lhs_layout 包含左手边矩阵的布局信息，lhs_buffer 是左手边矩阵的数据缓冲区。
                如下的结构是 一样的
                */
            MatrixDescriptor rhs = GetMatrixDesc(rhs_layout, rhs_buffer);
            MatrixDescriptor c = GetMatrixDesc(c_layout, c_buffer);
            MatrixDescriptor output = GetMatrixDesc(output_layout, output_buffer);
            int64_t batch_size = output_layout.batch_size;
            

            // 兼容 BLAS 的 GEMM 操作  调用 MakeBlasGemmCompatible 函数，确保矩阵描述符与 BLAS 的 GEMM 操作兼容。
            MakeBlasGemmCompatible(lhs, rhs, c, output);

            // 定义操作数类型   创建一个包含操作数数据类型的元组
            std::tuple operand_types{lhs_layout.dtype, rhs_layout.dtype,
                                    output_layout.dtype};
          
            // 类型检查与调用相应的 GEMM 函数
            // 这段代码使用宏 TYPED_GEMM 来根据不同的数据类型调用 DoGemm 函数。具体来说，它检查 operand_types 是否与某个类型组合匹配，如果匹配，则使用这些类型调用 DoGemm
           
            #define TYPED_GEMM(ATYPE, BTYPE, CTYPE)    // 定义了一个宏 TYPED_GEMM，它接受三个类型参数 ATYPE、BTYPE 和 CTYPE。                                   
              if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE)) {    // 检查 operand_types 是否等于由 ATYPE、BTYPE 和 CTYPE 组成的元组。如果 operand_types 与这个元组匹配，则进入代码块。    
                using NativeAType = PrimitiveTypeToXetlaNative<ATYPE>::type;  // 定义 NativeAType，使用 PrimitiveTypeToXetlaNative 模板将 ATYPE 转换为对应的本机类型，并定义类型别名 NativeAType。              
                using NativeCType = PrimitiveTypeToXetlaNative<CTYPE>::type;  // 定义 NativeCType,使用 PrimitiveTypeToXetlaNative 模板将 CTYPE 转换为对应的本机类型，并定义类型别名 NativeCType     
                
                return DoGemm<NativeAType, NativeCType>(                                  
                    batch_size, m, n, k, lhs, rhs, c, output, bias_buffer,                
                    config.alpha.real(), config.beta, epilogue, stream, config.algorithm, 
                    scratch_allocator, config.compute_precision);        
                    /*
                    调用 DoGemm 函数
                    使用 NativeAType 和 NativeCType 调用 DoGemm 函数，并传入相关参数：

                    batch_size：批处理大小。
                    m：矩阵 lhs 的行数。
                    n：矩阵 rhs 的列数。
                    k：矩阵 lhs 的列数。
                    lhs、rhs、c、output：矩阵描述符。
                    bias_buffer：偏置数据。
                    config.alpha.real() 和 config.beta：标量乘子。
                    epilogue：后处理操作。
                    stream：计算流。
                    config.algorithm：GEMM 算法。
                    scratch_allocator：临时工作空间分配器。
                    config.compute_precision：计算精度。
                    */                 
              }
                
              // 使用宏 TYPED_GEMM 定义一系列条件判断，检查操作数类型是否匹配预定义的类型组合。如果匹配，则调用相应的 DoGemm 函数
              // 在使用宏的部分，代码通过多个 TYPED_GEMM 调用来处理不同的类型组合：
              // 这些调用分别检查 operand_types 是否与以下类型组合匹配，并相应地调用 DoGemm 函数
              TYPED_GEMM(BF16, BF16, BF16)
              TYPED_GEMM(F16, F16, F16)
              TYPED_GEMM(BF16, BF16, F32)
              TYPED_GEMM(F16, F16, F32)
              TYPED_GEMM(F32, F32, F32)
              TYPED_GEMM(S8, S8, S32)

            // 返回错误状态, 如果没有匹配的类型组合，则返回错误状态，并提供详细的错误信息
            #undef TYPED_GEMM  // 取消定义宏 TYPED_GEMM： 这行代码取消之前定义的宏 TYPED_GEMM。宏在被使用完后取消定义是一个好的编程实践，可以避免宏定义在其他地方被意外使用或重新定义 
              return Internal( // 返回内部错误信息
                  "Unexpected GEMM lhs type %s, rhs type %s and output type %s",  // 这是错误消息的格式字符串，表示遇到了意外的 GEMM 输入矩阵、右侧矩阵和输出矩阵的数据类型
                  primitive_util::LowercasePrimitiveTypeName(lhs_layout.dtype),
                  primitive_util::LowercasePrimitiveTypeName(rhs_layout.dtype),
                  primitive_util::LowercasePrimitiveTypeName(output_layout.dtype));
        }
  }  // namespace gpu
}  // namespace xla
