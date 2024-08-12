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

/*
在 C++ 开发中
.h 文件（头文件）
主要用途：
1. 声明接口：头文件主要用于声明函数、类、变量和其他接口。它们不包含具体实现，只是提供接口说明。
2. 包含保护：头文件通常使用包含保护（include guards）来防止重复包含。包含保护的典型形式
        #ifndef FILENAME_H
        #define FILENAME_H

        // 头文件内容

        #endif // FILENAME_H
3.共享声明：通过头文件，多个源文件可以共享同一个声明，从而使代码模块化，易于维护和扩展。

.cc 文件（源文件）
主要用途：
定义实现：源文件包含函数和类的具体实现。头文件中声明的所有接口在源文件中定义。
编译单元：每个源文件是一个独立的编译单元，编译器会单独编译每个源文件，并生成对应的目标文件（例如 .o 或 .obj 文件）。
包含头文件：源文件通常会包含它们对应的头文件以及其他必要的头文件，以确保所有声明都有定义。

区别：
内容：

头文件主要包含声明（declarations）。
源文件主要包含定义（definitions）。
文件类型：

头文件通常以 .h 结尾。
源文件通常以 .cc、.cpp 或 .c 结尾。
编译：

头文件本身不直接编译。
源文件被编译器编译成目标文件。
联系：
相互依赖：源文件依赖头文件中的声明来知道如何使用某些函数或类。头文件定义了接口，源文件实现了这些接口。
包含关系：源文件通常包含它们对应的头文件。例如，example.cc 包含 example.h，以确保它实现的接口是正确的。
模块化：头文件和源文件一起使代码模块化。头文件提供模块的接口，源文件提供模块的实现。
总结
头文件（.h 文件）主要用于声明接口，使多个源文件可以共享相同的声明。
源文件（.cc 文件）主要用于实现头文件中声明的接口，并作为独立的编译单元进行编译。
头文件和源文件的结合使代码更模块化、结构化和易于维护。
*/




/*
这段代码定义了一个头文件 onednn_matmul_utils.h，用于在 XLA (Accelerated Linear Algebra) 的 GPU 后端实现与 oneDNN 库相关的矩阵乘法操作的实用工具
这个头文件定义了一些与 SYCL 和 oneDNN 相关的矩阵乘法操作的实用工具，
主要包含了后处理类型的枚举、几个转换和检查函数，以及一个用于运行矩阵乘法操作的函数。
通过这些定义:可以在 XLA 的 GPU 后端中更方便地管理和调用这些矩阵乘法操作,规定了 矩阵乘法的具体操作
*/




#ifndef XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_   // 宏定义和包含头文件，这些宏定义用于防止头文件被多次包含。它们确保了头文件中的内容只会被编译一次。
#define XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_   

#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/scratch_allocator.h" //这些包含指令引入了其他头文件，这些头文件包含了矩阵乘法的配置和临时内存分配器的定义

namespace xla {
namespace gpu {  这些命名空间用于组织代码，避免命名冲突，并表示这些功能属于 XLA 的 GPU 后端部分

namespace SYCLGemm{                      子命名空间 SYCLGemm, 用于包含与 SYCL (一种用于异构计算的并行编程模型) 和 Gemm (通用矩阵乘法) 相关的功能。
                                        
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
    
    //这些函数声明用于将字符串和 GemmBackendEpilogue 枚举之间进行转换，检查 GemmBackendEpilogue 是否添加向量偏置，是否有辅助输出，
    //以及将 GemmBackendConfig_Epilogue 转换为 GemmBackendEpilogue。


    absl::StatusOr<GemmBackendEpilogue> EpilogueCast(std::string& epilogue);  这个函数声明定义了一个名为 EpilogueCast 的函数，用于将一个字符串转换为 GemmBackendEpilogue 枚举类型
        ########
        absl::StatusOr 
            absl::StatusOr 是 Abseil 库中的一个模板类，用于表示一个函数返回值可能是某个类型的值（在这里是 GemmBackendEpilogue），也可能是一个错误状态（absl::Status）。
            //它可以有效地处理函数返回的错误情况，并且在出错时提供详细的错误信息。
            //这个类型允许函数返回一个有效的 GemmBackendEpilogue 值，或者返回一个错误状态，表示无法成功完成转换。
                
        <GemmBackendEpilogue>
           // 这是一个枚举类型，定义了几种可能的矩阵乘法后处理选项，例如 DEFAULT、RELU、GELU 等。

        EpilogueCast
            //函数名称:
            //EpilogueCast 是函数的名称,表示将一个字符串转换(cast)为 GemmBackendEpilogue 枚举类型。

        std::string& epilogue
            //参数类型:
            std::string& 表示参数是一个字符串的引用。引用意味着传递的是变量的地址，而不是变量的拷贝，这样可以避免拷贝开销并允许在函数中修改原始变量。
            //使用引用而不是指针的另一个好处是语法上更简洁，更易于使用。
            //参数名称:
            //epilogue 是参数的名称，表示要转换的字符串。这个字符串应该包含一个有效的后处理选项的名称（如 "RELU"、"GELU" 等）

        //函数的功能
            //根据函数声明，可以推测 EpilogueCast 函数的主要功能是将传入的字符串 epilogue 转换为对应的 GemmBackendEpilogue 枚举值。如果字符串有效且匹配某个枚举值，
            //函数将返回对应的 GemmBackendEpilogue 值。如果字符串无效或不匹配任何枚举值，函数将返回一个错误状态。

        //函数具体实现
            //这个函数的具体实现 是在 onednn_matmul_utils.cc 文件里面进行定义。详细解释看 onednn_matmul_utils.cc  文件
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

            //返回值:
                return GemmBackendEpilogue::DEFAULT; //等语句返回匹配的 GemmBackendEpilogue 枚举值。
                return absl::InvalidArgumentError("Invalid epilogue string: " + epilogue); 返回一个错误状态，表示输入的字符串不合法。
                //通过这种设计，调用者可以通过检查返回值来确认操作是否成功，并在出现错误时获取详细的错误信息。


    absl::StatusOr<std::string> EpilogueCast(GemmBackendEpilogue epilogue);
        //这段代码定义了一个函数 EpilogueCast，它将 GemmBackendEpilogue 枚举值转换为对应的字符串。函数返回一个 absl::StatusOr<std::string> 类型，
        //用于表示操作是否成功，并包含转换后的字符串或错误状态
        absl::StatusOr<std::string>: 返回类型，表示函数的返回值可能是一个字符串或一个错误状态。
        GemmBackendEpilogue epilogue: 参数类型，表示传入的 GemmBackendEpilogue 枚举值。
        /*
        函数具体实现
            这个函数的具体实现 是在 onednn_matmul_utils.cc 文件里面进行定义。详细解释看 onednn_matmul_utils.cc  文件
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
            absl::StatusOr 用于返回值的错误处理，确保函数能够优雅地处理错误情况。
            switch 语句 用于将输入的 GemmBackendEpilogue 枚举值转换为相应的字符串。
            错误处理 确保当输入的枚举值不匹配任何已知值时，返回一个描述错误的状态。
        */
    absl::StatusOr<bool> EpilogueAddsVectorBias(GemmBackendEpilogue epilogue);
        代码定义了一个函数 EpilogueAddsVectorBias，它判断传入的 GemmBackendEpilogue 枚举值是否对应于添加向量偏置（Vector Bias）的情况。
        函数返回一个 absl::StatusOr<bool> 类型，用于表示操作是否成功，并包含布尔值结果或错误状态
        absl::StatusOr 用于返回值的错误处理，确保函数能够优雅地处理错误情况。
        switch 语句 用于将输入的 GemmBackendEpilogue 枚举值映射到对应的布尔值，表示是否添加向量偏置。
        错误处理 确保当输入的枚举值不匹配任何已知值时，返回一个描述错误的状态。


    absl::StatusOr<bool> EpilogueHasAuxiliaryOutput(GemmBackendEpilogue epilogue);
        定义了一个函数 EpilogueHasAuxiliaryOutput，它判断传入的 GemmBackendEpilogue 枚举值是否对应于具有辅助输出的情况。
        函数返回一个 absl::StatusOr<bool> 类型，用于表示操作是否成功，并包含布尔值结果或错误状态

    absl::StatusOr<GemmBackendEpilogue> AsSYCLEpilogue(GemmBackendConfig_Epilogue epilogue);
        定义了一个函数 AsSYCLEpilogue，它将 GemmBackendConfig_Epilogue 类型的值转换为 GemmBackendEpilogue 类型。函数返回一个 absl::StatusOr<GemmBackendEpilogue> 类型，
        用于表示操作是否成功，并包含转换后的枚举值或错误状态
        absl::StatusOr 用于返回值的错误处理，确保函数能够优雅地处理错误情况。
        switch 语句 用于将输入的 GemmBackendConfig_Epilogue 枚举值映射到对应的 GemmBackendEpilogue 枚举值。
        错误处理 确保当输入的枚举值不匹配任何已知值时，返回一个描述错误的状态


}

//RunGemm 函数声明
absl::Status RunGemm(const GemmConfig& config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase add_buffer,
               se::DeviceMemoryBase output_buffer,
               se::DeviceMemoryBase bias_buffer, se::Stream* stream,
               SYCLGemm::GemmBackendEpilogue epilogue,
               se::ScratchAllocator* scratch_allocator = nullptr);
        /*
        absl::Status RunGemm(...)：这是一个函数声明，用于运行矩阵乘法操作。
        const GemmConfig& config：矩阵乘法的配置。
        se::DeviceMemoryBase lhs_buffer 和 se::DeviceMemoryBase rhs_buffer：左侧和右侧矩阵的数据缓冲区。
        se::DeviceMemoryBase add_buffer 和 se::DeviceMemoryBase output_buffer：附加数据缓冲区和输出缓冲区。
        se::DeviceMemoryBase bias_buffer：偏置数据缓冲区。
        se::Stream* stream：计算流，用于在设备上执行操作。
        SYCLGemm::GemmBackendEpilogue epilogue：枚举类型，指定后处理类型。
        se::ScratchAllocator* scratch_allocator = nullptr：可选的临时内存分配器，默认为空。

        */

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ONEDNN_MATMUL_UTILS_H_  结尾宏定义
        //这行宏定义结束了防止头文件多次包含的机制。
