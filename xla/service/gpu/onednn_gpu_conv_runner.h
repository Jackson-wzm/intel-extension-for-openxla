/* Copyright (c) 2024 Intel Corporation

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
// 宏定义和包含头文件，这些宏定义用于防止头文件被多次包含。它们确保了头文件中的内容只会被编译一次。
#ifndef XLA_SERVICE_GPU_ONEDNN_GPU_CONV_RUNNER_H_ 
#define XLA_SERVICE_GPU_ONEDNN_GPU_CONV_RUNNER_H_


//这些包含指令引入了其他头文件，这些头文件包含了conv的配置和临时内存分配器的定义
#include <optional>

#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/onednn_util.h"

namespace xla {

namespace gpu { //这些命名空间用于组织代码，避免命名冲突，并表示这些功能属于 XLA 的 GPU 后端部分

typedef struct OneDnnConvPrimitive {
  dnnl::memory src_memory;              // 源（输入）内存对象，用于存储输入数据
  dnnl::memory filter_memory;           // 滤波器（权重）内存对象，用于存储卷积核
  dnnl::memory dst_memory;              // 目标（输出）内存对象，用于存储输出数据
  dnnl::memory internal_filter_memory;  // 内部滤波器内存对象，用于存储内部格式的卷积核数据
  dnnl::memory scratchpad_memory;       // 临时内存对象，用于存储计算过程中需要的临时数据
  dnnl::memory bias_memory;             // 偏置内存对象，用于存储偏置数据
  dnnl::convolution_forward fwd_primitive;                  // 卷积前向操作原语，用于执行前向卷积
  dnnl::convolution_backward_data bwd_input_primitive;      // 卷积反向数据操作原语，用于执行反向数据传播
  dnnl::convolution_backward_weights bwd_filter_primitive;  // 卷积反向权重操作原语，用于执行反向权重更新
  dnnl::reorder filter_reorder_primitive;                   // 重排操作原语，用于调整滤波器数据格式

  std::unordered_map<int, dnnl::memory> fwd_primitives_args;       // 前向操作原语参数映射，键为参数标识，值为内存对象
  std::unordered_map<int, dnnl::memory> bwd_input_primitive_args;  // 反向数据操作原语参数映射，键为参数标识，值为内存对象
  std::unordered_map<int, dnnl::memory> bwd_filter_primitive_args; // 反向权重操作原语参数映射，键为参数标识，值为内存对象

  std::unordered_map<int, dnnl::memory> reorder_args;              // 重排操作参数映射，键为参数标识，值为内存对象

  dnnl::engine engine;        // OneDNN引擎对象，用于管理计算资源
  dnnl::stream stream;        // OneDNN流对象，用于管理计算流
  bool has_reorder = false;   // 布尔标识，指示是否需要重排操作
} OneDnnConvPrimitive;


absl::StatusOr<OneDnnConvPrimitive> GetOrCreateOneDnnConvPrimitive(
    se::Stream*, const ffi::Dictionary& dict,
    absl::flat_hash_map<std::string, std::string>& backend_dict,
    const std::vector<ffi::BufferBase>& operand_se_buffers,
    const ffi::BufferBase& result_buffer,
    se::ScratchAllocator* scratch_allocator, CudnnConvKind conv_kind);
  /*
  这个函数的作用是获取或创建一个 OneDnnConvPrimitive 对象。它可能会检查已有的卷积原语是否可用，
  如果不可用则创建一个新的卷积原语对象，并进行相应的配置
  
  参数：

  se::Stream* stream:          指向计算流的指针，表示在该流上进行计算。
  const ffi::Dictionary& dict: 常量引用，表示卷积操作的配置字典，包含各种配置参数。
  absl::flat_hash_map<std::string, std::string>& backend_dict: 后端字典，键值对表示后端相关的配置和参数。
  const std::vector<ffi::BufferBase>& operand_se_buffers: 操作数的缓冲区向量，包含输入和权重等数据。
  const ffi::BufferBase& result_buffer: 结果缓冲区，表示卷积操作的输出数据存储位置。
  se::ScratchAllocator* scratch_allocator: 指向临时内存分配器的指针，用于在计算过程中分配临时内存。
  CudnnConvKind conv_kind: 枚举类型，表示卷积操作的类型（例如前向卷积、反向数据传播、反向权重更新）。
    
  返回值：
  返回 absl::StatusOr<OneDnnConvPrimitive> 类型，表示返回值可能是一个 OneDnnConvPrimitive 对象或一个错误状态。
  OneDnnConvPrimitive 对象包含了卷积操作所需的各种原语和内存对象。
  作用：
  这个函数的作用是获取或创建一个 OneDnnConvPrimitive 对象。它可能会检查已有的卷积原语是否可用，如果不可用则创建一个新的卷积原语对象，并进行相应的配置。
  */

absl::Status RunGpuConv(const OneDnnConvPrimitive& onednn_primitive,
                        const ffi::Dictionary& dict,
                        absl::Span<const ffi::BufferBase> operand_buffers,
                        ffi::BufferBase result_buffer, CudnnConvKind conv_kind);
            /*
            absl::Status RunGemm(...)：这是一个函数声明，用于运行矩阵乘法操作。
            
            
            */
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_ONEDNN_GPU_CONV_RUNNER_H_  结尾宏定义
        //这行宏定义结束了防止头文件多次包含的机制。
		
