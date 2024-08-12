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

#include "xla/service/gpu/onednn_gpu_conv_runner.h"
#include <string>
#include "xla/service/gpu/scratch_allocator.h"
#include "xla/service/gpu/stream_executor_util.h"

namespace xla {
namespace gpu { // 这些命名空间用于组织代码，避免命名冲突，并表示这些功能属于 XLA 的 GPU 后端部分

/*
命名空间 se::dnn 中的类型别名
这些 using 语句创建了类型别名，主要目的是简化代码中的类型名称，方便后续使用。这些别名来自命名空间 se::dnn，
涉及设备内存管理、流管理以及深度神经网络（DNN）操作的描述符。
*/

using se::DeviceMemory;               // 用于表示设备（如 GPU）内存的类型，用于管理和访问设备内存。
using se::DeviceMemoryBase;           // 用于表示设备（如 GPU）内存的类型，用于管理和访问设备内存。
using se::Stream;                     // 表示计算流或命令队列，用于在设备上执行操作
using se::dnn::AlgorithmConfig;       // 用于配置算法的参数，如选择最佳算法进行卷积运算
using se::dnn::BatchDescriptor;       // 描述批量输入数据的形状和格式
using se::dnn::ConvolutionDescriptor; // 描述卷积运算的参数，如卷积核大小、步幅和填充方式
using se::dnn::DataLayout;            // 描述数据布局，如 NCHW（批量、通道、高度、宽度）或 NHWC（批量、高度、宽度、通道）。
using se::dnn::DimIndex;              // 数据维度的索引，通常用于访问特定维度的数据
using se::dnn::FilterDescriptor;      //描述 卷积核 （卷积核）的形状和格式
using se::dnn::FilterLayout;          // 描述 卷积核 的数据布局。
using se::dnn::ProfileResult;         //存储算法性能分析的结果，如执行时间。

// 使用 dnnl 命名空间的类型别名
using ConvFwdPd = dnnl::convolution_forward::primitive_desc;
/*
表示前向卷积操作的描述符，包含了卷积操作的具体参数和配置。
调用位置
Onednn: https://oneapi-src.github.io/oneDNN/struct_dnnl_convolution_backward_data_primitive_desc.html

*/

using ConvBwdInputPd = dnnl::convolution_backward_data::primitive_desc;
/*
表示反向传播时输入梯度计算的卷积操作描述符
*/


using ConvBwdFilterPd = dnnl::convolution_backward_weights::primitive_desc;
/*
表示反向传播时 卷积核  梯度计算 的卷积(卷积核 filter)操作描述符
卷积核（Convolution Kernel），也叫 卷积核 （filter）

卷积核怎么理解？就比如：
带有横条纹特征的卷积核就容易把原图中的横条纹识别出来；
带有竖条纹特征的卷积核就容易把原图中的竖条纹识别出来；
带有眼睛特征（类似躺倒的8字）的卷积核就容易把原图中的眼睛识别出来

*/


using ConvBwdFilterPrimitive = dnnl::convolution_backward_weights;
/*
表示执行反向传播时 卷积核  梯度计算  的卷积 操作对象。
*/

namespace {
    /*
    定义了两个用于获取向量化大小的函数，根据数据布局或 卷积核 布局的不同返回不同的值
    处理深度学习中的卷积操作，因为向量化处理可以显著提高计算效率。通过返回不同的向量化大小，
    这些函数允许代码在不同的数据或 卷积核 布局下进行优化

    函数 GetVectCSize(DataLayout layout)
      这个函数根据给定的数据布局（DataLayout）返回向量化的大小（VectCSize），即在数据处理过程中一次处理多少个元素。
      不同的数据布局可能有不同的向量化大小。
        函数签名：
        int64_t GetVectCSize(DataLayout layout)：函数返回一个 64 位的整数，参数是一个 DataLayout 类型的值。
        
        switch 语句：
        根据 layout 的值选择不同的执行路径。
        
        case DataLayout::kBatchDepthYX4：
        如果 layout 是 DataLayout::kBatchDepthYX4，则返回 4。
        表示数据布局为 kBatchDepthYX4 时，向量化大小为 4。
            kBatchDepthYX4 是一种数据布局（Data Layout），通常在机器学习和高性能计算（HPC）中用于描述多维数组（张量）的存储方式。它主要用于指定数据在内存中的排列顺序和访问模式，以便优化计算性能，尤其是在卷积神经网络（CNN）中。

              解释 kBatchDepthYX4
              kBatchDepthYX4 代表一种特定的张量数据布局模式：
              Batch：表示批处理（Batch），通常指多个样本一起处理，以提高计算效率。
              Depth：表示每个样本的深度（Depth），例如图像的通道数（Channels）。
              YX：表示空间维度，即高度（Y）和宽度（X）。
              4：表示在这个特定布局中，每个深度维度的数据块大小为 4。也就是说，深度维度被分成了大小为 4 的小块。
              
              具体的内存布局  [2, 2, 4, 4, 4]
              kBatchDepthYX4 的内存布局可以理解为一个 5 维的张量，通常以以下顺序存储：[Batch, Depth/4, Y, X, 4]。 [2, 2, 4, 4, 4]
              这意味着：

              Batch：批处理维度，表示样本的数量。
              Depth/4：深度维度分块后的维度，表示每个深度维度被分成了大小为 4 的小块。
              Y：高度维度，表示图像或特征图的高度。
              X：宽度维度，表示图像或特征图的宽度。
              4：每个小块的大小，表示深度维度的具体数据块。
              这种布局方式可以通过将深度维度分块来优化内存访问模式，提高缓存命中率和数据传输效率。

              示例
              假设有一个张量，其原始尺寸为 [Batch, Depth, Y, X]，具体为 [2, 8, 4, 4]。使用 kBatchDepthYX4 布局，
              这个张量将被重排为 [2, 2, 4, 4, 4]，其中 Depth 被分成了 8/4 = 2 个大小为 4 的块。
              伪代码示例
              int batch_size = 2;
              int depth = 8;
              int height = 4;
              int width = 4;

              float tensor[2][8][4][4]; // 原始张量

              // 转换为 kBatchDepthYX4 布局
              float tensor_kBatchDepthYX4[2][2][4][4][4]; // 转换后的张量

              for (int b = 0; b < batch_size; ++b) {
                for (int d = 0; d < depth; ++d) {
                  for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                      int new_d = d / 4;
                      int block = d % 4;
                      tensor_kBatchDepthYX4[b][new_d][y][x][block] = tensor[b][d][y][x];
                    }
                  }
                }
              }

              tensor 是原始张量，tensor_kBatchDepthYX4 是使用 kBatchDepthYX4 布局后的张量。通过这种重排方式，可以优化 GPU 内存访问模式，提高计算效率
              优点
              内存访问优化：通过分块，可以减少全局内存访问，提高缓存命中率。
              并行计算优化：这种布局方式更适合 GPU 的并行计算模型，提高数据并行度。
              计算效率：在卷积操作中，可以更高效地利用硬件加速功能，提高计算速度。
              总之，kBatchDepthYX4 是一种优化内存访问和计算效率的张量数据布局方式，特别适用于高性能计算和深度学习中的卷积操作。
        
        
        
        case DataLayout::kBatchDepthYX32：

        如果 layout 是 DataLayout::kBatchDepthYX32，则返回 32。
        表示数据布局为 kBatchDepthYX32 时，向量化大小为 32。
          kBatchDepthYX32 是另一种数据布局（Data Layout），与 kBatchDepthYX4 类似，但它将深度维度（Depth）分成了大小为 32 的小块。
          这种布局方式通常用于优化高性能计算，尤其是在卷积神经网络（CNN）中，提高计算效率。
          假设有一个张量，其原始尺寸为 [Batch, Depth, Y, X]，具体为 [2, 64, 4, 4]。使用 kBatchDepthYX32 布局，
          这个张量将被重排为 [2, 2, 4, 4, 32]，其中 Depth 被分成了 64/32 = 2 个大小为 32 的块。

          解释 kBatchDepthYX32
          kBatchDepthYX32 代表一种特定的张量数据布局模式：

          Batch：表示批处理（Batch），通常指多个样本一起处理，以提高计算效率。
          Depth：表示每个样本的深度（Depth），例如图像的通道数（Channels）。
          YX：表示空间维度，即高度（Y）和宽度（X）。
          32：表示在这个特定布局中，每个深度维度的数据块大小为 32。也就是说，深度维度被分成了大小为 32 的小块。
          
          具体的内存布局
          kBatchDepthYX32 的内存布局可以理解为一个 5 维的张量，通常以以下顺序存储：[Batch, Depth/32, Y, X, 32]。 [2, 2, 4, 4, 32]
          这意味着：

          Batch：批处理维度，表示样本的数量。
          Depth/32：深度维度分块后的维度，表示每个深度维度被分成了大小为 32 的小块。
          Y：高度维度，表示图像或特征图的高度。
          X：宽度维度，表示图像或特征图的宽度。
          32：每个小块的大小，表示深度维度的具体数据块。

        default：

        如果 layout 是其他任何值，则返回 1。
        表示默认情况下，向量化大小为 1。

    */ 
  // 函数签名：
  int64_t GetVectCSize(DataLayout layout) {
    // switch 语句
    switch (layout) {
      case DataLayout::kBatchDepthYX4:
        return 4;
      case DataLayout::kBatchDepthYX32:
        return 32;
      default:
        return 1;
    }
  }


  /*
  函数 GetVectCSize(FilterLayout layout)
  这个函数与前一个函数类似，但它是针对 卷积核 布局（FilterLayout）的。根据给定的 卷积核 布局返回向量化的大小
  具体解释：
  函数签名：

  int64_t GetVectCSize(FilterLayout layout)：函数返回一个 64 位的整数，参数是一个 FilterLayout 类型的值。
  switch 语句：

  根据 layout 的值选择不同的执行路径。
  case FilterLayout::kOutputInputYX4：

  如果 layout 是 FilterLayout::kOutputInputYX4，则返回 4。
  表示 卷积核 布局为 kOutputInputYX4 时，向量化大小为 4。
  case FilterLayout::kOutputInputYX32：

  如果 layout 是 FilterLayout::kOutputInputYX32，则返回 32。
  表示 卷积核 布局为 kOutputInputYX32 时，向量化大小为 32。
  default：

  如果 layout 是其他任何值，则返回 1。
  表示默认情况下，向量化大小为 1。
  */
  int64_t GetVectCSize(FilterLayout layout) {
    switch (layout) {
      case FilterLayout::kOutputInputYX4:
        return 4;
      case FilterLayout::kOutputInputYX32:
        return 32;
      default:
        return 1;
    }
  }


  // 创建 OneDnn 卷积原语函数
  absl::Status CreateOneDnnPrimitive(
      OneDnnConvPrimitive* onednn_primitive,  // 一个指向 OneDnnConvPrimitive 结构的指针，这个结构包含了所有需要的 OneDNN 卷积操作的相关信息
      const ffi::Dictionary& dict,  //一个字典（Dictionary）对象，包含了描述卷积操作的相关参数和配置，比如卷积核大小、步长、填充等
      absl::flat_hash_map<std::string, std::string>& backend_dict, //一个哈希表，用于存储后端相关的配置信息，通常用于区分不同计算后端（如 GPU、CPU 等）的特殊配置
      absl::Span<const ffi::BufferBase> operand_buffers, // 一个缓冲区的数组（Span），包含了操作数的缓冲区。这些缓冲区通常包括输入数据和卷积核数据
      ffi::BufferBase result_buffer, // 一个缓冲区，存储卷积操作的结果数据 
      se::Stream* stream, // 一个指向计算流（Stream）的指针，用于异步计算操作。它管理着计算设备（如 GPU）的执行队列。
      se::ScratchAllocator* scratch_allocator, // 一个指向内存分配器的指针，用于在计算过程中分配临时内存。
      CudnnConvKind conv_kind) // 枚举类型 CudnnConvKind，表示卷积操作的类型，比如前向卷积（Forward）、反向输入卷积（Backward Data）或反向权重卷积（Backward Filter）
  {


    // 将 TensorFlow 的计算流（stream）转换为 DPC++ 的计算流
    // stream 是 TensorFlow 的计算流对象，se::gpu::AsGpuStreamValue(stream) 将其转换为 DPC++ 的 sycl::queue 类型。
    sycl::queue* dpcpp_stream = se::gpu::AsGpuStreamValue(stream);

    /*
    查找或创建一个 OneDNN 的 engine，并将其绑定到计算流上
    FindOrCreateEngine 函数根据 dpcpp_stream 查找或创建一个 OneDNN 的 engine 对象。
    dnnl::sycl_interop::make_stream 函数将 engine 与 dpcpp_stream 绑定，创建一个 OneDNN 的 stream。
    */
    onednn_primitive->engine = FindOrCreateEngine(dpcpp_stream);
    onednn_primitive->stream =
        dnnl::sycl_interop::make_stream(onednn_primitive->engine, *dpcpp_stream);
    
    // 从字典中获取输入、 卷积核 和输出的数据布局类型
    /*
    dict.get<int32_t>("input_dl") 从字典中获取表示输入数据布局的整数值，并转换为 DataLayout 枚举类型。
    类似地，获取表示 卷积核 和输出数据布局的整数值，并分别转换为 FilterLayout 和 DataLayout 枚举类型。
    */
    DataLayout input_dl = static_cast<DataLayout>(*dict.get<int32_t>("input_dl"));
    FilterLayout filter_dl =static_cast<FilterLayout>(*dict.get<int32_t>("filter_dl"));
    DataLayout output_dl =static_cast<DataLayout>(*dict.get<int32_t>("output_dl"));

    // 初始化卷积所需的各种参数
    PrimitiveType input_type, filter_type, output_type;
    absl::Span<const int64_t> input_dimensions, filter_dimensions,
        output_dimensions;
    void* input_data;   // 初始化表示输入、 卷积核 和输出数据类型的变量。
    void* filter_data; // 初始化表示输入、 卷积核 和输出数据维度的变量。
    void* output_data; // 初始化表示输入、 卷积核 和输出数据的指针变量。
    void* bias_data = nullptr; // 初始化表示偏置和辅助输入数据的指针变量，初始值为 nullptr。
    void* side_input_data = nullptr;

    // 从 backend_dict 中获取卷积结果的缩放比例，并检查其是否为 1
    /*
    std::stof(backend_dict["conv_result_scale"]) 将 backend_dict 中表示卷积结果缩放比例的字符串转换为浮点数。
    conv_result_scale_one 检查缩放比例是否接近 1（即 fabs(conv_result_scale - 1.0f) < 1e-6）。
    */
    float conv_result_scale = std::stof(backend_dict["conv_result_scale"]);
    bool conv_result_scale_one = (fabs(conv_result_scale - 1.0f) < 1e-6);


    // 这段代码处理了不同类型的卷积操作，初始化了与这些操作相关的参数，包括输入、 卷积核 和输出的数据类型、维度以及指针。
    // 根据卷积操作的类型（前向、后向输入、后向 卷积核 ），代码会设置相应的数据参数。

    switch (conv_kind) {
      // 前向卷积
      case CudnnConvKind::kForward:

      // 前向带激活的前向卷积
      case CudnnConvKind::kForwardActivation:
        /*
        CudnnConvKind::kForward 和 CudnnConvKind::kForwardActivation 代表前向卷积和带激活的前向卷积。
        设置输入数据类型为第一个操作数的类型。
        设置输入、 卷积核 和输出的数据维度分别为第一个操作数、第二个操作数和结果缓冲区的维度。
        设置输入、 卷积核 和输出的数据指针分别为第一个操作数、第二个操作数和结果缓冲区的数据指针。
        */

        // 设置输入数据类型
        input_type = operand_buffers[0].dtype;

        // 设置输入、 卷积核 和输出的数据维度
        input_dimensions = operand_buffers[0].dimensions;
        filter_dimensions = operand_buffers[1].dimensions;
        output_dimensions = result_buffer.dimensions;

        // 设置输入、 卷积核 和输出的数据指针
        input_data = const_cast<void*>(operand_buffers[0].data.opaque());
        filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
        output_data = const_cast<void*>(result_buffer.data.opaque());
        break;
      
      // 后向输入卷积
      case CudnnConvKind::kBackwardInput:
      /*
      CudnnConvKind::kBackwardInput 代表后向输入卷积。
        设置输入数据类型为结果缓冲区的类型。
        设置输入、 卷积核 和输出的数据维度分别为结果缓冲区、第二个操作数和第一个操作数的维度。
        设置输入、 卷积核 和输出的数据指针分别为结果缓冲区、第二个操作数和第一个操作数的数据指针。
      */
      // 设置输入数据类型
        input_type = result_buffer.dtype;
      
      // 设置输入、 卷积核 和输出的数据维度
        input_dimensions = result_buffer.dimensions;
        filter_dimensions = operand_buffers[1].dimensions;
        output_dimensions = operand_buffers[0].dimensions;

        // 设置输入、 卷积核 和输出的数据指针
        input_data = const_cast<void*>(result_buffer.data.opaque());
        filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
        output_data = const_cast<void*>(operand_buffers[0].data.opaque());

        break;

        // 后向 卷积核 卷积
      case CudnnConvKind::kBackwardFilter:
        /*
        CudnnConvKind::kBackwardFilter 代表后向 卷积核 卷积。
        设置输入数据类型为第一个操作数的类型。
        设置输入、 卷积核 和输出的数据维度分别为第一个操作数、结果缓冲区和第二个操作数的维度。
        设置输入、 卷积核 和输出的数据指针分别为第一个操作数、结果缓冲区和第二个操作数的数据指针。
        
        */
      // 设置输入数据类型
        input_type = operand_buffers[0].dtype;

      // 设置输入、 卷积核 和输出的数据维度
        input_dimensions = operand_buffers[0].dimensions;
        filter_dimensions = result_buffer.dimensions;
        output_dimensions = operand_buffers[1].dimensions;

      // 设置输入、 卷积核 和输出的数据指针
        input_data = const_cast<void*>(operand_buffers[0].data.opaque());
        filter_data = const_cast<void*>(result_buffer.data.opaque());
        output_data = const_cast<void*>(operand_buffers[1].data.opaque());

        break;
      // 未知的卷积类型，返回错误
      default:
        return Internal("Unkown convolution kind");
    }


    // 这段代码处理了卷积操作中前向带激活操作的特殊情况。它从操作数缓冲区中提取偏置数据和侧输入数据，
    // 并相应地设置了侧输入的缩放比例和是否为零的标志。
    
    /*
    初始化变量:
    side_input_scale: 用于存储侧输入的缩放比例。
    side_input_scale_zero: 用于标志侧输入的缩放比例是否为零。
    */

    float side_input_scale;
    bool side_input_scale_zero;


    /*
    判断卷积类型:

    如果卷积类型是 CudnnConvKind::kForwardActivation（前向带激活卷积），则执行以下步骤：
    从 operand_buffers 中提取偏置数据的指针，并将其存储在 bias_data 中。
    如果 operand_buffers 的大小至少为 4，说明存在侧输入数据：
    从 operand_buffers 中提取侧输入数据的指针，并将其存储在 side_input_data 中。
    从 backend_dict 中提取侧输入的缩放比例，并将其存储在 side_input_scale 中。
    计算侧输入的缩放比例是否为零，并将结果存储在 side_input_scale_zero 中。
    */

    // 判断卷积类型:
    // 如果卷积类型是前向带激活操作
    if (conv_kind == CudnnConvKind::kForwardActivation) {
      // 提取侧输入数据指针
      bias_data = const_cast<void*>(operand_buffers[2].data.opaque());
      // 如果操作数缓冲区大小至少为4
      if (operand_buffers.size() >= 4) {
        // 提取侧输入数据指针
        side_input_data = const_cast<void*>(operand_buffers[3].data.opaque());
        // 从backend_dict中提取侧输入的缩放比例
        side_input_scale = std::stof(backend_dict["side_input_scale"]);
        // 判断侧输入缩放比例是否为零
        side_input_scale_zero = (fabs(side_input_scale - 0.0f) < 1e-6);
      }
    }



    /*
    提取和检查维度数量 提取窗口维度数量
    提取和检查维度数量:
    从 dict 中提取 window_num_dimensions（窗口维度数量），并将其存储在 num_dimensions 中。
    使用 CHECK_LE 宏检查 num_dimensions 是否小于或等于 3。如果 num_dimensions 超过 3，会触发检查失败。
    */ 
    const int num_dimensions = *dict.get<int32_t>("window_num_dimensions");
    // 检查维度数量不超过3
    CHECK_LE(num_dimensions, 3);

    // OneDNN does not support 1D convolutions. We therefore express 1D
    // convolutions as 2D convolutions where the first spatial dimension 空间维度 is 1.
    // This matches the behavior of TF (see definition of conv1d in
    // tensorflow/python/ops/nn_ops.py).
    
    /*
    从输入和 卷积核 维度中提取必要的信息，以确定卷积操作的形状参数。它根据维度的数量，分别处理一维、二维和三维卷积的情况
    */

    // 确定有效维度 确保 effective_num_dimensions 至少为 2，方便后续处理卷积操作的维度。
    const int effective_num_dimensions = std::max(2, num_dimensions);
    //提取输入通道数和批量大小， 这两行代码提取输入特征维度和批量维度，并计算输入通道数 ic 和批量大小 n。
    int ic = GetVectCSize(input_dl) *
            input_dimensions[*dict.get<int64_t>("input_feature_dimension")];
    int n = input_dimensions[*dict.get<int64_t>("input_batch_dimension")];
    
    // 提取输入的空间维度 根据 num_dimensions 的值提取输入数据的空间维度：
    int id, ih, iw;
    if (num_dimensions == 3) {
      id = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_0")];
      ih = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_1")];
      iw = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_2")];
    } else if (num_dimensions == 2) {
      ih = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_0")];
      iw = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_1")];
    } else if (num_dimensions == 1) {
      ih = 1;
      iw = input_dimensions[*dict.get<int64_t>("input_spatial_dimensions_0")];
    } else if (num_dimensions == 0) {
      ih = 1;
      iw = 1;
    } else {
      return Internal("Invalid convolution dimension num");
    }

    // 提取 卷积核 的空间维度： 根据 num_dimensions 的值提取 卷积核 的空间维度
    int kd, kh, kw;
    if (num_dimensions == 3) {
      kd = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_0")];
      kh = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_1")];
      kw = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_2")];
    } else if (num_dimensions == 2) {
      kh = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_0")];
      kw = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_1")];
    } else if (num_dimensions == 1) {
      kh = 1;
      kw = filter_dimensions[*dict.get<int64_t>("kernel_spatial_dimensions_0")];
    } else if (num_dimensions == 0) {
      kh = 1;
      kw = 1;
    } else {
      return Internal("Invalid convolution dimension num");
    }


    // 这段代码处理了卷积层的  组卷积  和深度卷积，并从字典中提取了填充、步幅和扩展参数
      /*
      组卷积（group convolution）是一种卷积操作，最初是为了减少计算量和参数数量引入的，但后来在深度学习的架构中（如ResNeXt和MobileNet）得到了广泛应用。
      在组卷积中，输入通道和输出通道被分成若干组，每组独立地执行卷积操作。
      优点
      计算效率：组卷积减少了计算量和参数数量。
      性能提升：对于大规模的卷积神经网络，组卷积可以显著提高性能和减少内存使用。
      通过组卷积，可以更高效地利用计算资源，同时也可以增强模型的性能和表达能力


      深度卷积
      深度卷积（depthwise convolution）是一种特殊类型的卷积操作，它主要用于提高计算效率和减少参数数量，广泛应用于轻量级神经网络模型（如MobileNet）。
      与标准卷积不同，深度卷积对每个输入通道独立进行卷积，而不是将所有输入通道与所有输出通道相连接
      基本概念
      在标准卷积中，每个输出通道是所有输入通道的线性组合。标准卷积的计算复杂度为： 𝑂(𝐾⋅𝐾⋅𝐶𝑖𝑛⋅𝐶𝑜𝑢𝑡⋅𝐻⋅𝑊) 其中：
      𝐾 是卷积核的大小。
      𝐶𝑖𝑛是输入通道数。
      𝐶𝑜𝑢𝑡 是输出通道数。
      H 和 W 分别是输入特征图的高度和宽度。

      在深度卷积中，每个输入通道仅与一个对应的卷积核进行卷积，输出通道数等于输入通道数。深度卷积的计算复杂度为：𝑂(𝐾⋅𝐾⋅𝐶𝑖𝑛⋅𝐻⋅𝑊)
      深度卷积的优点
      计算效率高：计算量和参数数量显著减少。
      适用于轻量级网络：适用于移动设备等计算资源有限的环境。
      深度卷积在实际应用中的扩展
      为了进一步提高效率，深度卷积常与逐点卷积（pointwise convolution，1x1卷积）结合使用。这种组合称为深度可分离卷积（depthwise separable convolution）。
      在深度可分离卷积中，首先对每个输入通道独立进行深度卷积，然后通过逐点卷积将结果进行线性组合。这种方法在保持准确度的同时，显著减少了计算量和参数数量。

      深度卷积及其扩展技术在现代卷积神经网络中得到了广泛应用，特别是在需要高效计算的场景中，如移动端和嵌入式设备

      */


    // It is group-conv if filter_in != src_in   如果 filter_ic 和 ic 不同，表示这是组卷积。G 是组的数量，O 是每组的输出通道数。
    // G = src_in/filter_in
    // O = filter_out/G
    // TODO: depthwise-conv
    
    //提取 卷积核 的输入和输出通道数:  filter_ic 表示 卷积核 的输入通道数。filter_oc 表示 卷积核 的输出通道数。
    int filter_ic =
        filter_dimensions[*dict.get<int64_t>("kernel_input_feature_dimension")];
    int filter_oc =
        filter_dimensions[*dict.get<int64_t>("kernel_output_feature_dimension")];
    
    // 判断是否是组卷积:  如果输入通道数 ic 和 卷积核 的输入通道数 filter_ic 不同，则为组卷积。
    bool is_group_conv = ic != filter_ic;
    
    //计算组卷积参数
    int kg = ic / filter_ic;  // kg for group-conv and depthwise-conv  kg 表示组卷积的组数。
    int ko = filter_oc / kg;  // ko 表示每组的输出通道数
    int ki = filter_ic;       // ki 表示每组的输入通道数。

    // 详细逻辑解释
    /*
    定义填充、步幅和扩展参数：
    padding_d_l, padding_h_l, padding_w_l 表示深度、高度和宽度方向的低填充。
    padding_d_h, padding_h_h, padding_w_h 表示深度、高度和宽度方向的高填充。
    stride_d, stride_h, stride_w 表示深度、高度和宽度方向的步幅。
    dilate_d, dilate_h, dilate_w 表示深度、高度和宽度方向的扩展。
    */
    int padding_d_l, padding_h_l, padding_w_l;
    int padding_d_h, padding_h_h, padding_w_h;
    int stride_d, stride_h, stride_w, dilate_d, dilate_h, dilate_w;

    /*
    处理三维卷积
    从字典中提取三维卷积的填充、步幅和扩展参数。
    padding_d_l, padding_h_l, padding_w_l 分别对应深度、高度和宽度方向的低填充。
    padding_d_h, padding_h_h, padding_w_h 分别对应深度、高度和宽度方向的高填充。
    stride_d, stride_h, stride_w 分别对应深度、高度和宽度方向的步幅。
    dilate_d, dilate_h, dilate_w 分别对应深度、高度和宽度方向的扩展。
    */
    if (num_dimensions == 3) {
      padding_d_l = *dict.get<int64_t>("window_padding_low_0");
      padding_h_l = *dict.get<int64_t>("window_padding_low_1");
      padding_w_l = *dict.get<int64_t>("window_padding_low_2");
      padding_d_h = *dict.get<int64_t>("window_padding_high_0");
      padding_h_h = *dict.get<int64_t>("window_padding_high_1");
      padding_w_h = *dict.get<int64_t>("window_padding_high_2");

      stride_d = *dict.get<int64_t>("window_stride_0");
      stride_h = *dict.get<int64_t>("window_stride_1");
      stride_w = *dict.get<int64_t>("window_stride_2");

      dilate_d = *dict.get<int64_t>("window_dilation_0");
      dilate_h = *dict.get<int64_t>("window_dilation_1");
      dilate_w = *dict.get<int64_t>("window_dilation_2");
    } else if (num_dimensions == 2) {  // 处理二维卷积
      padding_h_l = *dict.get<int64_t>("window_padding_low_0");
      padding_w_l = *dict.get<int64_t>("window_padding_low_1");
      padding_h_h = *dict.get<int64_t>("window_padding_high_0");
      ;
      padding_w_h = *dict.get<int64_t>("window_padding_high_1");

      stride_h = *dict.get<int64_t>("window_stride_0");
      stride_w = *dict.get<int64_t>("window_stride_1");

      dilate_h = *dict.get<int64_t>("window_dilation_0");
      dilate_w = *dict.get<int64_t>("window_dilation_1");
    } else if (num_dimensions == 1) { // 处理一维卷积  从字典中提取一维卷积的填充、步幅和扩展参数， 一维卷积的高度填充和步幅固定为 1，宽度方向的参数从字典中提取
      padding_h_l = 0;
      padding_w_l = *dict.get<int64_t>("window_padding_low_0");
      padding_h_h = 0;
      padding_w_h = *dict.get<int64_t>("window_padding_high_0");

      stride_h = 1;
      stride_w = *dict.get<int64_t>("window_stride_0");

      dilate_h = 1;
      dilate_w = *dict.get<int64_t>("window_dilation_0");
    } else if (num_dimensions == 0) {  // 处理零维卷积， 零维卷积的填充、步幅和扩展参数均设置为 0 或 1
      padding_h_l = 0;
      padding_w_l = 0;
      padding_h_h = 0;
      padding_w_h = 0;

      stride_h = 1;
      stride_w = 1;

      dilate_h = 1;
      dilate_w = 1;
    }

    // 这段代码处理了输出张量的维度信息，根据不同的维度数（num_dimensions），提取相应的输出张量维度参数
    /*
    定义输出张量维度
    od 表示输出张量在深度方向的维度。
    oh 表示输出张量在高度方向的维度。
    ow 表示输出张量在宽度方向的维度。
    
    */
    int od, oh, ow;

    // 提取输出通道数， 从字典中获取输出张量的通道数
    int oc = output_dimensions[*dict.get<int64_t>("output_feature_dimension")];
    
    // 处理三维卷积输出维度， 从字典中提取输出张量在深度、高度和宽度方向的维度
    if (num_dimensions == 3) {
      od = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_0")];
      oh = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_1")];
      ow = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_2")];
    // 处理二维卷积输出维度 从字典中提取输出张量在高度和宽度方向的维度
    } else if (num_dimensions == 2) {
      oh = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_0")];
      ow = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_1")];
    // 处理一维卷积输出维度，将输出张量的高度维度设为 1， 从字典中提取输出张量在宽度方向的维度。
    } else if (num_dimensions == 1) {
      oh = 1;
      ow = output_dimensions[*dict.get<int64_t>("output_spatial_dimensions_0")];

    // 处理零维卷积输出维度，将输出张量的高度和宽度维度均设为 1。
    } else if (num_dimensions == 0) {
      oh = 1;
      ow = 1;
    }


    // 这段代码是用来配置并初始化 OneDNN（DNNL）中的卷积操作，包括处理不同的输入、 卷积核 和输出格式，以及不同的维度（2D 或 3D）卷积操作
    // 判断是否为3D卷积
    bool is_conv3d = (num_dimensions == 3);
    // 定义卷积所需的各种维度和格式标签
    try {
      dnnl::memory::dims src_dims, filter_dims, bias_dims, dst_dims, stride_dims,
          padding_dims_l, padding_dims_r, dilation_dims;
      dnnl::memory::format_tag src_fmt, weight_fmt, dst_fmt;
      
      // 处理2D卷积的维度和格式
      // 处理2D卷积情况， 根据是否是组卷积，设置 卷积核 维度。设置输入、输出、偏置、步幅、填充和扩展维度
      if (!is_conv3d) {
        src_dims = {n, ic, ih, iw};
        if (is_group_conv)
          filter_dims = {kg, ko, ki, kh, kw};
        else
          filter_dims = {ko, ki, kh, kw};
        bias_dims = {oc};
        dst_dims = {n, oc, oh, ow};
        stride_dims = {stride_h, stride_w};
        padding_dims_l = {padding_h_l, padding_w_l};
        padding_dims_r = {padding_h_h, padding_w_h};
        dilation_dims = {dilate_h - 1, dilate_w - 1};

        // 处理输入格式， 根据输入数据布局设置源格式
        switch (input_dl) {
          case DataLayout::kBatchDepthYX:
            src_fmt = dnnl::memory::format_tag::nchw;
            break;
          case DataLayout::kBatchYXDepth:
            src_fmt = dnnl::memory::format_tag::nhwc;
            break;
          default:
            return Internal("Unsupported convolution input format");
        }

        // 处理 卷积核 格式， 根据 卷积核 数据布局设置 卷积核 格式
        switch (filter_dl) {
          case FilterLayout::kOutputInputYX:
            weight_fmt = is_group_conv ? dnnl::memory::format_tag::goihw
                                      : dnnl::memory::format_tag::oihw;
            break;
          case FilterLayout::kOutputYXInput:
            weight_fmt = is_group_conv ? dnnl::memory::format_tag::gohwi
                                      : dnnl::memory::format_tag::ohwi;
            break;
          case FilterLayout::kYXInputOutput:
            weight_fmt = is_group_conv ? dnnl::memory::format_tag::hwigo
                                      : dnnl::memory::format_tag::hwio;
            break;
          default:
            return Internal("Unsupported convolution weight format");
        }

        // 处理输出格式，根据输出数据布局设置目标格式。
        switch (output_dl) {
          case DataLayout::kBatchDepthYX:
            dst_fmt = dnnl::memory::format_tag::nchw;
            break;
          case DataLayout::kBatchYXDepth:
            dst_fmt = dnnl::memory::format_tag::nhwc;
            break;
          default:
            return Internal("Unsupported convolution output format");
        }
      
      // 处理3D卷积的维度和格式， 处理3D卷积情况，设置输入、输出、偏置、步幅、填充和扩展维度
      } else {
        src_dims = {n, ic, id, ih, iw};
        /*
        src_dims：源数据的维度。
        n：批量大小（batch size）。
        ic：输入通道数（input channels）。
        id：输入的深度（depth），适用于3D卷积。
        ih：输入的高度（height）。
        iw：输入的宽度（width）。
        这行代码定义了输入数据的形状，适用于3D卷积操作
        
        */
        if (is_group_conv)
          filter_dims = {kg, ko, ki, kd, kh, kw};
          /*
          filter_dims： 卷积核 的维度。
          如果是组卷积（is_group_conv 为 true）：
          kg：组数（groups）。
          ko：每组输出通道数（output channels per group）。
          ki：每组输入通道数（input channels per group）。
          kd： 卷积核 的深度（kernel depth）。
          kh： 卷积核 的高度（kernel height）。
          kw： 卷积核 的宽度（kernel width）
          
          如果不是组卷积：
          ko：输出通道数（output channels）。
          ki：输入通道数（input channels）。
          kd： 卷积核 的深度（kernel depth）。
          kh： 卷积核 的高度（kernel height）。
          kw： 卷积核 的宽度（kernel width）。
          */
        else
          filter_dims = {ko, ki, kd, kh, kw};

        bias_dims = {oc};   // bias_dims：偏置的维度， oc：输出通道数（output channels）。这行代码定义了偏置的形状。
        dst_dims = {n, oc, od, oh, ow};  
          /*
          输出维度（dst_dims） 
          dst_dims：目标数据的维度。
          n：批量大小（batch size）。
          oc：输出通道数（output channels）。
          od：输出的深度（depth），适用于3D卷积。
          oh：输出的高度（height）。
          ow：输出的宽度（width）。
          这行代码定义了输出数据的形状，适用于3D卷积操作
          */
        
        stride_dims = {stride_d, stride_h, stride_w};
          /*
          stride_dims：步幅的维度。
          stride_d：深度方向的步幅（stride in depth）。
          stride_h：高度方向的步幅（stride in height）。
          stride_w：宽度方向的步幅（stride in width）。
          这行代码定义了卷积操作的步幅。
          */
        padding_dims_l = {padding_d_l, padding_h_l, padding_w_l};
          /*
          padding_dims_l：填充低维度。
          padding_d_l：深度方向的低填充（padding low in depth）。
          padding_h_l：高度方向的低填充（padding low in height）。
          padding_w_l：宽度方向的低填充（padding low in width）。
          */
        padding_dims_r = {padding_d_h, padding_h_h, padding_w_h};
          /*
          padding_dims_r：填充高维度。
          padding_d_h：深度方向的高填充（padding high in depth）。
          padding_h_h：高度方向的高填充（padding high in height）。
          padding_w_h：宽度方向的高填充（padding high in width）。
          这段代码定义了卷积操作的填充参数
          */
        dilation_dims = {dilate_d - 1, dilate_h - 1, dilate_w - 1};
          /*
          dilation_dims：扩展的维度。
          dilate_d - 1：深度方向的扩展（dilation in depth）。
          dilate_h - 1：高度方向的扩展（dilation in height）。
          dilate_w - 1：宽度方向的扩展（dilation in width）。
          这行代码定义了卷积操作的扩展参数。注意扩展值通常减1，因为卷积框架的扩展定义和DNNL的定义略有不同。
          */


        // 处理3D输入格式， 根据输入数据布局设置源格式
        switch (input_dl) {
          case DataLayout::kBatchDepthYX:
            src_fmt = dnnl::memory::format_tag::ncdhw;
            break;
          case DataLayout::kBatchYXDepth:
            src_fmt = dnnl::memory::format_tag::ndhwc;
            break;
          default:
            return Internal("Unsupported convolution input format");
        }

        // 处理3D 卷积核 格式， 根据 卷积核 数据布局设置 卷积核 格式
        switch (filter_dl) {
          case FilterLayout::kOutputInputYX:
            weight_fmt = is_group_conv ? dnnl::memory::format_tag::goidhw
                                      : dnnl::memory::format_tag::oidhw;
            break;
          case FilterLayout::kOutputYXInput:
            weight_fmt = is_group_conv ? dnnl::memory::format_tag::godhwi
                                      : dnnl::memory::format_tag::odhwi;
            break;
          default:
            return Internal("Unsupported convolution weight format");
        }

        // 处理3D输出格式 根据输出数据布局设置目标格式
        switch (output_dl) {
          case DataLayout::kBatchDepthYX:
            dst_fmt = dnnl::memory::format_tag::ncdhw;
            break;
          case DataLayout::kBatchYXDepth:
            dst_fmt = dnnl::memory::format_tag::ndhwc;
            break;
          default:
            return Internal("Unsupported convolution output format");
        }
      }

      // 内存类型  kind：设置为dnnl::sycl_interop::memory_kind::usm，表示使用统一共享内存（USM）
      auto kind = dnnl::sycl_interop::memory_kind::usm;

      // 数据类型选择
      dnnl::memory::data_type data_type;
      // 
      switch (input_type) {   // input_type：输入数据的类型
        /*
        通过switch语句，根据input_type设置对应的DNNL数据类型data_type。
        BF16对应dnnl::memory::data_type::bf16。
        F32对应dnnl::memory::data_type::f32。
        F16对应dnnl::memory::data_type::f16。
        F64对应dnnl::memory::data_type::f64。
        S8对应dnnl::memory::data_type::s8。
        S32对应dnnl::memory::data_type::s32。
        如果输入类型不支持，返回内部错误信息。
        
        */
        case BF16:
          data_type = dnnl::memory::data_type::bf16;
          break;
        case F32:
          data_type = dnnl::memory::data_type::f32;
          break;
        case F16:
          data_type = dnnl::memory::data_type::f16;
          break;
        case F64:
          data_type = dnnl::memory::data_type::f64;
          break;
        case S8:
          data_type = dnnl::memory::data_type::s8;
          break;
        case S32:
          data_type = dnnl::memory::data_type::s32;
          break;
        default:
          return Internal("Unsupported convolution input data type");
      }

      // 内存描述符初始化
      /*
      src_md：源数据的内存描述符。
      {src_dims}：源数据的维度。
      data_type：数据类型。
      src_fmt：源数据的格式（如nchw或nhwc）。
      filter_md： 卷积核 的内存描述符。
      {filter_dims}： 卷积核 的维度。
      data_type：数据类型。
      weight_fmt： 卷积核 数据的格式（如oihw或hwio）。
      dst_md：目标数据的内存描述符。
      {dst_dims}：目标数据的维度。
      data_type：数据类型。
      dst_fmt：目标数据的格式（如nchw或nhwc）。
      这些描述符定义了卷积操作中输入、 卷积核 和输出数据的形状、数据类型和内存布局，从而使DNNL能够正确执行卷积操作
      */
      dnnl::memory::desc src_md =
          dnnl::memory::desc({src_dims}, data_type, src_fmt);
      dnnl::memory::desc filter_md =
          dnnl::memory::desc({filter_dims}, data_type, weight_fmt);
      dnnl::memory::desc dst_md =
          dnnl::memory::desc({dst_dims}, data_type, dst_fmt);


      /*
      这段代码的目的是配置并创建OneDNN（DNNL）卷积操作的内存对象（memory objects），包括源数据、 卷积核 数据和目标数据。
      这些内存对象将被传递给OneDNN卷积原语以执行实际的卷积运算
      */

      //定义一个布尔变量并从环境变量读取其值
      /*
      bool flag = false;：定义并初始化一个布尔变量flag为false。
      tsl::ReadBoolFromEnvVar("ONEDNN_PLAIN_WEIGHT", false, &flag);：从环境变量ONEDNN_PLAIN_WEIGHT读取一个布尔值，并将其赋值给flag。
      如果环境变量不存在或不可读取，则使用默认值false。
      */
      bool flag = false;
      tsl::ReadBoolFromEnvVar("ONEDNN_PLAIN_WEIGHT", false, &flag);

      /*
      根据环境变量设置 卷积核 内存描述符
      */ 
      dnnl::memory::desc filter_md_prefer = dnnl::memory::desc({filter_dims}, data_type, dnnl::memory::format_tag::any);
      /*
      创建一个 卷积核 内存描述符filter_md_prefer，使用filter_dims、data_type和格式标签dnnl::memory::format_tag::any。
      dnnl::memory::format_tag::any表示允许OneDNN自动选择最合适的内存格式。
      */

      if (flag)
        filter_md_prefer = dnnl::memory::desc({filter_dims}, data_type, weight_fmt);
        // 如果flag为true，则使用明确指定的 卷积核 格式weight_fmt重新创建 卷积核 内存描述符filter_md_prefer
      
      // 创建源、 卷积核 和目标内存对象
      onednn_primitive->src_memory = dnnl::sycl_interop::make_memory(
          src_md, onednn_primitive->engine, kind, input_data);
      onednn_primitive->filter_memory = dnnl::sycl_interop::make_memory(
          filter_md, onednn_primitive->engine, kind, filter_data);
      onednn_primitive->dst_memory = dnnl::sycl_interop::make_memory(
          dst_md, onednn_primitive->engine, kind, output_data);
      /*
      onednn_primitive->src_memory：创建源内存对象。
        使用内存描述符src_md、引擎onednn_primitive->engine、内存类型kind和源数据指针input_data。
      onednn_primitive->filter_memory：创建 卷积核 内存对象。
        使用内存描述符filter_md、引擎onednn_primitive->engine、内存类型kind和 卷积核 数据指针filter_data。
      onednn_primitive->dst_memory：创建目标内存对象。
        使用内存描述符dst_md、引擎onednn_primitive->engine、内存类型kind和目标数据指针output_data。
      
      这些内存对象将在后续的卷积操作中使用，确保数据按照指定的格式和维度进行存储和访问
      */




      // if alpha is 1:
      //   out = activation(conv(x, w, bias) + beta * side)
      //   po.append_sum(beta)
      //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);
      // else:
      //   out = activation(alpha * conv(x, w) + beta * side + bias)
      //   po.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0);
      //   po.append_sum(beta)
      //   po.append_binary(1, bias);
      //   po.append_eltwise(dnnl::algorithm::activation, 1, 0);

      /*
      这段代码配置了用于卷积运算的后处理操作（post-operations, post-ops），包括激活函数、偏置加法和其他操作。
      这些后处理操作通过OneDNN的dnnl::post_ops和dnnl::primitive_attr进行配置，并应用于卷积原语。
      
      */
    // 定义后处理操作对象和原语属性对象
      dnnl::post_ops po;  // 创建一个后处理操作对象po。
      dnnl::primitive_attr post_ops_attr;  // 创建一个原语属性对象post_ops_attr。

    // 配置卷积结果的缩放操作
      if (!conv_result_scale_one)
        po.append_eltwise(dnnl::algorithm::eltwise_linear, conv_result_scale, 0);
        /*
        检查conv_result_scale是否为1（conv_result_scale_one为true）。
        如果不是1，添加一个线性变换操作，用于缩放卷积结果。eltwise_linear表示线性缩放，参数conv_result_scale是缩放系数，0是偏移量。
        */
      
      // 配置旁路输入的缩放操作 
      // 如果存在旁路输入数据（side_input_data不为空）且缩放系数side_input_scale不为0，添加一个求和操作。side_input_scale是旁路输入的缩放系数
      if (side_input_data && !side_input_scale_zero)
        po.append_sum(side_input_scale);  // 

      // 配置偏置的加法操作
      if (!conv_result_scale_one && bias_data) {
        auto bias_post_md =
            dnnl::memory::desc(bias_dims, data_type, dnnl::memory::format_tag::x);
        po.append_binary(dnnl::algorithm::binary_add, bias_post_md);
        onednn_primitive->bias_memory = dnnl::sycl_interop::make_memory(
            bias_post_md, onednn_primitive->engine, kind, bias_data);
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(po.len() - 1) | DNNL_ARG_SRC_1,
            onednn_primitive->bias_memory});
      }
      /*
      如果conv_result_scale不是1并且存在偏置数据（bias_data不为空），添加一个二元加法操作，用于将偏置加到卷积结果中。
      创建一个偏置内存描述符bias_post_md，使用偏置维度、数据类型和格式标签x（表示一维）。
      使用偏置内存描述符、引擎和偏置数据创建偏置内存对象，并存储在onednn_primitive->bias_memory中。
      将偏置内存对象添加到卷积原语的参数中，使用DNNL_ARG_ATTR_MULTIPLE_POST_OP和DNNL_ARG_SRC_1表示这是一个多重后处理操作中的一个源数据。
          
      */


    // 配置激活函数操作： 后处理操作确保在卷积运算完成后，结果可以通过指定的激活函数、偏置加法和旁路输入缩放等操作进行进一步处理，从而满足不同的计算需求。
    /*
    如果卷积类型是前向激活（kForwardActivation），根据backend_dict中的activation_mode选择适当的激活函数并添加到后处理操作中。
    不同的激活函数如Sigmoid、Relu、Relu6、Tanh、Elu和LeakyRelu分别对应不同的DNNL算法。

    为什么需要加入激活函数？
    卷积运算结束之后，生成的是 特征图，接下来还有一个步骤：激活。
    就是让卷积之后的特征结果更明显， 比如 Relu的作用就是让得到负值的都归零，得正值的还是其本身。这样就能把这个特征给凸显出来了，使用不同的 激活函数进行下面的选择运算
    */

    if (conv_kind == CudnnConvKind::kForwardActivation) {
      if (backend_dict["activation_mode"] == "kNone") {
      } else if (backend_dict["activation_mode"] == "kSigmoid") {
        po.append_eltwise(dnnl::algorithm::eltwise_logistic, 1, 0);
      } else if (backend_dict["activation_mode"] == "kRelu") {
        po.append_eltwise(dnnl::algorithm::eltwise_relu, 0, 0);
      } else if (backend_dict["activation_mode"] == "kRelu6") {
        po.append_eltwise(dnnl::algorithm::eltwise_clip_v2, 0, 6);
      } else if (backend_dict["activation_mode"] == "kTanh") {
        po.append_eltwise(dnnl::algorithm::eltwise_tanh, 0, 0);
      } else if (backend_dict["activation_mode"] == "kElu") {
        po.append_eltwise(dnnl::algorithm::eltwise_elu, 1, 0);
      } else if (backend_dict["activation_mode"] == "kLeakyRelu") {
        float leakyrelu_alpha = std::stof(backend_dict["leakyrelu_alpha"]);
        po.append_eltwise(dnnl::algorithm::eltwise_relu, leakyrelu_alpha, 0);
      } else {
        return Internal("Unsupported Activation mode");
      }
    }
    // 设置卷积原语的后处理操作属性
    post_ops_attr.set_post_ops(po);  // 将配置好的后处理操作po设置到原语属性post_ops_attr中。
    post_ops_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);  // 设置原语属性中的暂存区模式为用户模式。


    // Set fp32 mode. 设置FP32数学模式
    // 这段代码主要设置了FP32数学模式，并创建卷积前向（或前向激活）原语描述符，配置相应的内存对象和参数

    // 设置FP32数学模式：
    dnnl::fpmath_mode fp32_math_mode = GetFP32MathMode(); // 调用GetFP32MathMode函数获取FP32数学模式，并将其存储在fp32_math_mode变量中。
    if (input_type == F32) {  // 检查输入数据类型是否为F32。
      post_ops_attr.set_fpmath_mode(fp32_math_mode);  // 如果输入数据类型为F32，将FP32数学模式设置到卷积原语属性中。
    }
    
    // 处理卷积前向和前向激活的情况
    if (conv_kind == CudnnConvKind::kForward || conv_kind == CudnnConvKind::kForwardActivation) {  // 检查卷积类型是否为前向卷积或前向激活卷积。
      ConvFwdPd fwd_pd;  // 定义一个卷积前向描述符fwd_pd
      if (bias_data != nullptr && conv_result_scale_one) {  // 检查是否有偏置数据并且卷积结果缩放为1
        
        // 创建偏置内存描述符并设置卷积前向描述符（带偏置）
        auto bias_md = dnnl::memory::desc(bias_dims, data_type,  // 创建一个偏置内存描述符bias_md，使用偏置维度、数据类型和一维格式标签x。
                                          dnnl::memory::format_tag::x);
        
        fwd_pd = ConvFwdPd(onednn_primitive->engine, dnnl::prop_kind::forward, // 使用偏置描述符bias_md，以及其他参数创建卷积前向描述符fwd_pd。
                           dnnl::algorithm::convolution_direct, src_md,
                           filter_md_prefer, bias_md, dst_md, stride_dims,
                           dilation_dims, padding_dims_l, padding_dims_r,
                           post_ops_attr);
        
        // 使用偏置内存描述符、引擎和偏置数据创建偏置内存对象，并存储在onednn_primitive->bias_memory中
        onednn_primitive->bias_memory = dnnl::sycl_interop::make_memory(
            bias_md, onednn_primitive->engine, kind, bias_data);
        
        // 将偏置内存对象添加到卷积原语的参数中，使用DNNL_ARG_BIAS表示这是偏置数据。
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_BIAS, onednn_primitive->bias_memory});
      
      // } else {：如果没有偏置数据或者卷积结果缩放系数不为1，创建不带偏置的卷积前向描述符fwd_pd。
      } else {
        // 设置卷积前向描述符（不带偏置）
        // 使用源内存描述符src_md、过滤器内存描述符filter_md_prefer、目标内存描述符dst_md以及其他参数创建卷积前向描述符fwd_pd。
        fwd_pd = ConvFwdPd(onednn_primitive->engine, dnnl::prop_kind::forward,
                           dnnl::algorithm::convolution_direct, src_md,
                           filter_md_prefer, dst_md, stride_dims, dilation_dims,
                           padding_dims_l, padding_dims_r, post_ops_attr);
              
        }

      //使用前面创建的卷积前向描述符fwd_pd创建卷积前向原语，并将其存储在onednn_primitive->fwd_primitive中。
      onednn_primitive->fwd_primitive = dnnl::convolution_forward(fwd_pd);
      
      // 获取并分配工作空间内存
      size_t scratchpad_size = fwd_pd.scratchpad_desc().get_size();  // 获取卷积前向描述符的工作空间大小。
      void* workspace; // 定义一个指向工作空间的指针workspace
      
      TF_RETURN_IF_ERROR( // 调用AllocateWorkspace函数为工作空间分配内存，并将结果存储在workspace指针中。如果分配失败，返回错误。
          AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));
      
      onednn_primitive->scratchpad_memory = dnnl::memory( // 使用工作空间描述符、引擎和分配的内存创建一个工作空间内存对象，并存储在onednn_primitive->scratchpad_memory中
          fwd_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

      // 检查过滤器是否需要重新排序， 查过滤器内存描述符filter_md是否与前向描述符的权重描述符不同。如果不同，则需要重新排序。
      bool is_filter_reordered = (filter_md != fwd_pd.weights_desc());
      
      // 处理需要重新排序的情况
      if (is_filter_reordered) {
        onednn_primitive->has_reorder = true;                                  // 设置has_reorder标志为true，表示需要重新排序
        size_t reorder_filter_data_size = fwd_pd.weights_desc().get_size();   // 获取重新排序后的过滤器数据大小。
        void* reorder_filter; //定义一个指向重新排序后的过滤器内存的指针
        TF_RETURN_IF_ERROR(AllocateWorkspace(&reorder_filter, scratch_allocator,
                                             reorder_filter_data_size)); // 为重新排序后的过滤器分配内存，并将结果存储在reorder_filter指针中。如果分配失败，返回错误。

        onednn_primitive->internal_filter_memory = dnnl::memory(
            fwd_pd.weights_desc(), onednn_primitive->engine, reorder_filter); //使用权重描述符、引擎和分配的内存创建一个内部过滤器内存对象，并存储在onednn_primitive->internal_filter_memory中。

        onednn_primitive->filter_reorder_primitive =    // 创建一个重新排序原语，将原始过滤器内存重新排序到内部过滤器内存。
            dnnl::reorder(onednn_primitive->filter_memory,
                          onednn_primitive->internal_filter_memory);

        onednn_primitive->reorder_args = {  // 设置重新排序原语的参数，源是原始过滤器内存，目标是内部过滤器内存。
            {DNNL_ARG_SRC, onednn_primitive->filter_memory},
            {DNNL_ARG_DST, onednn_primitive->internal_filter_memory}};

        onednn_primitive->fwd_primitives_args.insert(  // 将内部过滤器内存对象添加到卷积前向原语的参数中，使用DNNL_ARG_WEIGHTS表示这是权重数据。
            {DNNL_ARG_WEIGHTS, onednn_primitive->internal_filter_memory});
      
        // 处理不需要重新排序的情况,如果不需要重新排序，设置has_reorder标志为false
        } else {
          onednn_primitive->has_reorder = false;
          onednn_primitive->fwd_primitives_args.insert( // 将原始过滤器内存对象添加到卷积前向原语的参数中，使用DNNL_ARG_WEIGHTS表示这是权重数据。
              {DNNL_ARG_WEIGHTS, onednn_primitive->filter_memory});
        }

        // 添加其他参数到卷积前向原语
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_SRC, onednn_primitive->src_memory});  // ：将源内存对象添加到卷积前向原语的参数中，使用DNNL_ARG_SRC表示这是源数据。
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_DST, onednn_primitive->dst_memory}); // 将目标内存对象添加到卷积前向原语的参数中，使用DNNL_ARG_DST表示这是目标数据。
        onednn_primitive->fwd_primitives_args.insert(
            {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory}); // 将工作空间内存对象添加到卷积前向原语的参数中，使用DNNL_ARG_SCRATCHPAD表示这是工作空间数据

      
      // 处理卷积反向输入情况
      // 检查卷积类型是否为反向输入卷积（CudnnConvKind::kBackwardInput）
    } else if (conv_kind == CudnnConvKind::kBackwardInput) {
        // TODO: handle post_ops_attr.
        // 创建前向描述符
        // 使用引擎、前向卷积属性、卷积算法、源内存描述符、过滤器内存描述符、目标内存描述符、步幅、膨胀、左边和右边的填充尺寸创建卷积前向描述符fwd_pd
        ConvFwdPd fwd_pd = ConvFwdPd(
            onednn_primitive->engine, dnnl::prop_kind::forward,
            dnnl::algorithm::convolution_direct, src_md, filter_md_prefer, dst_md,
            stride_dims, dilation_dims, padding_dims_l, padding_dims_r);

        // 设置反向输入卷积属性, 创建一个dnnl::primitive_attr对象attr，并将其工作空间模式设置为用户模式
        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
        
        // 创建反向输入描述符：
        // 使用引擎、卷积算法、源内存描述符、过滤器内存描述符、目标内存描述符、步幅、膨胀、左边和右边的填充尺寸、前向描述符和属性创建卷积反向输入描述符bwd_input_pd。
        ConvBwdInputPd bwd_input_pd = ConvBwdInputPd(
            onednn_primitive->engine, dnnl::algorithm::convolution_direct, src_md,
            filter_md_prefer, dst_md, stride_dims, dilation_dims, padding_dims_l,
            padding_dims_r, fwd_pd, attr);

        // 获取并分配工作空间内存：
        size_t scratchpad_size = bwd_input_pd.scratchpad_desc().get_size(); // 获取反向输入描述符的工作空间大小
        void* workspace; // 定义一个指向工作空间的指针workspace
        TF_RETURN_IF_ERROR( // 调用AllocateWorkspace函数为工作空间分配内存，并将结果存储在workspace指针中。如果分配失败，返回错误
            AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));
        
        onednn_primitive->scratchpad_memory = dnnl::memory( // 使用工作空间描述符、引擎和分配的内存创建一个工作空间内存对象，并存储在onednn_primitive->scratchpad_memory中。
            bwd_input_pd.scratchpad_desc(), onednn_primitive->engine, workspace);
        
        // 检查过滤器是否需要重新排序
        // 检查过滤器内存描述符filter_md是否与反向输入描述符的权重描述符不同。如果不同，则需要重新排序。
        bool is_filter_reordered = (filter_md != bwd_input_pd.weights_desc());
        
        // 处理需要重新排序的情况
        if (is_filter_reordered) {
          size_t reorder_filter_data_size =
              bwd_input_pd.weights_desc().get_size(); // 获取重新排序后的过滤器数据大小。
          void* reorder_filter; // 定义一个指向重新排序后的过滤器内存的指针reorder_filter。
          
          // 为重新排序后的过滤器分配内存，并将结果存储在reorder_filter指针中。如果分配失败，返回错误。
          TF_RETURN_IF_ERROR(AllocateWorkspace(&reorder_filter, scratch_allocator,
                                              reorder_filter_data_size));
          
          // 使用权重描述符、引擎和分配的内存创建一个内部过滤器内存对象，并存储在onednn_primitive->internal_filter_memory中。
          onednn_primitive->internal_filter_memory =
              dnnl::memory(bwd_input_pd.weights_desc(), onednn_primitive->engine,
                          reorder_filter);
          
          // 创建一个重新排序原语，将原始过滤器内存重新排序到内部过滤器内存。
          onednn_primitive->filter_reorder_primitive =
              dnnl::reorder(onednn_primitive->filter_memory,
                            onednn_primitive->internal_filter_memory);
          // 设置重新排序原语的参数，源是原始过滤器内存，目标是内部过滤器内存。
          onednn_primitive->reorder_args = {
              {DNNL_ARG_SRC, onednn_primitive->filter_memory},
              {DNNL_ARG_DST, onednn_primitive->internal_filter_memory}};
          
          // 将内部过滤器内存对象添加到卷积反向输入原语的参数中，使用DNNL_ARG_WEIGHTS表示这是权重数据
          onednn_primitive->bwd_input_primitive_args.insert(
              {DNNL_ARG_WEIGHTS, onednn_primitive->internal_filter_memory});
          onednn_primitive->has_reorder = true; // 设置has_reorder标志为true，表示需要重新排序。
        
        // 处理不需要重新排序的情况
        } else {
          onednn_primitive->bwd_input_primitive_args.insert(  // 将原始过滤器内存对象添加到卷积反向输入原语的参数中，使用DNNL_ARG_WEIGHTS表示这是权重数据
              {DNNL_ARG_WEIGHTS, onednn_primitive->filter_memory});
          onednn_primitive->has_reorder = false;  // 如果不需要重新排序，设置has_reorder标志为false
        }

        // 添加其他参数到卷积反向输入原语
        onednn_primitive->bwd_input_primitive_args.insert(
            {DNNL_ARG_DIFF_DST, onednn_primitive->dst_memory});  // 将目标内存对象添加到卷积反向输入原语的参数中，使用DNNL_ARG_DIFF_DST表示这是目标数据。
        
        onednn_primitive->bwd_input_primitive_args.insert(
            {DNNL_ARG_DIFF_SRC, onednn_primitive->src_memory}); // 将源内存对象添加到卷积反向输入原语的参数中，使用DNNL_ARG_DIFF_SRC表示这是源数据。
        
        onednn_primitive->bwd_input_primitive_args.insert(
            {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory}); //将工作空间内存对象添加到卷积反向输入原语的参数中，使用DNNL_ARG_SCRATCHPAD表示这是工作空间数据

        onednn_primitive->bwd_input_primitive =
            dnnl::convolution_backward_data(bwd_input_pd);

      // 如果卷积类型是kBackwardFilter，即反向 卷积核 卷积
    } else if (conv_kind == CudnnConvKind::kBackwardFilter) {
        // TODO: handle post_ops_attr.
        ConvFwdPd fwd_pd = ConvFwdPd(
            onednn_primitive->engine, dnnl::prop_kind::forward,
            dnnl::algorithm::convolution_direct, src_md, filter_md_prefer, dst_md,
            stride_dims, dilation_dims, padding_dims_l, padding_dims_r);
            /*
            定义并初始化一个前向卷积描述符fwd_pd。这里传入了多个参数，包括引擎、卷积类型（前向）、算法（直接卷积）、源内存描述符、
            首选 卷积核 内存描述符、目标内存描述符、步幅、扩展、左侧和右侧的填充尺寸
            */

        dnnl::primitive_attr attr;  // 创建一个新的dnnl::primitive_attr对象attr，用于设置原始属性。
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);  // 设置暂存区模式为用户管理
        ConvBwdFilterPd bwd_filter_pd = ConvBwdFilterPd(
            onednn_primitive->engine, dnnl::algorithm::convolution_direct, src_md,
            filter_md_prefer, dst_md, stride_dims, dilation_dims, padding_dims_l,
            padding_dims_r, fwd_pd, attr);
            /*
            定义并初始化一个反向 卷积核 卷积描述符bwd_filter_pd，传入了多个参数，包括引擎、算法（直接卷积）、源内存描述符、首选 卷积核 内存描述符、
            目标内存描述符、步幅、扩展、左侧和右侧的填充尺寸、前向卷积描述符fwd_pd、以及原始属性attr。
            */

        size_t scratchpad_size = bwd_filter_pd.scratchpad_desc().get_size();  // 获取反向 卷积核 卷积所需的暂存区大小。
        void* workspace;  //定义一个指向工作区的指针workspace。

        // 调用AllocateWorkspace函数为工作区分配内存。如果分配失败，返回错误。
        TF_RETURN_IF_ERROR(AllocateWorkspace(&workspace, scratch_allocator, scratchpad_size));
        
        // 使用分配的工作区初始化onednn_primitive的暂存区内存。
        onednn_primitive->scratchpad_memory = dnnl::memory(
            bwd_filter_pd.scratchpad_desc(), onednn_primitive->engine, workspace);

        // 检查 卷积核 是否需要重新排序，通过比较 卷积核 内存描述符和反向 卷积核 卷积的权重差异描述符。
        bool is_filter_reordered =
            (filter_md != bwd_filter_pd.diff_weights_desc());
        // 如果 卷积核 需要重新排序
        if (is_filter_reordered) {
          onednn_primitive->has_reorder = true;  // 设置onednn_primitive的has_reorder属性为真，表示需要重新排序。
          size_t reorder_filter_data_size = bwd_filter_pd.diff_weights_desc().get_size(); //  获取重新排序后的 卷积核 数据大小。
          
          void* prefer_filter;  // 定义一个指向首选 卷积核 内存的指针prefer_filter
          TF_RETURN_IF_ERROR(AllocateWorkspace(&prefer_filter, scratch_allocator,
                                              reorder_filter_data_size));
          // 调用AllocateWorkspace函数为重新排序后的 卷积核 内存分配工作区。如果分配失败，返回错误。

          onednn_primitive->internal_filter_memory =  // 这是一个指向onednn_primitive对象的内部 卷积核 内存的指针
              dnnl::memory(bwd_filter_pd.diff_weights_desc(), 
                          onednn_primitive->engine, prefer_filter); 
              /*
              使用反向 卷积核 卷积描述符的权重差异描述符bwd_filter_pd.diff_weights_desc()、onednn_primitive的引擎和之前分配的prefer_filter内存，
              创建一个新的dnnl::memory对象，并将其分配给onednn_primitive->internal_filter_memory。
              */

          onednn_primitive->filter_reorder_primitive = // 这是一个指向onednn_primitive对象的 卷积核 重排序原语的指针
              dnnl::reorder(onednn_primitive->internal_filter_memory,
                            onednn_primitive->filter_memory);
              /*
              创建一个从internal_filter_memory到filter_memory的重排序原语，并将其分配给onednn_primitive->filter_reorder_primitive
              */              

          onednn_primitive->reorder_args = { // 这是一个包含重排序原语参数的容器
              {DNNL_ARG_SRC, onednn_primitive->internal_filter_memory}, // 表示源数据为internal_filter_memory
              {DNNL_ARG_DST, onednn_primitive->filter_memory}}; // 表示目标数据为filter_memory， 这两个参数被存储在reorder_args中，以便在执行重排序原语时使用

          onednn_primitive->bwd_filter_primitive_args.insert(  // 这是一个包含反向 卷积核 卷积原语参数的容器。
              {DNNL_ARG_DIFF_WEIGHTS, onednn_primitive->internal_filter_memory});
              // 将权重差异参数DNNL_ARG_DIFF_WEIGHTS和对应的内存对象internal_filter_memory插入到bwd_filter_primitive_args中。
              // 这样做是为了在执行反向 卷积核 卷积时使用重新排序后的 卷积核 内存
              //这段代码实现了在需要重新排序 卷积核 内存时的相关步骤，包括初始化内存对象、创建重排序原语、设置重排序和反向 卷积核 卷积的参数
        

        // 判断is_filter_reordered是否为真的分支，如果 卷积核 不需要重新排序，则执行如下代码
        } else {
          onednn_primitive->has_reorder = false;   // 设置onednn_primitive的has_reorder属性为假，表示不需要重排序
          
          onednn_primitive->bwd_filter_primitive_args.insert(
              {DNNL_ARG_DIFF_WEIGHTS, onednn_primitive->filter_memory});
              // 将权重差异参数DNNL_ARG_DIFF_WEIGHTS和对应的内存对象filter_memory插入到bwd_filter_primitive_args中。
        }

        // 将源数据参数DNNL_ARG_SRC和对应的内存对象src_memory插入到bwd_filter_primitive_args中。
        onednn_primitive->bwd_filter_primitive_args.insert(
            {DNNL_ARG_SRC, onednn_primitive->src_memory});
        
        // 将目标差异数据参数DNNL_ARG_DIFF_DST和对应的内存对象dst_memory插入到bwd_filter_primitive_args中
        onednn_primitive->bwd_filter_primitive_args.insert(
            {DNNL_ARG_DIFF_DST, onednn_primitive->dst_memory});

        // 将暂存区参数DNNL_ARG_SCRATCHPAD和对应的内存对象scratchpad_memory插入到bwd_filter_primitive_args中
        onednn_primitive->bwd_filter_primitive_args.insert(
            {DNNL_ARG_SCRATCHPAD, onednn_primitive->scratchpad_memory});

        // 
        onednn_primitive->bwd_filter_primitive =  // 这是一个指向onednn_primitive对象的反向 卷积核 卷积原语的指针
            ConvBwdFilterPrimitive(bwd_filter_pd); // 使用之前定义的反向 卷积核 卷积描述符bwd_filter_pd创建一个新的反向 卷积核 卷积原语，并将其分配给onednn_primitive->bwd_filter_primitive。

    } else {  // 如果无法处理，然会 未知的 conv 类型
      return Internal("Unkown convolutuion kind");
    }
      
    } catch (dnnl::error& e) {
      /*
      } catch (dnnl::error& e) {：捕捉在try块中可能抛出的dnnl::error异常。这表示如果在执行上述代码块时发生了OneDNN库相关的错误，
      程序会跳转到这里执行异常处理代码

      返回一个包含错误信息的Internal状态。e.message是异常对象中的错误消息字符串。这个错误信息详细描述了OneDNN卷积操作过程中发生的错误。
      Internal：可能是一个用来创建内部错误状态的函数。
      "OneDNN Conv error: %s"：错误消息的格式字符串，其中%s将被e.message的内容替换。
      
      */

    // 如果代码执行没有抛出异常，则返回一个表示成功状态的OkStatus。absl::OkStatus()表示操作成功并且没有错误
      return Internal("OneDNN Conv error: %s", e.message);
    }

    return absl::OkStatus();
  }  // NOLINT
}  // namespace



/*
这个函数GetOrCreateOneDnnConvPrimitive用于获取或创建一个OneDNN卷积原语。它接收计算流、配置字典、后端字典、
操作数缓冲区、结果缓冲区、暂存区分配器和卷积类型作为参数。
首先，它尝试创建一个OneDNN卷积原语并检查是否成功。如果成功，返回创建的原语对象，否则返回错误状态。
*/

// 获取或创建 OneDnn 卷积原语函数
absl::StatusOr<OneDnnConvPrimitive> GetOrCreateOneDnnConvPrimitive(
    se::Stream* stream, const ffi::Dictionary& dict,
    absl::flat_hash_map<std::string, std::string>& backend_dict,
    const std::vector<ffi::BufferBase>& operand_se_buffers,
    const ffi::BufferBase& result_buffer,
    se::ScratchAllocator* scratch_allocator, CudnnConvKind conv_kind) {
      /*
      absl::StatusOr<OneDnnConvPrimitive>：这是一个返回类型，表示返回一个包含OneDnnConvPrimitive对象或一个错误状态的对象。StatusOr类型用于函数可能返回有效结果或错误的情况。
      GetOrCreateOneDnnConvPrimitive(...)：这是函数的声明，函数名为GetOrCreateOneDnnConvPrimitive。
      se::Stream* stream：一个指向stream对象的指针，表示计算流。
      const ffi::Dictionary& dict：一个常量引用，指向包含一些参数或配置信息的字典。
      absl::flat_hash_map<std::string, std::string>& backend_dict：一个引用，指向包含后端配置信息的哈希映射。
      const std::vector<ffi::BufferBase>& operand_se_buffers：一个常量引用，指向操作数缓冲区的向量。
      const ffi::BufferBase& result_buffer：一个常量引用，指向结果缓冲区。
      se::ScratchAllocator* scratch_allocator：一个指向ScratchAllocator对象的指针，用于暂存区分配。
      CudnnConvKind conv_kind：卷积类型。
      
      */
  OneDnnConvPrimitive primitive; // 声明并定义一个OneDnnConvPrimitive对象primitive，用于存储创建的OneDNN卷积原语。

  auto status = CreateOneDnnPrimitive(&primitive, dict, backend_dict,
                                      absl::MakeSpan(operand_se_buffers),
                                      result_buffer, stream, scratch_allocator,
                                      conv_kind);
      /*
      auto status：使用自动类型推导声明变量status，存储创建OneDNN卷积原语的结果状态。
      CreateOneDnnPrimitive(...)：调用函数CreateOneDnnPrimitive，传入primitive的指针、字典dict、后端字典backend_dict、操作数缓冲区、
      结果缓冲区、计算流、暂存区分配器和卷积类型conv_kind。
      */                        
  if (TF_PREDICT_FALSE(!status.ok())) {  // 检查status是否表示成功状态。TF_PREDICT_FALSE是一个宏，提示预测分支大多数情况下为假，提高性能。
    return status;  // ：如果status不表示成功状态，返回错误状态。
  }
  return primitive; //如果创建OneDNN卷积原语成功，返回创建的OneDnnConvPrimitive对象。
}


// 入口函数
// 这段代码定义了一个名为 RunGpuConv 的函数，用于在 GPU 上运行卷积操作
absl::Status RunGpuConv(const OneDnnConvPrimitive& onednn_primitive,
                        const ffi::Dictionary& dict,
                        absl::Span<const ffi::BufferBase> operand_buffers,
                        ffi::BufferBase result_buffer, CudnnConvKind conv_kind) {
  /*
  absl::Status：函数返回一个 absl::Status 类型，用于表示操作的成功或失败。
  const OneDnnConvPrimitive& onednn_primitive：函数参数 onednn_primitive 是一个常量引用，表示 OneDNN 卷积原语。
  const ffi::Dictionary& dict：函数参数 dict 是一个常量引用，表示一个字典，包含额外的配置或参数。
  absl::Span<const ffi::BufferBase> operand_buffers：函数参数 operand_buffers 是一个 span，表示一组输入缓冲区。
  ffi::BufferBase result_buffer：函数参数 result_buffer 表示输出缓冲区。
  CudnnConvKind conv_kind：函数参数 conv_kind 表示卷积操作的类型。
  */
  
  // 声明了几个指向数据的指针，用于存储输入数据、 卷积核 数据、输出数据、偏置数据和侧输入数据。其中，bias_data 和 side_input_data 初始化为 nullptr
  void* input_data;  
  void* filter_data;
  void* output_data;
  void* bias_data = nullptr;
  void* side_input_data = nullptr;


  switch (conv_kind) {  // switch 语句根据 conv_kind 的值选择不同的处理方式
    // 处理前向卷积
    case CudnnConvKind::kForward:
    
    // 处理 前向激活卷积
    case CudnnConvKind::kForwardActivation:
      // 对于前向卷积和前向激活卷积，input_data 指向第一个输入缓冲区的数据，filter_data 指向第二个输入缓冲区的数据，
      // output_data 指向输出缓冲区的数据。const_cast<void*> 用于移除常量性，因为这些数据最初是 const 的
      input_data = const_cast<void*>(operand_buffers[0].data.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
      output_data = const_cast<void*>(result_buffer.data.opaque());
      break;
    
    // 处理反向输入卷积
    case CudnnConvKind::kBackwardInput:
    // 对于反向输入卷积，input_data 指向输出缓冲区的数据，filter_data 指向第二个输入缓冲区的数据，output_data 指向第一个输入缓冲区的数据
      input_data = const_cast<void*>(result_buffer.data.opaque());
      filter_data = const_cast<void*>(operand_buffers[1].data.opaque());
      output_data = const_cast<void*>(operand_buffers[0].data.opaque());
      break;
    
    // 处理反向 卷积核 卷积。
    case CudnnConvKind::kBackwardFilter:
    // 对于反向 卷积核 卷积，input_data 指向第一个输入缓冲区的数据，filter_data 指向输出缓冲区的数据，output_data 指向第二个输入缓冲区的数据。
      input_data = const_cast<void*>(operand_buffers[0].data.opaque());
      filter_data = const_cast<void*>(result_buffer.data.opaque());
      output_data = const_cast<void*>(operand_buffers[1].data.opaque());
      break;
    
    default:  // default 分支处理未知的卷积类型，返回一个内部错误状态，表示卷积类型未知。
      return Internal("Unkown convolution kind");
  }

  // 整个 switch 语句完成后，根据 conv_kind 的值，指针 input_data、filter_data 和 output_data 分别指向相应的数据。
  //接下来可以加入实际的卷积操作逻辑，例如调用 cuDNN 函数来执行卷积操作，并处理任何可能出现的错误。



  /*
  如果 conv_kind 是 CudnnConvKind::kForwardActivation（表示前向激活卷积），则需要使用偏置数据和可能的侧输入数据。
  bias_data 指向第三个输入缓冲区的数据
  如果输入缓冲区数量大于等于 4，则 side_input_data 指向第四个输入缓冲区的数据。
  */
  if (conv_kind == CudnnConvKind::kForwardActivation) {
    bias_data = const_cast<void*>(operand_buffers[2].data.opaque());
    if (operand_buffers.size() >= 4) {
      side_input_data = const_cast<void*>(operand_buffers[3].data.opaque());
    }
  }

  // 这三行 设置 OneDNN 卷积原语的源、 卷积核 和目标内存的数据句柄
  onednn_primitive.src_memory.set_data_handle(input_data);
  onednn_primitive.filter_memory.set_data_handle(filter_data);
  onednn_primitive.dst_memory.set_data_handle(output_data);
  
  // 如果 bias_data 不为空，则设置偏置内存的数据句柄。
  if (bias_data != nullptr) {
    onednn_primitive.bias_memory.set_data_handle(bias_data);
  }

  // 下面的代码开始 尝试执行卷积操作，根据 conv_kind 的值选择相应的执行路径
  try {
    if (conv_kind == CudnnConvKind::kForward ||
        conv_kind == CudnnConvKind::kForwardActivation) {
      /*
      如果 conv_kind 是 CudnnConvKind::kForward 或 CudnnConvKind::kForwardActivation：
        如果需要重新排序 卷积核 ，则执行重新排序操作。
        执行前向卷积原语。
      
      */
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
      onednn_primitive.fwd_primitive.execute(
          onednn_primitive.stream, onednn_primitive.fwd_primitives_args);

    } else if (conv_kind == CudnnConvKind::kBackwardInput) {
      /*
      如果 conv_kind 是 CudnnConvKind::kBackwardInput：
      如果需要重新排序 卷积核 ，则执行重新排序操作。
      执行反向输入卷积原语。
      */
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }
      onednn_primitive.bwd_input_primitive.execute(
          onednn_primitive.stream, onednn_primitive.bwd_input_primitive_args);
    
    } else if (conv_kind == CudnnConvKind::kBackwardFilter) {
      /*
      如果 conv_kind 是 CudnnConvKind::kBackwardFilter：
      执行反向 卷积核 卷积原语。
      如果需要重新排序 卷积核 ，则执行重新排序操作。
      
      */
      onednn_primitive.bwd_filter_primitive.execute(
          onednn_primitive.stream, onednn_primitive.bwd_filter_primitive_args);
      if (onednn_primitive.has_reorder) {
        onednn_primitive.filter_reorder_primitive.execute(
            onednn_primitive.stream, onednn_primitive.reorder_args);
      }

    } else {
  // 如果 conv_kind 的值未知，则返回一个内部错误状态
      return Internal("Unkown convolutuion kind");
    }

  // 捕获任何来自 OneDNN 的错误，并输出错误信
  } catch (dnnl::error& e) {
    std::string error_msg = "Status: " + std::to_string(e.status) +
                            ", message: " + std::string(e.message) +
                            ", in file " + std::string(__FILE__) + ":" +
                            std::to_string(__LINE__);
    std::cout << error_msg << std::endl;
  }
  // 如果没有发生错误，返回成功状态 absl::OkStatus()。
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
