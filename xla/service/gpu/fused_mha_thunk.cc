/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/fused_mha_thunk.h"

#include <memory>
#include <string>

#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/service/gpu/gpu_conv_runner.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

FusedMHAThunk::FusedMHAThunk(ThunkInfo thunk_info, GpufMHAConfig config,
                             BufferAllocation::Slice lhs_bmm1,
                             BufferAllocation::Slice rhs_bmm1,
                             BufferAllocation::Slice rhs_bmm2,
                             BufferAllocation::Slice output,
                             BufferAllocation::Slice scratch,
                             BufferAllocation::Slice mask,
                             BufferAllocation::Slice bias)
    : Thunk(Kind::kFusedMHA, thunk_info),
      lhs_bmm1_buffer_(lhs_bmm1),
      rhs_bmm1_buffer_(rhs_bmm1),
      rhs_bmm2_buffer_(rhs_bmm2),
      output_buffer_(output),
      scratch_buffer_(scratch),
      mask_buffer_(mask),
      bias_buffer_(bias),
      config_(std::move(config)) {}

Status FusedMHAThunk::ExecuteOnStream(const ExecuteParams& params) {
  const auto& buffer_allocations = *params.buffer_allocations;
  se::DeviceMemoryBase lhs_bmm1_buffer =
      buffer_allocations.GetDeviceAddress(lhs_bmm1_buffer_);
  se::DeviceMemoryBase rhs_bmm1_buffer =
      buffer_allocations.GetDeviceAddress(rhs_bmm1_buffer_);
  se::DeviceMemoryBase rhs_bmm2_buffer =
      buffer_allocations.GetDeviceAddress(rhs_bmm2_buffer_);
  se::DeviceMemoryBase output_buffer =
      buffer_allocations.GetDeviceAddress(output_buffer_);
  se::DeviceMemoryBase scratch_buffer =
      buffer_allocations.GetDeviceAddress(scratch_buffer_);

  se::DeviceMemoryBase mask_buffer;
  if (mask_buffer_.allocation() != nullptr) {
    mask_buffer = buffer_allocations.GetDeviceAddress(mask_buffer_);
  }
  se::DeviceMemoryBase bias_buffer;
  if (bias_buffer_.allocation() != nullptr) {
    bias_buffer = buffer_allocations.GetDeviceAddress(bias_buffer_);
  }

  TF_RETURN_IF_ERROR(RunGpuFMHA(config_, lhs_bmm1_buffer, rhs_bmm1_buffer,
                                rhs_bmm2_buffer, output_buffer, scratch_buffer,
                                mask_buffer, bias_buffer, params.stream));
  if (!params.stream->ok()) {
    return InternalError("FusedMHAThunk::ExecuteOnStream failed.");
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla