/* Copyright (c) 2023 Intel Corporation

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

#pragma once

#include <sycl/sycl.hpp>

namespace gpu::xetla {

void fmha_forward_kernel_fp16(sycl::queue& q, void* query, void* key,
                              void* value, void* bias, uint8_t* dropout,
                              float dropout_prob, void* out,
                              void* activation_ptr, uint32_t num_batches,
                              uint32_t num_heads, uint32_t head_size,
                              uint32_t num_queries, uint32_t num_keys,
                              float head_scale, bool is_training);

void fmha_forward_kernel_bf16(sycl::queue& q, void* query, void* key,
                              void* value, void* bias, uint8_t* dropout,
                              float dropout_prob, void* out,
                              void* activation_ptr, uint32_t num_batches,
                              uint32_t num_heads, uint32_t head_size,
                              uint32_t num_queries, uint32_t num_keys,
                              float head_scale, bool is_training);

}  // namespace gpu::xetla