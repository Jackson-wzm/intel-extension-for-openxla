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
#include <xetla.hpp>

#include "epilogue_impl.h"

using namespace sycl;
using namespace gpu::xetla;
using namespace gpu::xetla::group;
using namespace gpu::xetla::kernel;
using namespace gpu::xetla::subgroup;

// Command group function implementation
#define DPCPP_Q_CGF(h) [&](sycl::handler & h)

#define DPCPP_Q_SUBMIT(q, cgf, ...) \
  { auto e = (q).submit((cgf), ##__VA_ARGS__); }

namespace gpu {
namespace xetla {

template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_ADDMM_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_COMMON_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_BIAS_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_BIAS_GELU_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_BIAS_RES_RES_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_BIAS_RES_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_QKV_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_QKV_BIAS_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_MUL_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_SILU_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_RES_KERNEL;
template <typename scalar_t, int WG_M = 8, int WG_N = 32, int SG_M = 8,
          int SG_N = 16, int SG_K = 64, int SLM_KS = 8, int L3_KS = 1,
          int SYNC_FREQ = 1, int STAGES = 3, bool B_ROW_MAJOR = false>
class HGEMM_RES_RES_KERNEL;

#define HGEMM_DEFINITIONS                                                \
  static_assert(L3_KS == 1, "currently, L3_KS should be 1");             \
  constexpr mem_layout layout_a = mem_layout::row_major;                 \
  constexpr mem_layout layout_b =                                        \
      B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;       \
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;                        \
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;                        \
  uint32_t thread_range_m = WG_M / SG_M;                                 \
  uint32_t thread_range_n = WG_N / SG_N;                                 \
  uint32_t lda = k;                                                      \
  uint32_t ldb = B_ROW_MAJOR ? n : k;                                    \
  uint32_t ldc = n;                                                      \
  cl::sycl::range<3> GroupRange{L3_KS, group_range_m, group_range_n};    \
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n}; \
  cl::sycl::nd_range<3> NDRange(GroupRange* LocalRange, LocalRange);

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_addmm(sycl::queue& queue, scalar_t* out, const scalar_t* res,
                        const scalar_t* a, const scalar_t* b, const int m,
                        const int n, const int k, const float alpha,
                        const float beta) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<HGEMM_ADDMM_KERNEL<
        scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, L3_KS, SYNC_FREQ,
        STAGES, B_ROW_MAJOR>>(NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
      xetla_exec_item<3> ei(item);
      using data_type_b = scalar_t;
      using data_type_a = scalar_t;
      using data_type_c = scalar_t;
      using data_type_acc = float;
      static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
      static constexpr uint32_t prefetch_distance = STAGES;
      using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
      using brgemm_t = typename brgemm_selector_t<
          data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
          mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
          mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
          periodic_sync_interval>::brgemm;
      using epilogue_t = epilogue_t<
          epilogue_policy_tile_op<
              chained_tile_op_t<epilogue_impl::alpha_beta_op_t<data_type_c>>,
              gpu_arch::Xe>,
          tile_shape,
          mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
      using gemm_op_t =
          gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                 brgemm_t, epilogue_t>;
      typename gemm_op_t::arguments_t arg(
          m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b), ldb,
          out, ldc, {{{const_cast<scalar_t*>(res), {n, m, n}, alpha, beta}}});
      slm_barrier_init<gemm_op_t>();
      gemm_op_t gemm_op;
      gemm_op(ei, arg);
    });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_common(sycl::queue& queue, scalar_t* out, const scalar_t* a,
                         const scalar_t* b, const int m, const int n,
                         const int k) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_COMMON_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                            L3_KS, SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<chained_tile_op_t<>, gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(m, k, n, const_cast<scalar_t*>(a),
                                              lda, const_cast<scalar_t*>(b),
                                              ldb, out, ldc);
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_res(sycl::queue& queue, scalar_t* out, const scalar_t* a,
                      const scalar_t* b, const scalar_t* res, const int m,
                      const int n, const int k, const float res_factor) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_RES_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, L3_KS,
                         SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);

          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_res = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::res_op_t<data_type_res>>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b),
              ldb, out, ldc,
              {{{const_cast<scalar_t*>(res), {n, m, n}, res_factor}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_res_res(sycl::queue& queue, scalar_t* out, const scalar_t* a,
                          const scalar_t* b, const scalar_t* res0,
                          const scalar_t* res1, const int m, const int n,
                          const int k, const float res0_factor,
                          const float res1_factor) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_RES_RES_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                             L3_KS, SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_res = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::res_op_t<data_type_res>,
                                    epilogue_impl::res_op_t<data_type_res>>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b),
              ldb, out, ldc,
              {{{const_cast<scalar_t*>(res0), {n, m, n}, res0_factor},
                {const_cast<scalar_t*>(res1), {n, m, n}, res1_factor}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_bias(sycl::queue& queue, scalar_t* out, const scalar_t* a,
                       const scalar_t* b, const scalar_t* bias, const int m,
                       const int n, const int k, const float bias_factor) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_BIAS_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, L3_KS,
                          SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_bias = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::bias_op_t<data_type_bias>>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b),
              ldb, out, ldc,
              {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_bias_res(sycl::queue& queue, scalar_t* out, const scalar_t* a,
                           const scalar_t* b, const scalar_t* bias,
                           const scalar_t* res, const int m, const int n,
                           const int k, const float bias_factor,
                           const float res_factor) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_BIAS_RES_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                              L3_KS, SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_bias = scalar_t;
          using data_type_res = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::bias_op_t<data_type_bias>,
                                    epilogue_impl::res_op_t<data_type_res>>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b),
              ldb, out, ldc,
              {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor},
                {const_cast<scalar_t*>(res), {n, m, n}, res_factor}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_bias_res_res(sycl::queue& queue, scalar_t* out,
                               const scalar_t* a, const scalar_t* b,
                               const scalar_t* bias, const scalar_t* res0,
                               const scalar_t* res1, const int m, const int n,
                               const int k, const float bias_factor,
                               const float res0_factor,
                               const float res1_factor) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<HGEMM_BIAS_RES_RES_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N,
                                               SG_K, SLM_KS, L3_KS, SYNC_FREQ,
                                               STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_bias = scalar_t;
          using data_type_res = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::bias_op_t<data_type_bias>,
                                    epilogue_impl::res_op_t<data_type_res>,
                                    epilogue_impl::res_op_t<data_type_res>>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b),
              ldb, out, ldc,
              {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor},
                {const_cast<scalar_t*>(res0), {n, m, n}, res0_factor},
                {const_cast<scalar_t*>(res1), {n, m, n}, res1_factor}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_bias_gelu(sycl::queue& queue, scalar_t* out,
                            const scalar_t* a, const scalar_t* b,
                            const scalar_t* bias, const int m, const int n,
                            const int k, const float bias_factor) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_BIAS_GELU_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                               L3_KS, SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_bias = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::bias_op_t<data_type_bias>,
                                    gelu_fwd_op_t>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b),
              ldb, out, ldc,
              {{{const_cast<scalar_t*>(bias), {n, 1, n}, bias_factor}, {}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_mul(sycl::queue& queue, scalar_t* out, const scalar_t* a,
                      const scalar_t* b, const scalar_t* mul, const int m,
                      const int n, const int k) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_MUL_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, L3_KS,
                         SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_mul = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<elemwise_reduce_op_t<
                      reduce_op::prod, data_type_mul, gpu_arch::Xe>>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda, const_cast<scalar_t*>(b),
              ldb, out, ldc, {{{const_cast<scalar_t*>(mul), {n, m, n}}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_silu(sycl::queue& queue, scalar_t* out, const scalar_t* a,
                       const scalar_t* b, const int m, const int n,
                       const int k) {
  HGEMM_DEFINITIONS
  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_SILU_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, L3_KS,
                          SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);
          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;
          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::silu_op_t>, gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;
          typename gemm_op_t::arguments_t arg(m, k, n, const_cast<scalar_t*>(a),
                                              lda, const_cast<scalar_t*>(b),
                                              ldb, out, ldc, {{{}}});
          slm_barrier_init<gemm_op_t>();
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_qkv(sycl::queue& queue, scalar_t* out0, scalar_t* out1,
                      scalar_t* out2, const scalar_t* a, const scalar_t* b,
                      const int m, const int n, const int k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  constexpr mem_layout layout_a = mem_layout::row_major;
  constexpr mem_layout layout_b =
      B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? n : k;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_QKV_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS, L3_KS,
                         SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);

          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;

          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<chained_tile_op_t<>, gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;

          uint32_t batch_id = ei.get_group(0);
          slm_barrier_init<gemm_op_t>();
          scalar_t* out =
              (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);

          uint32_t size_b = k * n;

          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda,
              const_cast<scalar_t*>(b) + size_b * batch_id, ldb, out, ldc);
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

template <typename scalar_t, int WG_M, int WG_N, int SG_M, int SG_N, int SG_K,
          int SLM_KS, int L3_KS, int SYNC_FREQ, int STAGES, bool B_ROW_MAJOR>
inline void hgemm_qkv_bias(sycl::queue& queue, scalar_t* out0, scalar_t* out1,
                           scalar_t* out2, const scalar_t* a, const scalar_t* b,
                           const scalar_t* bias, const int m, const int n,
                           const int k) {
  static_assert(L3_KS == 1, "for qkv fusion, L3_KS should be 1");
  constexpr mem_layout layout_a = mem_layout::row_major;
  constexpr mem_layout layout_b =
      B_ROW_MAJOR ? mem_layout::row_major : mem_layout::col_major;
  uint32_t group_range_m = (m + WG_M - 1) / WG_M;
  uint32_t group_range_n = (n + WG_N - 1) / WG_N;
  uint32_t thread_range_m = WG_M / SG_M;
  uint32_t thread_range_n = WG_N / SG_N;
  uint32_t lda = k;
  uint32_t ldb = B_ROW_MAJOR ? n : k;
  uint32_t ldc = n;
  cl::sycl::range<3> GroupRange{3, group_range_m, group_range_n};
  cl::sycl::range<3> LocalRange{SLM_KS, thread_range_m, thread_range_n};
  cl::sycl::nd_range<3> NDRange(GroupRange * LocalRange, LocalRange);

  auto cgf = DPCPP_Q_CGF(cgh) {
    cgh.parallel_for<
        HGEMM_QKV_BIAS_KERNEL<scalar_t, WG_M, WG_N, SG_M, SG_N, SG_K, SLM_KS,
                              L3_KS, SYNC_FREQ, STAGES, B_ROW_MAJOR>>(
        NDRange, [=](nd_item<3> item) SYCL_ESIMD_KERNEL {
          xetla_exec_item<3> ei(item);

          using data_type_b = scalar_t;
          using data_type_a = scalar_t;
          using data_type_c = scalar_t;
          using data_type_bias = scalar_t;
          using data_type_acc = float;
          static constexpr uint32_t periodic_sync_interval = SYNC_FREQ;
          static constexpr uint32_t prefetch_distance = STAGES;
          using tile_shape = tile_shape_t<WG_N, WG_M, SG_N, SG_M>;

          using brgemm_t = typename brgemm_selector_t<
              data_type_a, data_type_b, layout_a, layout_b, mem_space::global,
              mem_space::global, 8, 8, data_type_acc, tile_shape, SG_K,
              mma_engine::xmx, gpu_arch::Xe, prefetch_distance,
              periodic_sync_interval>::brgemm;
          using epilogue_t = epilogue_t<
              epilogue_policy_tile_op<
                  chained_tile_op_t<epilogue_impl::bias_op_t<data_type_bias>>,
                  gpu_arch::Xe>,
              tile_shape,
              mem_desc_t<scalar_t, mem_layout::row_major, mem_space::global>>;
          using gemm_op_t =
              gemm_t<dispatch_policy_kslicing<L3_KS, SLM_KS, gpu_arch::Xe>,
                     brgemm_t, epilogue_t>;

          uint32_t batch_id = ei.get_group(0);
          slm_barrier_init<gemm_op_t>();
          scalar_t* out =
              (batch_id == 0) ? out0 : ((batch_id == 1) ? out1 : out2);

          uint32_t size_b = k * n;
          uint32_t size_bias = n;

          typename gemm_op_t::arguments_t arg(
              m, k, n, const_cast<scalar_t*>(a), lda,
              const_cast<scalar_t*>(b) + size_b * batch_id, ldb, out, ldc,
              {{{const_cast<scalar_t*>(bias) + size_bias * batch_id,
                 {n, 1, n},
                 {1}}}});
          gemm_op_t gemm_op;
          gemm_op(ei, arg);
        });
  };
  DPCPP_Q_SUBMIT(queue, cgf);
}

#undef HGEMM_DEFINITIONS

}  // namespace xetla
}  // namespace gpu
