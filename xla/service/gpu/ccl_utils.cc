/* Copyright (c) 2023 Intel Corporation

Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/ccl_utils.h"

#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tsl/platform/env.h"
#include "xla/debug_options_flags.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/rendezvous.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

bool IsGlobalCclConfig() {
  static const bool global_nccl_config = std::getenv("NCCL_COMM_ID") != nullptr;
  return global_nccl_config;
}

bool IsCclLaunchModeParallel() {
  static const bool is_launch_mode_parallel = []() {
    const char* launch_mode = std::getenv("NCCL_LAUNCH_MODE");
    return launch_mode && std::string_view(launch_mode) == "PARALLEL";
  }();
  return is_launch_mode_parallel;
}

#if ITEX_USE_CCL
ccl::reduction ToCclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ccl::reduction::sum;
    case ReductionKind::PRODUCT:
      return ccl::reduction::prod;
    case ReductionKind::MIN:
      return ccl::reduction::min;
    case ReductionKind::MAX:
      return ccl::reduction::max;
  }
}
#endif  // ITEX_USE_CCL

namespace {
StatusOr<std::string> ToNcclUniqueId(const std::string& id_str) {
  return id_str;
}

StatusOr<std::string> LocalNcclUniqueIdCallback(const NcclCliqueKey&) {
  return std::string("");
}

struct CclCliqueState {
  std::string unique_id;
  int64_t run_id = -1;

  // `mu` guards `communicators` and `status` during initialization.
  // Once `ready` has been notified, the communicators may be accessed without
  // synchronization.
  absl::Mutex mu;
  absl::Notification ready;
  Status status;
  absl::flat_hash_map<int, std::unique_ptr<CclComm>> communicators;
};

using CclClique = Lockable<CclCliqueState>;

std::shared_ptr<StatusOr<CclClique::Lock>> AcquireCclClique(
    RunId run_id, OpId op_id, NcclCliqueKey clique_key,
    const NcclUniqueIdCallback& unique_id_callback,
    size_t num_local_participants) {
  static auto& cliques = *new ThreadSafeMap<NcclCliqueKey, CclClique>;

  auto rendezvous_key = std::make_tuple(run_id, op_id, std::move(clique_key));

  int64_t terminate_timeout = xla::GetDebugOptionsFromFlags()
                                  .xla_gpu_nccl_termination_timeout_seconds();

  return RendezvousSingle<StatusOr<CclClique::Lock>>(
      rendezvous_key, num_local_participants,
      [&]() -> StatusOr<CclClique::Lock> {
        const NcclCliqueKey& clique_key = std::get<2>(rendezvous_key);
        CclClique::Lock clique = cliques[clique_key].Acquire();
        if (clique->run_id < 0) {
          // TF_ASSIGN_OR_RETURN(std::string id,
          // unique_id_callback(clique_key));
          std::string id = run_id.ToString();
          TF_ASSIGN_OR_RETURN(clique->unique_id, ToNcclUniqueId(id));
        }
        // If multiple executable are running simultaneously while using
        // multiple hosts, it is possible that different executables could
        // acquire the same clique on different hosts. We protect against this
        // by checking that the run ID increases monotonically.
        bool is_local = clique_key.devices().size() == num_local_participants;
        TF_RET_CHECK(is_local || (run_id.ToInt() >= clique->run_id));
        clique->run_id = run_id.ToInt();
        return clique;
      },
      /*warn_stuck_timeout=*/absl::Seconds(10),
      (terminate_timeout >= 0) ? absl::Seconds(terminate_timeout)
                               : absl::InfiniteDuration());
}
#if 0
void CheckNcclAsyncError(CclComm& lockable_comm) {
  ncclComm_t comm = *lockable_comm.Acquire();
  if (comm == nullptr) return;

  Status status = [comm] {
    ncclResult_t async_err;
    XLA_CUDA_RETURN_IF_ERROR(ncclCommGetAsyncError(comm, &async_err));
    if (async_err != ncclSuccess) {
      LOG(ERROR) << "Aborting communicator: " << comm
                 << " due to async NCCL error: "
                 << ncclGetErrorString(async_err);
      XLA_CUDA_RETURN_IF_ERROR(ncclCommAbort(comm));
    }
    return XLA_CUDA_STATUS(async_err);
  }();

  if (!status.ok()) LOG(ERROR) << status.ToString();
}
#endif
}  // namespace
#if ITEX_USE_CCL
StatusOr<std::pair<ncclDataType_t, int>> ToCclDataTypeAndCountMultiplier(
    PrimitiveType element_type, Thunk::Kind reduction_op) {
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype,
                      ToNcclDataType(element_type, reduction_op));
  bool is_complex = primitive_util::IsComplexType(element_type);
  return std::make_pair(dtype, is_complex ? 2 : 1);
}
#endif  // ITEX_USE_CCL

size_t GetNumLocalParticipants(
    const std::vector<GlobalDeviceId>& participants,
    const std::vector<GlobalDeviceId>* local_devices) {
  if (local_devices == nullptr) return participants.size();

  return absl::c_count_if(participants, [&](const GlobalDeviceId& device_id) {
    return absl::c_linear_search(*local_devices, device_id);
  });
}

StatusOr<const NcclUniqueIdCallback*> GetNcclUniqueIdCallback(
    const NcclUniqueIdCallback* unique_id_callback, bool is_local) {
  if (unique_id_callback != nullptr) return unique_id_callback;

  TF_RET_CHECK(is_local || IsGlobalCclConfig())
      << "If non-local devices are taking part of a collective API on "
         "GPU, the nccl_unique_id_callback must be provided by the client.";

  static auto* local_callback =
      new NcclUniqueIdCallback(LocalNcclUniqueIdCallback);
  return local_callback;
}

StatusOr<CclComm::Lock> AcquireCclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    size_t num_local_participants,
    const NcclUniqueIdCallback& unique_id_callback, int rank,
    int64_t stream_id) {
  // Ensure that this group of threads have exclusive access to the clique to
  // prevent threads from different groups locking communicators in the clique.
  NcclCliqueKey clique_key(std::move(participants), stream_id);
  std::shared_ptr<StatusOr<CclClique::Lock>> clique = AcquireCclClique(
      run_id, op_id, clique_key, unique_id_callback, num_local_participants);

  if (!clique->ok()) return clique->status();

  struct AllCommunicators {
    absl::Mutex mu;
    std::vector<CclComm*> communicators ABSL_GUARDED_BY(mu);
  };
  static auto& all_communicators = *new AllCommunicators;

  CclCliqueState& state = ***clique;
  if (!state.ready.HasBeenNotified()) {
    int nranks = clique_key.devices().size();
    const std::string& id = state.unique_id;

    ccl::communicator* comm =
        new ccl::communicator(nranks, rank, state.unique_id);
    // Status status = XLA_CUDA_STATUS(ncclCommInitRank(&comm, nranks, id,
    // rank));
    Status status = tsl::OkStatus();
    size_t num_initialized = [&] {
      absl::MutexLock lock(&state.mu);
      state.status.Update(status);
      state.communicators[rank] = std::make_unique<CclComm>(comm);
      return state.communicators.size();
    }();

    // Wait for all communicators to initialize before allowing any progress.
    // Otherwise we may get deadlocks, because ncclCommInitRank may allocate,
    // which may block on the completion of device activity on a peer device,
    // which may depend on the completion of this collective if we do not have a
    // barrier to prevent it.
    if (num_initialized == num_local_participants) {
      state.ready.Notify();
    } else {
      TF_RETURN_IF_ERROR(status);
      state.ready.WaitForNotification();
    }

    absl::MutexLock lock(&all_communicators.mu);
    all_communicators.communicators.push_back(state.communicators[rank].get());
  }

  TF_RETURN_IF_ERROR(state.status);
  return state.communicators[rank]->Acquire();
}
}  // namespace gpu
}  // namespace xla
