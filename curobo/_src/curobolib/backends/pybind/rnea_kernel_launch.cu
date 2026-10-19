/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file rnea_kernel_launch.cu
 * @brief Host launchers for the RNEA forward and backward kernels (compiled backend).
 *
 * The compiled backend cannot instantiate a kernel per robot, so it launches
 * rnea_forward_kernel_rt / rnea_backward_kernel_rt, which take the link and
 * DOF counts as arguments, one instantiation per (threads per batch, external
 * forces) pair. The launch geometry mirrors DynamicsLaunchCfg of the cuda.core
 * backend so both backends run the kernels with the same block layout.
 */

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <tuple>

#include <torch/extension.h>

#include "common/torch_cuda_utils.h"
#include "dynamics/rnea_forward_kernel.cuh"
#include "dynamics/rnea_backward_kernel.cuh"

namespace curobo {
namespace dynamics {

namespace {

// Same limits as DynamicsLaunchCfg (cuda_core_backend/dynamics_config.py).
constexpr int kMaxBatchesPerBlock = 256;
constexpr int kMaxSharedMemoryBytes = 48 * 1024;
constexpr int kSmSharedMemoryCapacityBytes = 100 * 1024;
constexpr int kMaxThreadsPerBlock = 1024;
constexpr int kWarpSize = curobo::common::warpSize;
constexpr int kBackwardBlockSharedFloatsPerLink = 12;  // mc(4) + inertia(8)
constexpr int kFixedTransformFloats = 12;
constexpr int kMassComFloats = 4;
constexpr int kInertiaFloats = 8;
constexpr int kJointOffsetFloats = 2;

struct RneaLaunchConfig {
  int blocks_per_grid;
  int threads_per_block;
  int shared_mem_bytes;
};

struct RobotParams {
  const float *fixed_transforms;
  const float *link_masses_com;
  const float *link_inertias;
  const int8_t *joint_map_type;
  const int16_t *joint_map;
  const int16_t *link_map;
  const float *joint_offset_map;
  const float *gravity;
  const int16_t *level_starts;
  const int16_t *level_links;
};

/**
 * @brief Round the batch count per block down to a warp-aligned value that
 *        maximises resident threads per SM (DynamicsLaunchCfg._warp_align_batches).
 */
int warp_align_batches(
  const int batches_per_block,
  const int threads_per_batch,
  const int smem_block_shared,
  const int smem_per_batch)
{
  const int granularity = std::max(1, kWarpSize / threads_per_batch);
  const int candidate = (batches_per_block / granularity) * granularity;
  if (candidate < granularity) {
    return batches_per_block;
  }
  int candidate_lower = candidate - granularity;
  if (candidate_lower < granularity) {
    candidate_lower = 0;
  }
  int best_batches = candidate;
  long long best_score = 0;
  for (const int cand : {candidate, candidate_lower}) {
    if (cand < 1) {
      continue;
    }
    const int smem = smem_block_shared + cand * smem_per_batch;
    const int blocks_per_sm = std::max(1, smem > 0 ? kSmSharedMemoryCapacityBytes / smem : 1);
    const long long score = static_cast<long long>(blocks_per_sm) * cand * threads_per_batch;
    if (score > best_score) {
      best_score = score;
      best_batches = cand;
    }
  }
  return best_batches;
}

/**
 * @brief Launch geometry for one RNEA kernel (DynamicsLaunchCfg.calculate_*_config).
 *
 * Forward: num_links x 12 floats of shared memory per batch element.
 * Backward: num_links x 12 floats shared by the block plus num_links x 30 per batch element.
 */
RneaLaunchConfig calculate_launch_config(
  const int batch_size,
  const int num_links,
  const int threads_per_batch,
  const bool backward)
{
  const int floats_per_batch_link = backward ? SMEM_FLOATS_PER_LINK_BWD : SMEM_FLOATS_PER_LINK;
  const int smem_block_shared =
    backward ? num_links * kBackwardBlockSharedFloatsPerLink * static_cast<int>(sizeof(float)) : 0;
  const int smem_per_batch = num_links * floats_per_batch_link * static_cast<int>(sizeof(float));
  if (smem_block_shared + smem_per_batch > kMaxSharedMemoryBytes) {
    throw std::runtime_error(
      "RNEA: a single batch element needs " + std::to_string(smem_block_shared + smem_per_batch) +
      " bytes of shared memory, more than the limit of " + std::to_string(kMaxSharedMemoryBytes));
  }
  const int max_from_smem = (kMaxSharedMemoryBytes - smem_block_shared) / smem_per_batch;
  const int max_from_threads = kMaxThreadsPerBlock / threads_per_batch;
  // Two-argument min only: the HIP build rewrites std::min to ::min, which has no
  // initializer-list form.
  int batches_per_block =
    std::min(std::min(max_from_smem, max_from_threads), std::min(kMaxBatchesPerBlock, batch_size));
  batches_per_block = std::max(1, batches_per_block);
  batches_per_block =
    warp_align_batches(batches_per_block, threads_per_batch, smem_block_shared, smem_per_batch);

  RneaLaunchConfig config;
  config.threads_per_block = batches_per_block * threads_per_batch;
  config.blocks_per_grid = (batch_size + batches_per_block - 1) / batches_per_block;
  config.shared_mem_bytes = smem_block_shared + batches_per_block * smem_per_batch;
  return config;
}

void check_tensor(
  const torch::Tensor &t,
  const char *name,
  const c10::ScalarType dtype,
  const int64_t min_numel)
{
  curobo::common::validate_cuda_input(t, name);
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ", dtype, ", got ", t.scalar_type());
  TORCH_CHECK(t.numel() >= min_numel, name, " has ", t.numel(), " elements, needs at least ", min_numel);
}

void check_sizes(
  const int64_t batch_size,
  const int64_t num_links,
  const int64_t num_dof,
  const int64_t n_levels,
  const int64_t threads_per_batch)
{
  TORCH_CHECK(batch_size > 0, "batch_size must be positive, got ", batch_size);
  TORCH_CHECK(num_links > 0, "num_links must be positive, got ", num_links);
  TORCH_CHECK(num_dof > 0, "num_dof must be positive, got ", num_dof);
  TORCH_CHECK(n_levels > 0, "n_levels must be positive, got ", n_levels);
  TORCH_CHECK(
    threads_per_batch == 1 || threads_per_batch == 2 || threads_per_batch == 4 ||
      threads_per_batch == 8 || threads_per_batch == 16 || threads_per_batch == 32,
    "threads_per_batch must be one of 1, 2, 4, 8, 16, 32, got ", threads_per_batch);
}

RobotParams check_robot_params(
  const torch::Tensor &fixed_transforms,
  const torch::Tensor &link_masses_com,
  const torch::Tensor &link_inertias,
  const torch::Tensor &joint_map_type,
  const torch::Tensor &joint_map,
  const torch::Tensor &link_map,
  const torch::Tensor &joint_offset_map,
  const torch::Tensor &gravity,
  const torch::Tensor &level_starts,
  const torch::Tensor &level_links,
  const int64_t num_links,
  const int64_t n_levels)
{
  check_tensor(fixed_transforms, "fixed_transforms", torch::kFloat32, num_links * kFixedTransformFloats);
  check_tensor(link_masses_com, "link_masses_com", torch::kFloat32, num_links * kMassComFloats);
  check_tensor(link_inertias, "link_inertias", torch::kFloat32, num_links * kInertiaFloats);
  check_tensor(joint_map_type, "joint_map_type", torch::kInt8, num_links);
  check_tensor(joint_map, "joint_map", torch::kInt16, num_links);
  check_tensor(link_map, "link_map", torch::kInt16, num_links);
  check_tensor(joint_offset_map, "joint_offset_map", torch::kFloat32, num_links * kJointOffsetFloats);
  check_tensor(gravity, "gravity", torch::kFloat32, SPATIAL_DIM);
  check_tensor(level_starts, "level_starts", torch::kInt16, n_levels + 1);
  check_tensor(level_links, "level_links", torch::kInt16, num_links);

  RobotParams params;
  params.fixed_transforms = fixed_transforms.data_ptr<float>();
  params.link_masses_com = link_masses_com.data_ptr<float>();
  params.link_inertias = link_inertias.data_ptr<float>();
  params.joint_map_type = joint_map_type.data_ptr<int8_t>();
  params.joint_map = joint_map.data_ptr<int16_t>();
  params.link_map = link_map.data_ptr<int16_t>();
  params.joint_offset_map = joint_offset_map.data_ptr<float>();
  params.gravity = gravity.data_ptr<float>();
  params.level_starts = level_starts.data_ptr<int16_t>();
  params.level_links = level_links.data_ptr<int16_t>();
  return params;
}

template<int TPB, bool HAS_EXTERNAL_FORCES>
void run_forward(
  const RneaLaunchConfig &config,
  cudaStream_t stream,
  float *tau,
  const float *q,
  const float *qd,
  const float *qdd,
  const RobotParams &robot,
  float *forward_cache,
  const float *f_ext,
  const int batch_size,
  const int n_levels,
  const int n_links,
  const int n_dof)
{
  rnea_forward_kernel_rt<TPB, HAS_EXTERNAL_FORCES>
    <<<config.blocks_per_grid, config.threads_per_block, config.shared_mem_bytes, stream>>>(
      tau, q, qd, qdd,
      robot.fixed_transforms, robot.link_masses_com, robot.link_inertias,
      robot.joint_map_type, robot.joint_map, robot.link_map, robot.joint_offset_map,
      robot.gravity, robot.level_starts, robot.level_links,
      forward_cache, f_ext, batch_size, n_levels, n_links, n_dof);
}

template<int TPB, bool HAS_EXTERNAL_FORCES>
void run_backward(
  const RneaLaunchConfig &config,
  cudaStream_t stream,
  float *grad_q,
  float *grad_qd,
  float *grad_qdd,
  float *grad_f_ext,
  const float *grad_tau,
  const float *q,
  const float *qd,
  const RobotParams &robot,
  const float *forward_cache,
  const int batch_size,
  const int n_levels,
  const int n_links,
  const int n_dof)
{
  rnea_backward_kernel_rt<TPB, HAS_EXTERNAL_FORCES>
    <<<config.blocks_per_grid, config.threads_per_block, config.shared_mem_bytes, stream>>>(
      grad_q, grad_qd, grad_qdd, grad_f_ext, grad_tau, q, qd,
      robot.fixed_transforms, robot.link_masses_com, robot.link_inertias,
      robot.joint_map_type, robot.joint_map, robot.link_map, robot.joint_offset_map,
      robot.gravity, robot.level_starts, robot.level_links,
      forward_cache, batch_size, n_levels, n_links, n_dof);
}

// Dispatch on the thread count; HAS_EXTERNAL_FORCES is chosen by the caller.
#define CUROBO_RNEA_DISPATCH_TPB(RUN, HAS_F_EXT, TPB_VALUE, ...)   \
  case TPB_VALUE:                                                  \
    RUN<TPB_VALUE, HAS_F_EXT>(__VA_ARGS__);                        \
    break;

#define CUROBO_RNEA_DISPATCH(RUN, HAS_F_EXT, TPB, ...)                        \
  switch (TPB) {                                                              \
    CUROBO_RNEA_DISPATCH_TPB(RUN, HAS_F_EXT, 1, __VA_ARGS__)                  \
    CUROBO_RNEA_DISPATCH_TPB(RUN, HAS_F_EXT, 2, __VA_ARGS__)                  \
    CUROBO_RNEA_DISPATCH_TPB(RUN, HAS_F_EXT, 4, __VA_ARGS__)                  \
    CUROBO_RNEA_DISPATCH_TPB(RUN, HAS_F_EXT, 8, __VA_ARGS__)                  \
    CUROBO_RNEA_DISPATCH_TPB(RUN, HAS_F_EXT, 16, __VA_ARGS__)                 \
    CUROBO_RNEA_DISPATCH_TPB(RUN, HAS_F_EXT, 32, __VA_ARGS__)                 \
    default:                                                                  \
      throw std::runtime_error("RNEA: unsupported threads_per_batch");        \
  }

}  // namespace

/**
 * @brief Launch geometry the compiled backend uses, for parity checks against DynamicsLaunchCfg.
 *
 * @return (blocks per grid, threads per block, dynamic shared memory in bytes).
 */
std::tuple<int64_t, int64_t, int64_t> rnea_launch_config(
  const int64_t batch_size,
  const int64_t num_links,
  const int64_t threads_per_batch,
  const bool backward)
{
  TORCH_CHECK(batch_size > 0 && num_links > 0 && threads_per_batch > 0, "arguments must be positive");
  const RneaLaunchConfig config = calculate_launch_config(
    static_cast<int>(batch_size), static_cast<int>(num_links), static_cast<int>(threads_per_batch), backward);
  return std::make_tuple(
    static_cast<int64_t>(config.blocks_per_grid),
    static_cast<int64_t>(config.threads_per_block),
    static_cast<int64_t>(config.shared_mem_bytes));
}

/**
 * @brief RNEA forward: tau = RNEA(q, qd, qdd, f_ext); fills forward_cache for the backward pass.
 *
 * Same contract as cuda_core_backend.dynamics.launch_rnea_forward. Launches on the
 * current CUDA stream.
 */
void launch_rnea_forward(
  torch::Tensor tau,                   // [batch_size, num_dof]
  const torch::Tensor q,               // [batch_size, num_dof]
  const torch::Tensor qd,              // [batch_size, num_dof]
  const torch::Tensor qdd,             // [batch_size, num_dof]
  const torch::Tensor fixed_transforms,  // [num_links, 3, 4]
  const torch::Tensor link_masses_com,   // [num_links, 4]
  const torch::Tensor link_inertias,     // [num_links, 8]
  const torch::Tensor joint_map_type,    // [num_links] int8
  const torch::Tensor joint_map,         // [num_links] int16
  const torch::Tensor link_map,          // [num_links] int16
  const torch::Tensor joint_offset_map,  // [num_links * 2]
  const torch::Tensor gravity,           // [6]
  const torch::Tensor level_starts,      // [n_levels + 1] int16
  const torch::Tensor level_links,       // [num_links] int16
  torch::Tensor forward_cache,           // [batch_size, num_links * 20]
  const int64_t batch_size,
  const int64_t num_links,
  const int64_t num_dof,
  const int64_t n_levels,
  const int64_t threads_per_batch,
  const std::optional<torch::Tensor> f_ext)  // [batch_size, num_links, 6] or None
{
  check_sizes(batch_size, num_links, num_dof, n_levels, threads_per_batch);
  const int64_t state_numel = batch_size * num_dof;
  check_tensor(tau, "tau", torch::kFloat32, state_numel);
  check_tensor(q, "q", torch::kFloat32, state_numel);
  check_tensor(qd, "qd", torch::kFloat32, state_numel);
  check_tensor(qdd, "qdd", torch::kFloat32, state_numel);
  check_tensor(forward_cache, "forward_cache", torch::kFloat32, batch_size * num_links * CACHE_FLOATS_PER_LINK);
  const RobotParams robot = check_robot_params(
    fixed_transforms, link_masses_com, link_inertias, joint_map_type, joint_map, link_map,
    joint_offset_map, gravity, level_starts, level_links, num_links, n_levels);

  const bool has_f_ext = f_ext.has_value();
  const float *f_ext_ptr = nullptr;
  if (has_f_ext) {
    check_tensor(*f_ext, "f_ext", torch::kFloat32, batch_size * num_links * SPATIAL_DIM);
    f_ext_ptr = f_ext->data_ptr<float>();
  }

  const RneaLaunchConfig config = calculate_launch_config(
    static_cast<int>(batch_size), static_cast<int>(num_links), static_cast<int>(threads_per_batch), false);
  cudaStream_t stream = curobo::common::get_cuda_stream();
  const int tpb = static_cast<int>(threads_per_batch);

  if (has_f_ext) {
    CUROBO_RNEA_DISPATCH(
      run_forward, true, tpb, config, stream, tau.data_ptr<float>(), q.data_ptr<float>(),
      qd.data_ptr<float>(), qdd.data_ptr<float>(), robot, forward_cache.data_ptr<float>(), f_ext_ptr,
      static_cast<int>(batch_size), static_cast<int>(n_levels), static_cast<int>(num_links),
      static_cast<int>(num_dof));
  } else {
    CUROBO_RNEA_DISPATCH(
      run_forward, false, tpb, config, stream, tau.data_ptr<float>(), q.data_ptr<float>(),
      qd.data_ptr<float>(), qdd.data_ptr<float>(), robot, forward_cache.data_ptr<float>(), f_ext_ptr,
      static_cast<int>(batch_size), static_cast<int>(n_levels), static_cast<int>(num_links),
      static_cast<int>(num_dof));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

/**
 * @brief RNEA backward (VJP): dL/dq, dL/dqd, dL/dqdd and optionally dL/df_ext from dL/dtau.
 *
 * Same contract as cuda_core_backend.dynamics.launch_rnea_backward: the gradient
 * buffers must be zeroed by the caller; forward_cache comes from the forward launch.
 */
void launch_rnea_backward(
  torch::Tensor grad_q,                // [batch_size, num_dof]
  torch::Tensor grad_qd,               // [batch_size, num_dof]
  torch::Tensor grad_qdd,              // [batch_size, num_dof]
  const torch::Tensor grad_tau,        // [batch_size, num_dof]
  const torch::Tensor q,               // [batch_size, num_dof]
  const torch::Tensor qd,              // [batch_size, num_dof]
  const torch::Tensor fixed_transforms,
  const torch::Tensor link_masses_com,
  const torch::Tensor link_inertias,
  const torch::Tensor joint_map_type,
  const torch::Tensor joint_map,
  const torch::Tensor link_map,
  const torch::Tensor joint_offset_map,
  const torch::Tensor gravity,
  const torch::Tensor level_starts,
  const torch::Tensor level_links,
  const torch::Tensor forward_cache,   // [batch_size, num_links * 20]
  const int64_t batch_size,
  const int64_t num_links,
  const int64_t num_dof,
  const int64_t n_levels,
  const int64_t threads_per_batch,
  std::optional<torch::Tensor> grad_f_ext)  // [batch_size, num_links, 6] or None
{
  check_sizes(batch_size, num_links, num_dof, n_levels, threads_per_batch);
  const int64_t state_numel = batch_size * num_dof;
  check_tensor(grad_q, "grad_q", torch::kFloat32, state_numel);
  check_tensor(grad_qd, "grad_qd", torch::kFloat32, state_numel);
  check_tensor(grad_qdd, "grad_qdd", torch::kFloat32, state_numel);
  check_tensor(grad_tau, "grad_tau", torch::kFloat32, state_numel);
  check_tensor(q, "q", torch::kFloat32, state_numel);
  check_tensor(qd, "qd", torch::kFloat32, state_numel);
  check_tensor(forward_cache, "forward_cache", torch::kFloat32, batch_size * num_links * CACHE_FLOATS_PER_LINK);
  const RobotParams robot = check_robot_params(
    fixed_transforms, link_masses_com, link_inertias, joint_map_type, joint_map, link_map,
    joint_offset_map, gravity, level_starts, level_links, num_links, n_levels);

  const bool has_f_ext = grad_f_ext.has_value();
  float *grad_f_ext_ptr = nullptr;
  if (has_f_ext) {
    check_tensor(*grad_f_ext, "grad_f_ext", torch::kFloat32, batch_size * num_links * SPATIAL_DIM);
    grad_f_ext_ptr = grad_f_ext->data_ptr<float>();
  }

  const RneaLaunchConfig config = calculate_launch_config(
    static_cast<int>(batch_size), static_cast<int>(num_links), static_cast<int>(threads_per_batch), true);
  cudaStream_t stream = curobo::common::get_cuda_stream();
  const int tpb = static_cast<int>(threads_per_batch);

  if (has_f_ext) {
    CUROBO_RNEA_DISPATCH(
      run_backward, true, tpb, config, stream, grad_q.data_ptr<float>(), grad_qd.data_ptr<float>(),
      grad_qdd.data_ptr<float>(), grad_f_ext_ptr, grad_tau.data_ptr<float>(), q.data_ptr<float>(),
      qd.data_ptr<float>(), robot, forward_cache.data_ptr<float>(), static_cast<int>(batch_size),
      static_cast<int>(n_levels), static_cast<int>(num_links), static_cast<int>(num_dof));
  } else {
    CUROBO_RNEA_DISPATCH(
      run_backward, false, tpb, config, stream, grad_q.data_ptr<float>(), grad_qd.data_ptr<float>(),
      grad_qdd.data_ptr<float>(), grad_f_ext_ptr, grad_tau.data_ptr<float>(), q.data_ptr<float>(),
      qd.data_ptr<float>(), robot, forward_cache.data_ptr<float>(), static_cast<int>(batch_size),
      static_cast<int>(n_levels), static_cast<int>(num_links), static_cast<int>(num_dof));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#undef CUROBO_RNEA_DISPATCH
#undef CUROBO_RNEA_DISPATCH_TPB

}  // namespace dynamics
}  // namespace curobo
