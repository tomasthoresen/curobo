/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * Modified 2026 by Tomas Thoresen: port to AMD ROCm/HIP (gfx1151).
 */
/**
 * @file rnea_forward_kernel.cuh
 * @brief RNEA forward pass CUDA kernel with tree-level parallelism.
 *
 * Computes inverse dynamics: τ = RNEA(q, q̇, q̈, f_ext) for a batch of robot states.
 * Also saves v, a, f to forward_cache for backward kernel reuse.
 *
 * Algorithm (Featherstone Chapter 5):
 *   Loop 1 (base→tip): Compute spatial velocity v[k] and acceleration a[k]
 *   Loop 2 (independent): Flush v+a to cache, then compute f[k] = I·a + v ×* (I·v) - f_ext[k]
 *   Loop 3 (tip→base): Propagate forces, extract torques τ = S^T · f
 *   Flush f: Parallel copy from shared memory to forward_cache
 *
 * Threading: TPB threads per batch element. When TPB=1, degenerates to serial.
 * Links at the same tree depth are processed in parallel by different workers.
 *
 * Shared memory per batch: N_LINKS × 12 floats = v(6) + a/f(6).
 *
 * Template params:
 *   - N_LINKS, N_DOF: Compile-time smem layout, loop unrolling
 *   - TPB: Threads per batch (1=serial, >1=tree-parallel)
 *   - HAS_EXTERNAL_FORCES: When true, subtracts f_ext from computed forces
 *
 * The kernel body takes the link and DOF counts at runtime; the templated
 * kernel passes its compile-time values, the `_rt` kernel takes them as
 * arguments (compiled pybind backend, one instantiation per thread count).
 */

#pragma once

#include "spatial_algebra.cuh"
#include "rnea_helpers.cuh"

namespace curobo {
namespace dynamics {

/**
 * @brief Shared body of the rnea forward kernels: link and DOF counts at runtime.
 *
 * Called by rnea_forward_kernel with its compile-time counts (the cuda.core
 * path, one kernel per robot) and by rnea_forward_kernel_rt with runtime counts (the
 * compiled pybind path, one kernel per thread count).
 *
 * @tparam TPB Threads per batch element (power of 2, 1..32).
 * @tparam HAS_EXTERNAL_FORCES See rnea_forward_kernel.
 * @param n_links Number of links.
 * @param n_dof   Number of degrees of freedom.
 */
template<int TPB, bool HAS_EXTERNAL_FORCES>
__device__ __forceinline__ void rnea_forward_body(
    float * __restrict__ tau,
    const float * __restrict__ q,
    const float * __restrict__ qd,
    const float * __restrict__ qdd,
    const float * __restrict__ fixed_transforms,
    const float * __restrict__ link_masses_com,
    const float * __restrict__ link_inertias,
    const int8_t * __restrict__ joint_map_type,
    const int16_t * __restrict__ joint_map,
    const int16_t * __restrict__ link_map,
    const float * __restrict__ joint_offset_map,
    const float * __restrict__ gravity,
    const int16_t * __restrict__ level_starts,
    const int16_t * __restrict__ level_links,
    float * __restrict__ forward_cache,
    const float * __restrict__ f_ext,  // External forces (or nullptr if !HAS_EXTERNAL_FORCES)
    const int batch_size,
    const int n_levels,
    const int n_links,
    const int n_dof
) {
    // Thread mapping: TPB threads cooperate on one batch element
    const int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    const int batch = global_tid / TPB;
    const int worker = (TPB > 1) ? (global_tid % TPB) : 0;

    // For TPB=1 (serial), early return is safe; no sync needed
    if (TPB == 1 && batch >= batch_size) return;

    const bool active = (batch < batch_size);

    // Shared memory layout (offsets from the runtime link count):
    //   sv: n_links × 6 floats  (velocity)
    //   sf: n_links × 6 floats  (accel in Loop 1, force in Loop 2-3)
    const int smem_per_batch = n_links * SMEM_FLOATS_PER_LINK;
    extern __shared__ __align__(16) float smem[];
    const int local_batch = threadIdx.x / TPB;
    float *sv = &smem[local_batch * smem_per_batch];
    float *sf = &smem[local_batch * smem_per_batch + n_links * 6];

    const int q_base = batch * n_dof;

    // Warp mask for intra-batch synchronization (TPB threads within a warp)
    // mask_t is 64 bits wide on HIP, where the warp-sync builtins require it.
    curobo::common::mask_t batch_mask = 0;
    if (TPB > 1) {
        constexpr curobo::common::mask_t tpb_mask =
            (TPB >= 32) ? curobo::common::fullMask : ((1u << TPB) - 1u);
        const int lane = threadIdx.x & 31;
        batch_mask = tpb_mask << ((lane / TPB) * TPB);
    }

    // Zero tau output (distributed across workers, runtime n_dof bound)
    if (active) {
        for (int j = worker; j < n_dof; j += TPB) {
            tau[q_base + j] = 0.0f;
        }
    }

    // ==================================================================
    // Loop 1: Forward pass, compute v[k] and a[k] (root→leaf by level)
    // ==================================================================
    for (int level = 0; level < n_levels; level++) {
        if (active) {
            const int start = level_starts[level];
            const int end = level_starts[level + 1];
            for (int idx = start + worker; idx < end; idx += TPB) {
                const int k = level_links[idx];
                const int8_t j_type = joint_map_type[k];
                const int16_t j_idx = joint_map[k];
                const int16_t parent = link_map[k];
                const bool is_root = (parent < 0) || (parent == k);

                float q_eff = 0.0f, qd_eff = 0.0f, qdd_eff = 0.0f;
                float multiplier = 1.0f;
                if (j_type != FIXED && j_idx >= 0) {
                    const float2 j_off = *(const float2 *)&joint_offset_map[k * 2];
                    multiplier = j_off.x;
                    q_eff = multiplier * q[q_base + j_idx] + j_off.y;
                    qd_eff = multiplier * qd[q_base + j_idx];
                    qdd_eff = multiplier * qdd[q_base + j_idx];
                }

                float R[9], p[3];
                compute_local_Rp(&fixed_transforms[k * 12], j_type, q_eff, R, p);

                float *v_k = &sv[k * 6];
                float *a_k = &sf[k * 6];

                if (is_root) {
                    for (int i = 0; i < 6; i++) v_k[i] = 0.0f;
                    if (j_type != FIXED) {
                        const int s_idx = get_s_index(j_type);
                        v_k[s_idx] = qd_eff;
                    }
                    spatial_Xv(R, p, gravity, a_k);
                    if (j_type != FIXED) {
                        const int s_idx = get_s_index(j_type);
                        a_k[s_idx] += qdd_eff;
                    }
                } else {
                    spatial_Xv(R, p, &sv[parent * 6], v_k);
                    if (j_type != FIXED) {
                        const int s_idx = get_s_index(j_type);
                        v_k[s_idx] += qd_eff;
                    }
                    spatial_Xv(R, p, &sf[parent * 6], a_k);
                    if (j_type != FIXED) {
                        const int s_idx = get_s_index(j_type);
                        a_k[s_idx] += qdd_eff;
                    }
                }

                if (j_type != FIXED) {
                    const int s_idx = get_s_index(j_type);
                    float coriolis[6];
                    motion_cross_S(v_k, s_idx, qd_eff, coriolis);
                    for (int i = 0; i < 6; i++) a_k[i] += coriolis[i];
                }
                // v and a are now in shared memory; NO global stores here.
                // They are flushed to forward_cache in a parallel loop below.
            }
        }
        if (TPB > 1) __syncwarp(batch_mask);
    }

    // ==================================================================
    // Loop 2: Flush v+a to cache, then compute f[k] = I·a + v ×* (I·v)
    //
    // Merged: each worker saves v+a to global cache BEFORE overwriting
    // sf (which holds a) with f.  This avoids a separate parallel loop
    // and syncwarp, and the global store can overlap with subsequent math.
    // Independent per link; distribute across workers.
    // ==================================================================
    if (active) {
        for (int k = worker; k < n_links; k += TPB) {
            const float *v_k = &sv[k * 6];
            float *af_k = &sf[k * 6];

            // Flush v+a to forward cache while a is still in sf
            float *cache_k = &forward_cache[
                batch * n_links * CACHE_FLOATS_PER_LINK
                + k * CACHE_FLOATS_PER_LINK];
            cache_save_va(v_k, af_k, &cache_k[CACHE_VA_OFFSET]);

            // Now compute f[k] = I·a + v ×* (I·v), overwriting a with f
            const float *mc = &link_masses_com[k * 4];
            const float *inertia = &link_inertias[k * 8];

            float Ia[6];
            spatial_inertia_times_vec(mc, inertia, af_k, Ia);
            float Iv[6];
            spatial_inertia_times_vec(mc, inertia, v_k, Iv);
            float vxIv[6];
            force_cross(v_k, Iv, vxIv);

            // f[k] = I·a + v ×* (I·v) - f_ext[k]
            // External forces are subtracted (they reduce the required torque)
            if constexpr (HAS_EXTERNAL_FORCES) {
                const float *f_ext_k = &f_ext[batch * n_links * 6 + k * 6];
                for (int i = 0; i < 6; i++) {
                    af_k[i] = Ia[i] + vxIv[i] - f_ext_k[i];
                }
            } else {
                for (int i = 0; i < 6; i++) af_k[i] = Ia[i] + vxIv[i];
            }
        }
    }
    if (TPB > 1) __syncwarp(batch_mask);

    // ==================================================================
    // Loop 3: Backward pass, force propagation and torque extraction
    //         (leaf→root by level)
    // ==================================================================
    for (int level = n_levels - 1; level >= 0; level--) {
        if (active) {
            const int start = level_starts[level];
            const int end = level_starts[level + 1];
            for (int idx = start + worker; idx < end; idx += TPB) {
                const int k = level_links[idx];
                const int8_t j_type = joint_map_type[k];
                const int16_t j_idx = joint_map[k];
                const int16_t parent = link_map[k];
                const bool is_root = (parent < 0) || (parent == k);

                const float *f_k = &sf[k * 6];

                // Extract torque: τ[joint] += multiplier · S^T · f[k]
                if (j_type != FIXED && j_idx >= 0) {
                    const float2 j_off = *(const float2 *)&joint_offset_map[k * 2];
                    const int s_idx = get_s_index(j_type);
                    const float tau_k = j_off.x * f_k[s_idx];
                    if (TPB > 1) {
                        atomicAdd(&tau[q_base + j_idx], tau_k);
                    } else {
                        tau[q_base + j_idx] += tau_k;
                    }
                }

                // Propagate force to parent: f[parent] += X^T · f[k]
                if (!is_root) {
                    float q_eff = 0.0f;
                    if (j_type != FIXED && j_idx >= 0) {
                        const float2 j_off = *(const float2 *)&joint_offset_map[k * 2];
                        q_eff = j_off.x * q[q_base + j_idx] + j_off.y;
                    }
                    float R[9], p_loc[3];
                    compute_local_Rp(&fixed_transforms[k * 12], j_type, q_eff, R, p_loc);

                    float f_contrib[6];
                    spatial_XTf(R, p_loc, f_k, f_contrib);

                    float *f_parent = &sf[parent * 6];
                    if (TPB > 1) {
                        for (int i = 0; i < 6; i++) atomicAdd(&f_parent[i], f_contrib[i]);
                    } else {
                        for (int i = 0; i < 6; i++) f_parent[i] += f_contrib[i];
                    }
                }
            }
        }
        if (TPB > 1) __syncwarp(batch_mask);
    }

    // ==================================================================
    // Save final forces to forward cache (for backward kernel reuse)
    // ==================================================================
    if (active) {
        for (int k = worker; k < n_links; k += TPB) {
            float *cache_k = &forward_cache[
                batch * n_links * CACHE_FLOATS_PER_LINK
                + k * CACHE_FLOATS_PER_LINK];
            cache_save_f(&sf[k * 6], &cache_k[CACHE_F_OFFSET]);  // f at [12:20]
        }
    }
}

/**
 * @brief RNEA forward kernel: computes joint torques from (q, qd, qdd, f_ext).
 *
 * @tparam N_LINKS Compile-time link count (must be > 0).
 * @tparam N_DOF   Compile-time DOF count (must be > 0).
 * @tparam TPB     Threads per batch element (must be power of 2, 1..32).
 *                 TPB=1 gives serial per-batch execution (no sync overhead).
 *                 TPB>1 enables tree-level parallelism for branched robots.
 * @tparam HAS_EXTERNAL_FORCES When true, subtracts f_ext from computed forces.
 *                 When false, f_ext pointer is ignored (zero overhead).
 *
 * @param f_ext External spatial forces [batch_size, N_LINKS, 6] or nullptr.
 *              Each f_ext[b,k] is a 6D spatial wrench in link k's frame.
 *              Forces are subtracted: f[k] = I·a + v×*(I·v) - f_ext[k].
 *              Only accessed when HAS_EXTERNAL_FORCES=true.
 */
template<int N_LINKS, int N_DOF, int TPB = 1, bool HAS_EXTERNAL_FORCES = false>
__global__ void rnea_forward_kernel(
    float * __restrict__ tau,
    const float * __restrict__ q,
    const float * __restrict__ qd,
    const float * __restrict__ qdd,
    const float * __restrict__ fixed_transforms,
    const float * __restrict__ link_masses_com,
    const float * __restrict__ link_inertias,
    const int8_t * __restrict__ joint_map_type,
    const int16_t * __restrict__ joint_map,
    const int16_t * __restrict__ link_map,
    const float * __restrict__ joint_offset_map,
    const float * __restrict__ gravity,
    const int16_t * __restrict__ level_starts,
    const int16_t * __restrict__ level_links,
    float * __restrict__ forward_cache,
    const float * __restrict__ f_ext,  // External forces (or nullptr if !HAS_EXTERNAL_FORCES)
    const int batch_size,
    const int n_levels
) {
    static_assert(N_LINKS > 0, "N_LINKS must be > 0");
    static_assert(N_DOF > 0, "N_DOF must be > 0");
    rnea_forward_body<TPB, HAS_EXTERNAL_FORCES>(
        tau,
        q,
        qd,
        qdd,
        fixed_transforms,
        link_masses_com,
        link_inertias,
        joint_map_type,
        joint_map,
        link_map,
        joint_offset_map,
        gravity,
        level_starts,
        level_links,
        forward_cache,
        f_ext,
        batch_size,
        n_levels,
        N_LINKS,
        N_DOF);
}

/**
 * @brief rnea_forward_kernel with the link and DOF counts as kernel arguments.
 *
 * Same contract as rnea_forward_kernel; used by the compiled (pybind) backend,
 * which cannot instantiate a kernel per robot.
 */
template<int TPB, bool HAS_EXTERNAL_FORCES>
__global__ void rnea_forward_kernel_rt(
    float * __restrict__ tau,
    const float * __restrict__ q,
    const float * __restrict__ qd,
    const float * __restrict__ qdd,
    const float * __restrict__ fixed_transforms,
    const float * __restrict__ link_masses_com,
    const float * __restrict__ link_inertias,
    const int8_t * __restrict__ joint_map_type,
    const int16_t * __restrict__ joint_map,
    const int16_t * __restrict__ link_map,
    const float * __restrict__ joint_offset_map,
    const float * __restrict__ gravity,
    const int16_t * __restrict__ level_starts,
    const int16_t * __restrict__ level_links,
    float * __restrict__ forward_cache,
    const float * __restrict__ f_ext,  // External forces (or nullptr if !HAS_EXTERNAL_FORCES)
    const int batch_size,
    const int n_levels,
    const int n_links,
    const int n_dof
) {
    rnea_forward_body<TPB, HAS_EXTERNAL_FORCES>(
        tau,
        q,
        qd,
        qdd,
        fixed_transforms,
        link_masses_com,
        link_inertias,
        joint_map_type,
        joint_map,
        link_map,
        joint_offset_map,
        gravity,
        level_starts,
        level_links,
        forward_cache,
        f_ext,
        batch_size,
        n_levels,
        n_links,
        n_dof);
}

} // namespace dynamics
} // namespace curobo
