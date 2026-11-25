// SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility prelude for compiling cuRobo's CUDA kernels with hipRTC. It is force-included ahead of every
// kernel source by the hip_core backend.
//
// CUDA's warp-sync builtins take a 32-bit lane mask. HIP's take a 64-bit mask that must name the active lanes.
// Each overload below accepts the 32-bit mask the kernels pass and calls the HIP builtin with the active-lane
// set. Where the result depends on the mask (ballot, match, all, any), it is then restricted to the lanes the
// caller named, as CUDA does. Shuffles and __syncwarp do not depend on the mask value.
#pragma once
#include <hip/hip_runtime.h>
#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#error "compile with -DHIP_ENABLE_WARP_SYNC_BUILTINS"
#endif

namespace curobo_hip {
__device__ inline unsigned long long active() { return __activemask(); }

// CUDA's round-to-nearest intrinsics return a correctly rounded result that is never merged
// into a fused multiply-add. HIP's are the plain operators, which the compiler contracts.
// An ordinary expression that reuses the very product or sum passed through one of these
// intrinsics shares the uncontracted value and is then not contracted either.
__device__ inline float fadd_rn(float x, float y) {
#pragma clang fp contract(off)
  return x + y;
}
__device__ inline float fsub_rn(float x, float y) {
#pragma clang fp contract(off)
  return x - y;
}
__device__ inline float fmul_rn(float x, float y) {
#pragma clang fp contract(off)
  return x * y;
}
__device__ inline double dadd_rn(double x, double y) {
#pragma clang fp contract(off)
  return x + y;
}
__device__ inline double dsub_rn(double x, double y) {
#pragma clang fp contract(off)
  return x - y;
}
__device__ inline double dmul_rn(double x, double y) {
#pragma clang fp contract(off)
  return x * y;
}
}  // namespace curobo_hip

#define __fadd_rn curobo_hip::fadd_rn
#define __fsub_rn curobo_hip::fsub_rn
#define __fmul_rn curobo_hip::fmul_rn
#define __dadd_rn curobo_hip::dadd_rn
#define __dsub_rn curobo_hip::dsub_rn
#define __dmul_rn curobo_hip::dmul_rn

__device__ inline void __syncwarp(unsigned /*mask*/) { __syncwarp(curobo_hip::active()); }
__device__ inline unsigned __ballot_sync(unsigned mask, int pred) {
  return static_cast<unsigned>(__ballot_sync(curobo_hip::active(), pred)) & mask;
}
__device__ inline int __all_sync(unsigned mask, int pred) {
  const unsigned named = static_cast<unsigned>(curobo_hip::active()) & mask;
  return (__ballot_sync(mask, pred) & named) == named;
}
__device__ inline int __any_sync(unsigned mask, int pred) { return __ballot_sync(mask, pred) != 0u; }
template <typename T>
__device__ inline T __shfl_sync(unsigned /*mask*/, T var, int src_lane, int width = warpSize) {
  return __shfl_sync(curobo_hip::active(), var, src_lane, width);
}
template <typename T>
__device__ inline T __shfl_up_sync(unsigned /*mask*/, T var, unsigned int delta, int width = warpSize) {
  return __shfl_up_sync(curobo_hip::active(), var, delta, width);
}
template <typename T>
__device__ inline T __shfl_down_sync(unsigned /*mask*/, T var, unsigned int delta, int width = warpSize) {
  return __shfl_down_sync(curobo_hip::active(), var, delta, width);
}
template <typename T>
__device__ inline T __shfl_xor_sync(unsigned /*mask*/, T var, int lane_mask, int width = warpSize) {
  return __shfl_xor_sync(curobo_hip::active(), var, lane_mask, width);
}
template <typename T>
__device__ inline unsigned __match_any_sync(unsigned mask, T value) {
  return static_cast<unsigned>(__match_any_sync(curobo_hip::active(), value)) & mask;
}

// The same builtins with a signed mask, as in "__shfl_down_sync(-1, v, 1)" or a mask held in an int.
__device__ inline void __syncwarp(int mask) { __syncwarp(static_cast<unsigned>(mask)); }
__device__ inline unsigned __ballot_sync(int mask, int pred) {
  return __ballot_sync(static_cast<unsigned>(mask), pred);
}
__device__ inline int __all_sync(int mask, int pred) { return __all_sync(static_cast<unsigned>(mask), pred); }
__device__ inline int __any_sync(int mask, int pred) { return __any_sync(static_cast<unsigned>(mask), pred); }
template <typename T>
__device__ inline T __shfl_sync(int mask, T var, int src_lane, int width = warpSize) {
  return __shfl_sync(static_cast<unsigned>(mask), var, src_lane, width);
}
template <typename T>
__device__ inline T __shfl_up_sync(int mask, T var, unsigned int delta, int width = warpSize) {
  return __shfl_up_sync(static_cast<unsigned>(mask), var, delta, width);
}
template <typename T>
__device__ inline T __shfl_down_sync(int mask, T var, unsigned int delta, int width = warpSize) {
  return __shfl_down_sync(static_cast<unsigned>(mask), var, delta, width);
}
template <typename T>
__device__ inline T __shfl_xor_sync(int mask, T var, int lane_mask, int width = warpSize) {
  return __shfl_xor_sync(static_cast<unsigned>(mask), var, lane_mask, width);
}
template <typename T>
__device__ inline unsigned __match_any_sync(int mask, T value) {
  return __match_any_sync(static_cast<unsigned>(mask), value);
}
