/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
 * SPDX-License-Identifier: Apache-2.0
 */
#include <optional>
#include <tuple>

#include <torch/extension.h>
#include <pybind11/stl.h>

namespace curobo {
namespace dynamics {

std::tuple<int64_t, int64_t, int64_t> rnea_launch_config(
  const int64_t batch_size,
  const int64_t num_links,
  const int64_t threads_per_batch,
  const bool backward);

void launch_rnea_forward(
  torch::Tensor tau,
  const torch::Tensor q,
  const torch::Tensor qd,
  const torch::Tensor qdd,
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
  torch::Tensor forward_cache,
  const int64_t batch_size,
  const int64_t num_links,
  const int64_t num_dof,
  const int64_t n_levels,
  const int64_t threads_per_batch,
  const std::optional<torch::Tensor> f_ext);

void launch_rnea_backward(
  torch::Tensor grad_q,
  torch::Tensor grad_qd,
  torch::Tensor grad_qdd,
  const torch::Tensor grad_tau,
  const torch::Tensor q,
  const torch::Tensor qd,
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
  const torch::Tensor forward_cache,
  const int64_t batch_size,
  const int64_t num_links,
  const int64_t num_dof,
  const int64_t n_levels,
  const int64_t threads_per_batch,
  std::optional<torch::Tensor> grad_f_ext);

}  // namespace dynamics
}  // namespace curobo

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.doc() = "RNEA inverse dynamics kernels (compiled backend)";

  m.def("launch_rnea_forward", &curobo::dynamics::launch_rnea_forward,
        "RNEA forward: joint torques and the forward cache",
        py::arg("tau"),
        py::arg("q"),
        py::arg("qd"),
        py::arg("qdd"),
        py::arg("fixed_transforms"),
        py::arg("link_masses_com"),
        py::arg("link_inertias"),
        py::arg("joint_map_type"),
        py::arg("joint_map"),
        py::arg("link_map"),
        py::arg("joint_offset_map"),
        py::arg("gravity"),
        py::arg("level_starts"),
        py::arg("level_links"),
        py::arg("forward_cache"),
        py::arg("batch_size"),
        py::arg("num_links"),
        py::arg("num_dof"),
        py::arg("n_levels"),
        py::arg("threads_per_batch") = 1,
        py::arg("f_ext") = py::none());

  m.def("launch_rnea_backward", &curobo::dynamics::launch_rnea_backward,
        "RNEA backward: gradients with respect to q, qd, qdd and optionally f_ext",
        py::arg("grad_q"),
        py::arg("grad_qd"),
        py::arg("grad_qdd"),
        py::arg("grad_tau"),
        py::arg("q"),
        py::arg("qd"),
        py::arg("fixed_transforms"),
        py::arg("link_masses_com"),
        py::arg("link_inertias"),
        py::arg("joint_map_type"),
        py::arg("joint_map"),
        py::arg("link_map"),
        py::arg("joint_offset_map"),
        py::arg("gravity"),
        py::arg("level_starts"),
        py::arg("level_links"),
        py::arg("forward_cache"),
        py::arg("batch_size"),
        py::arg("num_links"),
        py::arg("num_dof"),
        py::arg("n_levels"),
        py::arg("threads_per_batch") = 1,
        py::arg("grad_f_ext") = py::none());

  m.def("rnea_launch_config", &curobo::dynamics::rnea_launch_config,
        "Launch geometry (blocks, threads per block, shared memory bytes) of an RNEA kernel",
        py::arg("batch_size"),
        py::arg("num_links"),
        py::arg("threads_per_batch"),
        py::arg("backward"));
}
