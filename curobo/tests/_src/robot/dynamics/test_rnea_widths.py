# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for RNEA at every cooperative width the launch policy selects.

The launch policy assigns 1, 2, 4, 16 or 32 threads to each robot state,
depending on the tree shape, the batch size and the SM count. Below 32, several
states share one warp, and a batch that does not fill its last block leaves
lanes without a state. Each case here fixes the SM count given to the policy,
checks the width the kernels are launched with, and compares the torques and
all gradients with the float64 NumPy reference, with and without external
wrenches.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import curobo._src.runtime as curobo_runtime
from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
    FIXED,
    build_joint_transform,
    get_joint_s_index,
    rnea_backward_from_kinematics_params,
    rnea_from_kinematics_params,
    spatial_transform_transpose_multiply,
)

#: SM count given to the launch policy, so the width does not depend on the device.
POLICY_SM_COUNT = 1
GRAVITY = -9.81
#: The kernels run in float32 and the reference in float64.
ABSOLUTE_TOLERANCE = 1.0e-3
RELATIVE_TOLERANCE = 1.0e-3

#: ``(robot file, batch size, width the policy selects)``. The serial chains
#: reach 32, 4, 2 and 1, the single-level tree 32 and 4, the multi-level tree
#: 32 and 16. Most batch sizes leave the last warp or block partially filled.
WIDTH_CASES = [
    ("franka.yml", 5, 32),
    ("franka.yml", 23, 4),
    ("franka.yml", 32, 2),
    ("franka.yml", 33, 2),
    ("franka.yml", 67, 1),
    ("ur10e.yml", 24, 4),
    ("ur10e.yml", 129, 1),
    ("dual_ur10e.yml", 3, 32),
    ("dual_ur10e.yml", 7, 4),
    ("dual_ur10e.yml", 33, 4),
    ("unitree_g1.yml", 3, 32),
    ("unitree_g1.yml", 5, 16),
    ("unitree_g1.yml", 8, 16),
    ("unitree_g1.yml", 33, 16),
]
#: Cases repeated with external wrenches: one per width and tree shape.
WRENCH_CASES = [
    ("franka.yml", 5, 32),
    ("franka.yml", 23, 4),
    ("franka.yml", 33, 2),
    ("franka.yml", 67, 1),
    ("dual_ur10e.yml", 3, 32),
    ("dual_ur10e.yml", 33, 4),
    ("unitree_g1.yml", 3, 32),
    ("unitree_g1.yml", 5, 16),
]

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_dynamics(robot_file: str, batch_size: int) -> tuple[Dynamics, KinematicsParams]:
    """Return a dynamics model set up for ``batch_size`` states, and its robot parameters."""
    device_cfg = DeviceCfg()
    kinematics = KinematicsCfg.from_robot_yaml_file(robot_file, device_cfg=device_cfg)
    params = kinematics.kinematics_config
    dynamics = Dynamics(
        DynamicsCfg(
            kinematics_config=params,
            device_cfg=device_cfg,
            gravity=[0.0, 0.0, GRAVITY],
        )
    )
    dynamics.setup_batch_size(batch_size=batch_size)
    return dynamics, params


def _inputs(batch_size: int, num_dof: int) -> tuple[np.ndarray, ...]:
    """Return seeded float32 ``q``, ``qd``, ``qdd`` and ``tau_bar`` of shape ``[batch, dof]``."""
    rng = np.random.RandomState(2000 + batch_size)
    q = rng.uniform(-1.5, 1.5, size=(batch_size, num_dof)).astype(np.float32)
    return (q, *[rng.randn(batch_size, num_dof).astype(np.float32) for _ in range(3)])


def _forward_backward(dynamics, q, qd, qdd, tau_bar, f_ext=None):
    """Run RNEA and its vector-Jacobian product on the device.

    Args:
        dynamics: Model set up for the batch size of the inputs.
        q: Joint positions, shape ``[batch, dof]``, float32.
        qd: Joint velocities, shape ``[batch, dof]``, float32.
        qdd: Joint accelerations, shape ``[batch, dof]``, float32.
        tau_bar: Torque cotangent, shape ``[batch, dof]``, float32.
        f_ext: External wrenches, shape ``[batch, links, 6]``, float32, or None.

    Returns:
        ``tau``, ``grad_q``, ``grad_qd``, ``grad_qdd`` and ``grad_f_ext`` (None
        without wrenches) as NumPy arrays.
    """
    device = dynamics.device_cfg.device

    def leaf(value):
        return torch.tensor(value, dtype=torch.float32, device=device, requires_grad=True)

    leaves = [leaf(value) for value in (q, qd, qdd)]
    wrench = leaf(f_ext) if f_ext is not None else None
    tau = dynamics.compute_inverse_dynamics(
        JointState(position=leaves[0], velocity=leaves[1], acceleration=leaves[2]),
        f_ext=wrench,
    )
    tau_np = tau.detach().cpu().numpy().copy()
    tau.backward(torch.tensor(tau_bar, dtype=torch.float32, device=device).reshape(tau.shape))
    grads = [value.grad.detach().cpu().numpy().copy() for value in leaves]
    grad_f_ext = wrench.grad.detach().cpu().numpy().copy() if wrench is not None else None
    return (tau_np, *grads, grad_f_ext)


def _reference_wrench_gradient(params: KinematicsParams, q: np.ndarray, tau_bar: np.ndarray):
    """Return the gradient of ``tau_bar . tau`` with respect to the external wrenches.

    A wrench on link ``k`` is subtracted from that link's force, which is then
    carried to the base through ``X^T`` of every joint on the way and projected
    on each joint axis. The gradient is the adjoint of that chain, propagated
    from the base to the tips.

    Args:
        params: Robot parameters.
        q: Joint positions of one state, shape ``[dof]``.
        tau_bar: Torque cotangent of one state, shape ``[dof]``.

    Returns:
        Array of shape ``[links, 6]``.
    """
    fixed_transforms = params.fixed_transforms.cpu().numpy()
    link_map = params.link_map.cpu().numpy()
    joint_map = params.joint_map.cpu().numpy()
    joint_map_type = params.joint_map_type.cpu().numpy()
    num_links = len(link_map)
    offsets = params.joint_offset_map.cpu().numpy().reshape(num_links, 2)

    adjoint = np.zeros((num_links, 6))
    for k in range(num_links):
        joint_type = int(joint_map_type[k])
        joint = int(joint_map[k])
        parent = int(link_map[k])
        q_k = 0.0
        if joint_type != FIXED and joint >= 0:
            multiplier, offset = offsets[k]
            q_k = multiplier * q[joint] + offset
            adjoint[k, get_joint_s_index(joint_type)] += multiplier * tau_bar[joint]
        if parent >= 0 and parent != k:
            assert parent < k, "links are expected in base-to-tip order"
            rotation, position = build_joint_transform(fixed_transforms[k], joint_type, q_k)
            to_parent = np.stack(
                [
                    spatial_transform_transpose_multiply(rotation, position, unit)
                    for unit in np.eye(6)
                ],
                axis=1,
            )
            adjoint[k] += to_parent.T @ adjoint[parent]
    return -adjoint


def _reference(params, q, qd, qdd, tau_bar, f_ext=None):
    """Return the float64 NumPy ``tau``, ``grad_q``, ``grad_qd`` and ``grad_qdd`` of a batch."""
    rows = []
    for b in range(q.shape[0]):
        wrench = None if f_ext is None else f_ext[b].astype(np.float64)
        state = [value[b].astype(np.float64) for value in (q, qd, qdd)]
        tau, v, a, f = rnea_from_kinematics_params(*state, params, GRAVITY, f_ext=wrench)
        grads = rnea_backward_from_kinematics_params(
            tau_bar[b].astype(np.float64), *state, v, a, f, params, GRAVITY
        )
        rows.append((tau, *grads))
    return [np.stack([row[i] for row in rows]) for i in range(4)]


def _assert_close(actual, expected, label: str) -> None:
    for name, got, want in zip(("tau", "grad_q", "grad_qd", "grad_qdd"), actual, expected):
        np.testing.assert_allclose(
            got,
            want,
            atol=ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
            err_msg=f"{label}: {name} differs from the NumPy reference",
        )


@pytest.mark.parametrize(("robot_file", "batch_size", "width"), WIDTH_CASES)
def test_each_policy_width_matches_the_reference(
    robot_file: str, batch_size: int, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Torques and gradients at each launch width equal the NumPy reference."""
    monkeypatch.setattr(curobo_runtime, "kernel_launch_sm_count", POLICY_SM_COUNT)
    dynamics, params = _make_dynamics(robot_file, batch_size)
    assert dynamics._reduced_model.threads_per_batch == width

    q, qd, qdd, tau_bar = _inputs(batch_size, params.num_dof)
    actual = _forward_backward(dynamics, q, qd, qdd, tau_bar)
    expected = _reference(params, q, qd, qdd, tau_bar)
    _assert_close(actual, expected, f"{robot_file} batch {batch_size} width {width}")


@pytest.mark.parametrize(("robot_file", "batch_size", "width"), WRENCH_CASES)
def test_each_policy_width_matches_the_reference_with_wrenches(
    robot_file: str, batch_size: int, width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With external wrenches, torques and all four gradients equal the NumPy reference."""
    monkeypatch.setattr(curobo_runtime, "kernel_launch_sm_count", POLICY_SM_COUNT)
    dynamics, params = _make_dynamics(robot_file, batch_size)
    assert dynamics._reduced_model.threads_per_batch == width
    label = f"{robot_file} batch {batch_size} width {width} with wrenches"

    q, qd, qdd, tau_bar = _inputs(batch_size, params.num_dof)
    rng = np.random.RandomState(3000 + batch_size)
    f_ext = rng.randn(batch_size, params.num_links, 6).astype(np.float32)

    actual = _forward_backward(dynamics, q, qd, qdd, tau_bar, f_ext)
    expected = _reference(params, q, qd, qdd, tau_bar, f_ext)
    _assert_close(actual, expected, label)

    expected_wrench_gradient = np.stack(
        [
            _reference_wrench_gradient(
                params, q[b].astype(np.float64), tau_bar[b].astype(np.float64)
            )
            for b in range(batch_size)
        ]
    )
    np.testing.assert_allclose(
        actual[4],
        expected_wrench_gradient,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
        err_msg=f"{label}: grad_f_ext differs from the NumPy reference",
    )

    # The wrenches have to reach the torques, or the comparison above would hold without them.
    without = _forward_backward(dynamics, q, qd, qdd, tau_bar)
    assert np.abs(actual[0] - without[0]).max() > 1.0e-2, label
    assert np.abs(expected_wrench_gradient).max() > 1.0e-3, label
