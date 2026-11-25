# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for RNEA when several batch elements share one warp.

At a cooperative width of 16 two batch elements occupy each 32-lane warp. The
kernels then merge link contributions with a warp-level equality match that is
restricted to the lanes of one batch element. If that restriction is lost, an
element receives the contributions of its warp neighbour. These tests compare
every row of such a launch with the same row computed alone, at a width of 32,
where one element owns the whole warp.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import curobo._src.runtime as curobo_runtime
from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg

#: Multi-level tree robot; the launch policy selects a width of 16 for it.
ROBOT_FILE = "unitree_g1.yml"
#: SM count given to the launch policy, so the width does not depend on the device.
POLICY_SM_COUNT = 16
#: Batch sizes above the policy threshold of 48: an odd count, a count that
#: leaves a partially filled last block, and a count spanning several blocks.
BATCH_SIZES = [49, 65, 129]
#: Both sides run the same float32 kernels and differ only in summation order.
#: A lost lane restriction changes torques by the size of a neighbour's
#: contribution, several orders of magnitude above these bounds.
ABSOLUTE_TOLERANCE = 1.0e-3
RELATIVE_TOLERANCE = 1.0e-3

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_dynamics(batch_size: int) -> Dynamics:
    """Return a dynamics model for :data:`ROBOT_FILE` set up for ``batch_size`` states."""
    device_cfg = DeviceCfg()
    kinematics = KinematicsCfg.from_robot_yaml_file(ROBOT_FILE, device_cfg=device_cfg)
    dynamics = Dynamics(
        DynamicsCfg(
            kinematics_config=kinematics.kinematics_config,
            device_cfg=device_cfg,
            gravity=[0.0, 0.0, -9.81],
        )
    )
    dynamics.setup_batch_size(batch_size=batch_size)
    return dynamics


def _forward_backward(
    dynamics: Dynamics,
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    tau_bar: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run RNEA and its vector-Jacobian product.

    Args:
        dynamics: Model set up for the batch size of the inputs.
        q: Joint positions, shape ``[batch, dof]``, float32.
        qd: Joint velocities, shape ``[batch, dof]``, float32.
        qdd: Joint accelerations, shape ``[batch, dof]``, float32.
        tau_bar: Torque cotangent, shape ``[batch, dof]``, float32.

    Returns:
        ``tau``, ``grad_q``, ``grad_qd`` and ``grad_qdd``, each ``[batch, dof]``.
    """
    device = dynamics.device_cfg.device
    leaves = [
        torch.tensor(value, dtype=torch.float32, device=device, requires_grad=True)
        for value in (q, qd, qdd)
    ]
    tau = dynamics.compute_inverse_dynamics(
        JointState(position=leaves[0], velocity=leaves[1], acceleration=leaves[2])
    )
    tau_np = tau.detach().cpu().numpy().reshape(q.shape).copy()
    tau.backward(torch.tensor(tau_bar, dtype=torch.float32, device=device).reshape(tau.shape))
    return (tau_np, *[leaf.grad.detach().cpu().numpy().copy() for leaf in leaves])


def _inputs(batch_size: int, num_dof: int) -> tuple[np.ndarray, ...]:
    """Return seeded float32 ``q``, ``qd``, ``qdd`` and ``tau_bar`` of shape ``[batch, dof]``."""
    rng = np.random.RandomState(1000 + batch_size)
    return tuple(rng.randn(batch_size, num_dof).astype(np.float32) for _ in range(4))


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_shared_warp_rows_match_single_state_rows(
    batch_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rows computed two per warp equal the same rows computed one per warp."""
    monkeypatch.setattr(curobo_runtime, "kernel_launch_sm_count", POLICY_SM_COUNT)

    batched = _make_dynamics(batch_size)
    single = _make_dynamics(1)
    assert batched._threads_per_batch == 16
    assert single._threads_per_batch == 32

    num_dof = batched.kinematics_config.num_dof
    q, qd, qdd, tau_bar = _inputs(batch_size, num_dof)

    actual = _forward_backward(batched, q, qd, qdd, tau_bar)
    rows = [
        _forward_backward(
            single, q[i : i + 1], qd[i : i + 1], qdd[i : i + 1], tau_bar[i : i + 1]
        )
        for i in range(batch_size)
    ]
    expected = [np.concatenate([row[k] for row in rows], axis=0) for k in range(4)]

    for name, got, want in zip(("tau", "grad_q", "grad_qd", "grad_qdd"), actual, expected):
        np.testing.assert_allclose(
            got,
            want,
            atol=ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
            err_msg=f"{ROBOT_FILE} batch {batch_size}: {name} differs from single-state rows",
        )


def test_neighbour_state_does_not_change_a_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing one state leaves the torques and gradients of every other row unchanged."""
    monkeypatch.setattr(curobo_runtime, "kernel_launch_sm_count", POLICY_SM_COUNT)
    batch_size = 64
    dynamics = _make_dynamics(batch_size)
    assert dynamics._threads_per_batch == 16

    num_dof = dynamics.kinematics_config.num_dof
    q, qd, qdd, tau_bar = _inputs(batch_size, num_dof)
    baseline = _forward_backward(dynamics, q, qd, qdd, tau_bar)

    # Odd rows are the second element of each warp; perturb them only.
    perturbed_rows = np.arange(1, batch_size, 2)
    kept_rows = np.arange(0, batch_size, 2)
    q2, qd2, qdd2 = q.copy(), qd.copy(), qdd.copy()
    q2[perturbed_rows] += 0.5
    qd2[perturbed_rows] *= -2.0
    qdd2[perturbed_rows] += 1.0
    changed = _forward_backward(dynamics, q2, qd2, qdd2, tau_bar)

    assert np.abs(changed[0][perturbed_rows] - baseline[0][perturbed_rows]).max() > 1.0e-2
    for name, got, want in zip(("tau", "grad_q", "grad_qd", "grad_qdd"), changed, baseline):
        np.testing.assert_allclose(
            got[kept_rows],
            want[kept_rows],
            atol=ABSOLUTE_TOLERANCE,
            rtol=RELATIVE_TOLERANCE,
            err_msg=f"{name} of an unperturbed row changed with its warp neighbour",
        )
