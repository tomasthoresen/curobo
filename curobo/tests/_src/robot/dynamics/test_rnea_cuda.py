# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for CUDA RNEA forward and backward kernels.

Validates the CUDA kernel output against the NumPy reference and Pinocchio.
"""

import numpy as np
import pytest
import torch

from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
    rnea_from_kinematics_params,
)


def _load_robot(robot_yml):
    """Load curobo robot and return KinematicsParams."""
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.types.device_cfg import DeviceCfg

    device_cfg = DeviceCfg()
    cfg = KinematicsCfg.from_robot_yaml_file(robot_yml, device_cfg=device_cfg)
    return cfg.kinematics_config, device_cfg


def _run_cuda_rnea(kp, device_cfg, q_np, qd_np, qdd_np, gravity=-9.81):
    """Run RNEA via the Dynamics class."""
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo._src.state.state_joint import JointState

    cfg = DynamicsCfg(
        kinematics_config=kp,
        device_cfg=device_cfg,
        gravity=[0.0, 0.0, gravity],
    )
    dynamics = Dynamics(cfg)

    batch_size = q_np.shape[0] if q_np.ndim == 2 else 1
    dynamics.setup_batch_size(batch_size=batch_size)

    q_t = torch.tensor(q_np, dtype=torch.float32, device=device_cfg.device)
    qd_t = torch.tensor(qd_np, dtype=torch.float32, device=device_cfg.device)
    qdd_t = torch.tensor(qdd_np, dtype=torch.float32, device=device_cfg.device)

    if q_t.ndim == 1:
        q_t = q_t.unsqueeze(0)
        qd_t = qd_t.unsqueeze(0)
        qdd_t = qdd_t.unsqueeze(0)

    joint_state = JointState(position=q_t, velocity=qd_t, acceleration=qdd_t)
    tau = dynamics.compute_inverse_dynamics(joint_state)
    return tau.detach().cpu().numpy()


def _run_numpy_rnea(kp, q_np, qd_np, qdd_np, gravity=-9.81):
    """Run RNEA via NumPy reference."""
    if q_np.ndim == 1:
        tau, _, _, _ = rnea_from_kinematics_params(q_np, qd_np, qdd_np, kp, gravity)
        return tau
    else:
        taus = []
        for b in range(q_np.shape[0]):
            tau, _, _, _ = rnea_from_kinematics_params(
                q_np[b], qd_np[b], qdd_np[b], kp, gravity
            )
            taus.append(tau)
        return np.stack(taus)


# ---------------------------------------------------------------------------
# CUDA forward vs NumPy reference
# ---------------------------------------------------------------------------


class TestCUDAvsNumPy:
    """Validate CUDA RNEA forward kernel against NumPy reference."""

    ROBOT_CONFIGS = [
        "franka.yml",
        "ur10e.yml",
    ]

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_gravity_compensation(self, robot_yml):
        """Gravity-only torques (qd=0, qdd=0) should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        q = np.zeros(num_dof)
        qd = np.zeros(num_dof)
        qdd = np.zeros(num_dof)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda.flatten(), tau_numpy, atol=1e-4, rtol=1e-4,
            err_msg=f"{robot_yml}: gravity compensation mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_random_single(self, robot_yml):
        """Random single configuration should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(42)
        q = rng.randn(num_dof).astype(np.float32)
        qd = rng.randn(num_dof).astype(np.float32)
        qdd = rng.randn(num_dof).astype(np.float32)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda.flatten(), tau_numpy, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: random single config mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_batch(self, robot_yml):
        """Batch of 16 random configs should match per-element NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(123)
        batch_size = 16
        q = rng.randn(batch_size, num_dof).astype(np.float32)
        qd = rng.randn(batch_size, num_dof).astype(np.float32)
        qdd = rng.randn(batch_size, num_dof).astype(np.float32)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda, tau_numpy, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: batch mismatch",
        )

    def test_large_batch_franka(self):
        """Large batch (1000) stress test."""
        kp, device_cfg = _load_robot("franka.yml")
        num_dof = kp.num_dof

        rng = np.random.RandomState(999)
        batch_size = 1000
        q = rng.randn(batch_size, num_dof).astype(np.float32)
        qd = rng.randn(batch_size, num_dof).astype(np.float32)
        qdd = rng.randn(batch_size, num_dof).astype(np.float32)

        tau_cuda = _run_cuda_rnea(kp, device_cfg, q, qd, qdd)
        tau_numpy = _run_numpy_rnea(kp, q, qd, qdd)

        np.testing.assert_allclose(
            tau_cuda, tau_numpy, atol=1e-3, rtol=1e-3,
            err_msg="franka large batch mismatch",
        )


# ---------------------------------------------------------------------------
# CUDA forward vs Pinocchio (gold standard)
# ---------------------------------------------------------------------------

try:
    import pinocchio as pin

    PINOCCHIO_AVAILABLE = True
except Exception:
    PINOCCHIO_AVAILABLE = False


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not installed")
class TestCUDAvsPinocchio:
    """Validate CUDA RNEA forward against Pinocchio."""

    ROBOT_CONFIGS = [
        "franka.yml",
        "ur10e.yml",
        "dual_ur10e.yml",
        "unitree_g1.yml",
    ]

    @staticmethod
    def _load_pin(robot_yml):
        """Load curobo robot and matching Pinocchio model, return joint permutation."""
        import os

        kp, device_cfg = _load_robot(robot_yml)

        urdf_path = f"/tmp/test_rnea_cuda_{robot_yml.replace('.yml', '')}.urdf"
        kp.export_to_urdf(
            robot_name="dynamics", output_path=urdf_path, include_spheres=False
        )
        pin_model, _, _ = pin.buildModelsFromUrdf(urdf_path)
        pin_data = pin_model.createData()
        if os.path.exists(urdf_path):
            os.remove(urdf_path)

        # Build joint order permutation: perm[curobo_idx] = pin_idx
        curobo_names = list(kp.joint_names)
        pin_names = [pin_model.names[i] for i in range(1, pin_model.njoints)]
        num_dof = len(curobo_names)
        perm = np.zeros(num_dof, dtype=int)
        for ci, cn in enumerate(curobo_names):
            perm[ci] = pin_names.index(cn)

        return kp, device_cfg, pin_model, pin_data, perm

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_gravity(self, robot_yml):
        """Gravity torques match Pinocchio."""
        kp, device_cfg, pin_model, pin_data, perm = self._load_pin(robot_yml)
        num_dof = kp.num_dof
        inv_perm = np.argsort(perm)

        q = np.zeros(num_dof, dtype=np.float64)
        qd = np.zeros(num_dof, dtype=np.float64)
        qdd = np.zeros(num_dof, dtype=np.float64)

        # Pinocchio: send inputs in Pinocchio joint order (inv_perm gathers
        # curobo-ordered vector into Pinocchio order), then map output back
        # to curobo order with perm.
        tau_pin_ordered = pin.rnea(
            pin_model, pin_data, q[inv_perm], qd[inv_perm], qdd[inv_perm]
        )
        tau_pin = np.array(tau_pin_ordered)[perm]

        # CUDA
        tau_cuda = _run_cuda_rnea(
            kp, device_cfg, q.astype(np.float32), qd.astype(np.float32),
            qdd.astype(np.float32)
        )

        np.testing.assert_allclose(
            tau_cuda.flatten(), tau_pin, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: CUDA vs Pinocchio gravity",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_random(self, robot_yml):
        """Random configurations match Pinocchio."""
        kp, device_cfg, pin_model, pin_data, perm = self._load_pin(robot_yml)
        num_dof = kp.num_dof
        inv_perm = np.argsort(perm)

        rng = np.random.RandomState(42)
        for _ in range(5):
            q = rng.randn(num_dof).astype(np.float64)
            qd = rng.randn(num_dof).astype(np.float64)
            qdd = rng.randn(num_dof).astype(np.float64)

            # Pinocchio: curobo→pin reorder with inv_perm, pin→curobo with perm
            tau_pin_ordered = pin.rnea(
                pin_model, pin_data, q[inv_perm], qd[inv_perm], qdd[inv_perm]
            )
            tau_pin = np.array(tau_pin_ordered)[perm]

            # CUDA
            tau_cuda = _run_cuda_rnea(
                kp, device_cfg, q.astype(np.float32), qd.astype(np.float32),
                qdd.astype(np.float32)
            )

            np.testing.assert_allclose(
                tau_cuda.flatten(), tau_pin, atol=1e-3, rtol=1e-3,
                err_msg=f"{robot_yml}: CUDA vs Pinocchio random",
            )


# ---------------------------------------------------------------------------
# Dynamics class interface tests
# ---------------------------------------------------------------------------


class TestDynamicsInterface:
    """Test the Dynamics high-level interface."""

    def test_setup_and_compute(self):
        """Basic setup_batch_size + compute_inverse_dynamics flow."""
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.state.state_joint import JointState

        kp, device_cfg = _load_robot("franka.yml")
        cfg = DynamicsCfg(
            kinematics_config=kp,
            device_cfg=device_cfg,
        )
        dynamics = Dynamics(cfg)
        dynamics.setup_batch_size(batch_size=10)

        num_dof = kp.num_dof
        q = torch.randn(10, num_dof, device=device_cfg.device, dtype=torch.float32)
        qd = torch.randn(10, num_dof, device=device_cfg.device, dtype=torch.float32)
        qdd = torch.randn(10, num_dof, device=device_cfg.device, dtype=torch.float32)

        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau = dynamics.compute_inverse_dynamics(joint_state)

        assert tau.shape == (10, num_dof)
        assert tau.device == q.device
        assert torch.all(torch.isfinite(tau))

    def test_sum_backward(self):
        """tau.sum().backward() should work with PyTorch's expanded sum gradient."""
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.state.state_joint import JointState

        kp, device_cfg = _load_robot("franka.yml")
        cfg = DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg)
        dynamics = Dynamics(cfg)
        dynamics.setup_batch_size(batch_size=4)

        num_dof = kp.num_dof
        q = torch.randn(
            4, num_dof, device=device_cfg.device, dtype=torch.float32, requires_grad=True
        )
        qd = torch.randn(
            4, num_dof, device=device_cfg.device, dtype=torch.float32, requires_grad=True
        )
        qdd = torch.randn(
            4, num_dof, device=device_cfg.device, dtype=torch.float32, requires_grad=True
        )

        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau = dynamics.compute_inverse_dynamics(joint_state)
        tau.sum().backward()

        assert q.grad is not None
        assert qd.grad is not None
        assert qdd.grad is not None
        assert torch.all(torch.isfinite(q.grad))
        assert torch.all(torch.isfinite(qd.grad))
        assert torch.all(torch.isfinite(qdd.grad))

    def test_rebatch(self):
        """Can call setup_batch_size multiple times."""
        from curobo._src.robot.dynamics.dynamics import Dynamics
        from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
        from curobo._src.state.state_joint import JointState

        kp, device_cfg = _load_robot("franka.yml")
        cfg = DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg)
        dynamics = Dynamics(cfg)

        # First setup
        dynamics.setup_batch_size(batch_size=5)
        num_dof = kp.num_dof
        q = torch.zeros(5, num_dof, device=device_cfg.device, dtype=torch.float32)
        qd = torch.zeros(5, num_dof, device=device_cfg.device, dtype=torch.float32)
        qdd = torch.zeros(5, num_dof, device=device_cfg.device, dtype=torch.float32)
        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau1 = dynamics.compute_inverse_dynamics(joint_state)
        assert tau1.shape == (5, num_dof)

        # Second setup with different batch size
        dynamics.setup_batch_size(batch_size=20)
        q2 = torch.zeros(20, num_dof, device=device_cfg.device, dtype=torch.float32)
        qd2 = torch.zeros(20, num_dof, device=device_cfg.device, dtype=torch.float32)
        qdd2 = torch.zeros(20, num_dof, device=device_cfg.device, dtype=torch.float32)
        js2 = JointState(position=q2, velocity=qd2, acceleration=qdd2)
        tau2 = dynamics.compute_inverse_dynamics(js2)
        assert tau2.shape == (20, num_dof)


# ---------------------------------------------------------------------------
# CUDA backward (VJP) vs NumPy backward
# ---------------------------------------------------------------------------



def _run_cuda_backward(kp, device_cfg, q_np, qd_np, qdd_np, tau_bar_np, gravity=-9.81):
    """Run RNEA backward via autograd by calling forward then backward."""
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
    from curobo._src.state.state_joint import JointState

    cfg = DynamicsCfg(
        kinematics_config=kp,
        device_cfg=device_cfg,
        gravity=[0.0, 0.0, gravity],
    )
    dynamics = Dynamics(cfg)

    # Ensure 2D from the start so tensors are leaf nodes with requires_grad
    if q_np.ndim == 1:
        q_2d = q_np[np.newaxis, :]
        qd_2d = qd_np[np.newaxis, :]
        qdd_2d = qdd_np[np.newaxis, :]
        tb_2d = tau_bar_np[np.newaxis, :]
    else:
        q_2d = q_np
        qd_2d = qd_np
        qdd_2d = qdd_np
        tb_2d = tau_bar_np

    batch_size = q_2d.shape[0]
    dynamics.setup_batch_size(batch_size=batch_size)

    q_t = torch.tensor(q_2d, dtype=torch.float32, device=device_cfg.device,
                        requires_grad=True)
    qd_t = torch.tensor(qd_2d, dtype=torch.float32, device=device_cfg.device,
                         requires_grad=True)
    qdd_t = torch.tensor(qdd_2d, dtype=torch.float32, device=device_cfg.device,
                          requires_grad=True)
    tau_bar_t = torch.tensor(tb_2d, dtype=torch.float32, device=device_cfg.device)

    joint_state = JointState(position=q_t, velocity=qd_t, acceleration=qdd_t)
    tau = dynamics.compute_inverse_dynamics(joint_state)
    tau.backward(tau_bar_t)

    return (
        q_t.grad.detach().cpu().numpy(),
        qd_t.grad.detach().cpu().numpy(),
        qdd_t.grad.detach().cpu().numpy(),
    )


def _run_numpy_backward(kp, q_np, qd_np, qdd_np, tau_bar_np, gravity=-9.81):
    """Run RNEA backward via NumPy reference."""
    from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
        rnea_backward_from_kinematics_params,
        rnea_from_kinematics_params,
    )

    if q_np.ndim == 1:
        _, v, a, f = rnea_from_kinematics_params(
            q_np, qd_np, qdd_np, kp, gravity
        )
        gq, gqd, gqdd = rnea_backward_from_kinematics_params(
            tau_bar_np, q_np, qd_np, qdd_np, v, a, f, kp, gravity
        )
        return gq, gqd, gqdd
    else:
        gqs, gqds, gqdds = [], [], []
        for b in range(q_np.shape[0]):
            _, v, a, f = rnea_from_kinematics_params(
                q_np[b], qd_np[b], qdd_np[b], kp, gravity
            )
            gq, gqd, gqdd = rnea_backward_from_kinematics_params(
                tau_bar_np[b], q_np[b], qd_np[b], qdd_np[b],
                v, a, f, kp, gravity
            )
            gqs.append(gq)
            gqds.append(gqd)
            gqdds.append(gqdd)
        return np.stack(gqs), np.stack(gqds), np.stack(gqdds)


class TestCUDABackwardVsNumPy:
    """Validate CUDA RNEA backward (VJP) kernel against NumPy reference."""

    ROBOT_CONFIGS = [
        "franka.yml",
        "ur10e.yml",
        "dual_ur10e.yml",
        "unitree_g1.yml",
    ]

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_grad_q_identity_tau_bar(self, robot_yml):
        """Gradient with tau_bar = ones should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(42)
        q = rng.randn(num_dof).astype(np.float32)
        qd = rng.randn(num_dof).astype(np.float32)
        qdd = rng.randn(num_dof).astype(np.float32)
        tau_bar = np.ones(num_dof, dtype=np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda.flatten(), gq_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_q mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda.flatten(), gqd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qd mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda.flatten(), gqdd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qdd mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_grad_random_tau_bar(self, robot_yml):
        """Gradient with random tau_bar should match NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof

        rng = np.random.RandomState(123)
        q = rng.randn(num_dof).astype(np.float32)
        qd = rng.randn(num_dof).astype(np.float32)
        qdd = rng.randn(num_dof).astype(np.float32)
        tau_bar = rng.randn(num_dof).astype(np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda.flatten(), gq_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_q random tau_bar mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda.flatten(), gqd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qd random tau_bar mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda.flatten(), gqdd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qdd random tau_bar mismatch",
        )

    @pytest.mark.parametrize("robot_yml", ["franka.yml", "ur10e.yml", "unitree_g1.yml"])
    def test_grad_batch(self, robot_yml):
        """Batched backward should match per-element NumPy."""
        kp, device_cfg = _load_robot(robot_yml)
        num_dof = kp.num_dof
        batch_size = 8

        rng = np.random.RandomState(99)
        q = rng.randn(batch_size, num_dof).astype(np.float32)
        qd = rng.randn(batch_size, num_dof).astype(np.float32)
        qdd = rng.randn(batch_size, num_dof).astype(np.float32)
        tau_bar = rng.randn(batch_size, num_dof).astype(np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda, gq_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_q batch mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda, gqd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qd batch mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda, gqdd_np, atol=1e-3, rtol=1e-3,
            err_msg=f"{robot_yml}: grad_qdd batch mismatch",
        )

    def test_grad_zero_input(self):
        """Gradient at q=0, qd=0, qdd=0 should match NumPy (gravity only)."""
        kp, device_cfg = _load_robot("franka.yml")
        num_dof = kp.num_dof

        q = np.zeros(num_dof, dtype=np.float32)
        qd = np.zeros(num_dof, dtype=np.float32)
        qdd = np.zeros(num_dof, dtype=np.float32)
        tau_bar = np.ones(num_dof, dtype=np.float32)

        gq_cuda, gqd_cuda, gqdd_cuda = _run_cuda_backward(
            kp, device_cfg, q, qd, qdd, tau_bar
        )
        gq_np, gqd_np, gqdd_np = _run_numpy_backward(kp, q, qd, qdd, tau_bar)

        np.testing.assert_allclose(
            gq_cuda.flatten(), gq_np, atol=1e-4, rtol=1e-4,
            err_msg="franka: grad_q at zero mismatch",
        )
        np.testing.assert_allclose(
            gqd_cuda.flatten(), gqd_np, atol=1e-4, rtol=1e-4,
            err_msg="franka: grad_qd at zero mismatch",
        )
        np.testing.assert_allclose(
            gqdd_cuda.flatten(), gqdd_np, atol=1e-4, rtol=1e-4,
            err_msg="franka: grad_qdd at zero mismatch",
        )


# ---------------------------------------------------------------------------
# Optimization convergence: fwd → ||τ||² → bwd → update qdd  (no CUDA graph)
# ---------------------------------------------------------------------------


def _create_dynamics(robot_yml, batch_size):
    """Create a Dynamics model for the given robot."""
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg

    kp, device_cfg = _load_robot(robot_yml)

    cfg = DynamicsCfg(kinematics_config=kp, device_cfg=device_cfg)
    dynamics = Dynamics(cfg)
    dynamics.setup_batch_size(batch_size)
    return dynamics, kp, device_cfg


def _run_optimization(dynamics, num_dof, device, batch_size=10, n_steps=1000, lr=0.001):
    """Run gradient descent on qdd to minimize ||τ||².

    The unique minimum is qdd* = -M⁻¹(C·qd + g), i.e. the free-fall acceleration.
    Since M is PD, the problem is convex in qdd and must converge.

    Returns:
        losses: list of loss values per step.
        tau_final: [batch, num_dof] final torques.
        qdd_final: [batch, num_dof] optimized accelerations.
    """
    from curobo._src.state.state_joint import JointState

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_dof, device=device)
    qd = torch.randn(batch_size, num_dof, device=device)
    qdd = torch.zeros(batch_size, num_dof, device=device, requires_grad=True)

    losses = []
    for _step in range(n_steps):
        if qdd.grad is not None:
            qdd.grad.zero_()
        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau = dynamics.compute_inverse_dynamics(joint_state)
        loss = (tau ** 2).sum()
        loss.backward()
        with torch.no_grad():
            grad = qdd.grad.clamp(-1e3, 1e3)
            qdd -= lr * grad
        losses.append(loss.item())

    # Final evaluation (no grad)
    with torch.no_grad():
        joint_state = JointState(position=q, velocity=qd, acceleration=qdd)
        tau_final = dynamics.compute_inverse_dynamics(joint_state)

    return losses, tau_final.detach(), qdd.detach()


class TestOptimizationConvergence:
    """Verify fwd+bwd drives tau→0 by optimizing qdd (gradient descent)."""

    ROBOT_CONFIGS = ["franka.yml", "ur10e.yml"]

    @pytest.mark.parametrize("robot_yml", ROBOT_CONFIGS)
    def test_curobo_converges(self, robot_yml):
        """CuRobo RNEA: loss should drop significantly, proving gradients work."""
        dynamics, kp, device_cfg = _create_dynamics(robot_yml, 10)
        losses, tau_final, _ = _run_optimization(
            dynamics, kp.num_dof, device_cfg.device
        )
        # Just verify gradients reduce the loss significantly
        assert losses[-1] < losses[0] * 0.1, (
            f"{robot_yml}: cuRobo loss didn't reduce "
            f"({losses[-1]:.2f} vs {losses[0]:.2f})"
        )
        assert tau_final.abs().max() < 10.0, (
            f"{robot_yml}: tau too large, max={tau_final.abs().max():.2f}"
        )



if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--tb=short"])


# ---------------------------------------------------------------------------
# Backend parity cases: external wrenches, tree-parallel launches, mimic
# joints and batch shapes. These run unchanged on every kernel backend.
# ---------------------------------------------------------------------------

PARITY_TOL = {"atol": 1e-3, "rtol": 1e-3}
PARITY_ROBOTS = ["franka.yml", "ur10e.yml", "dual_ur10e.yml", "unitree_g1.yml"]
TREE_THREADS = [1, 4, 8, 16, 32]
TREE_BATCH_SIZES = [1, 5, 33]


def _make_dynamics(kp, device_cfg, batch_size, gravity=-9.81, threads_per_batch=None):
    """Create a Dynamics model, optionally forcing the tree-parallel thread count.

    Args:
        kp: KinematicsParams of the robot.
        device_cfg: Device configuration.
        batch_size: Batch size to allocate buffers for.
        gravity: Gravity along z.
        threads_per_batch: Threads per batch element for the kernels (None = the
            model's own choice, 1 for serial chains).

    Returns:
        Dynamics instance ready for compute_inverse_dynamics.
    """
    from curobo._src.robot.dynamics.dynamics import Dynamics
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg

    cfg = DynamicsCfg(
        kinematics_config=kp,
        device_cfg=device_cfg,
        gravity=[0.0, 0.0, gravity],
    )
    dynamics = Dynamics(cfg)
    dynamics.setup_batch_size(batch_size=batch_size)
    if threads_per_batch is not None:
        dynamics._threads_per_batch = threads_per_batch
    return dynamics


def _random_inputs(num_dof, batch_size, seed):
    """Seeded joint positions, velocities, accelerations and upstream gradient."""
    rng = np.random.RandomState(seed)
    q = rng.uniform(-1.5, 1.5, size=(batch_size, num_dof))
    qd = rng.randn(batch_size, num_dof)
    qdd = rng.randn(batch_size, num_dof)
    tau_bar = rng.randn(batch_size, num_dof)
    return q, qd, qdd, tau_bar


def _run_cuda_fwd_bwd(dynamics, device, q_np, qd_np, qdd_np, tau_bar_np, f_ext_np=None):
    """Forward and VJP on the device.

    Args:
        dynamics: Dynamics model (buffers already allocated).
        device: Torch device.
        q_np, qd_np, qdd_np: [batch, num_dof] joint state.
        tau_bar_np: [batch, num_dof] upstream gradient dL/dtau.
        f_ext_np: [batch, num_links, 6] external wrenches, or None.

    Returns:
        tau, grad_q, grad_qd, grad_qdd, grad_f_ext (None without f_ext), as NumPy copies.
    """
    from curobo._src.state.state_joint import JointState

    def _leaf(x):
        return torch.tensor(x, dtype=torch.float32, device=device, requires_grad=True)

    q_t, qd_t, qdd_t = _leaf(q_np), _leaf(qd_np), _leaf(qdd_np)
    f_ext_t = _leaf(f_ext_np) if f_ext_np is not None else None
    joint_state = JointState(position=q_t, velocity=qd_t, acceleration=qdd_t)
    tau = dynamics.compute_inverse_dynamics(joint_state, f_ext=f_ext_t)
    tau_np = tau.detach().cpu().numpy().copy()
    tau.backward(torch.tensor(tau_bar_np, dtype=torch.float32, device=device))
    grads = [t.grad.detach().cpu().numpy().copy() for t in (q_t, qd_t, qdd_t)]
    grad_f_ext = f_ext_t.grad.detach().cpu().numpy().copy() if f_ext_t is not None else None
    return (tau_np, *grads, grad_f_ext)


def _numpy_rnea_with_f_ext(kp, q, qd, qdd, f_ext, gravity=-9.81):
    """NumPy RNEA with external wrenches, built on the reference primitives.

    tau is linear in f_ext and v, a do not depend on it, so the wrench
    contribution is the tip-to-base propagation of -f_ext on its own:
    f[k] = I*a + v x* (I*v) - f_ext[k] before propagation.

    Args:
        kp: KinematicsParams.
        q, qd, qdd: [num_dof] joint state (one sample).
        f_ext: [num_links, 6] wrenches [torque(3), force(3)] in each link frame.
        gravity: Gravity along z.

    Returns:
        tau [num_dof], and the f_ext-aware v, a, f [num_links, 6] for the adjoint.
    """
    from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
        FIXED,
        build_joint_transform,
        get_joint_s_index,
        spatial_transform_transpose_multiply,
    )

    tau, v, a, f = rnea_from_kinematics_params(q, qd, qdd, kp, gravity)
    fixed_transforms = kp.fixed_transforms.cpu().numpy()
    link_map = kp.link_map.cpu().numpy()
    joint_map = kp.joint_map.cpu().numpy()
    joint_map_type = kp.joint_map_type.cpu().numpy()
    num_links = len(link_map)
    offsets = kp.joint_offset_map.cpu().numpy().reshape(num_links, 2)

    f_x = -np.asarray(f_ext, dtype=np.float64).copy()
    tau_x = np.zeros_like(tau)
    for k in range(num_links - 1, -1, -1):
        jtype = int(joint_map_type[k])
        jidx = int(joint_map[k])
        parent = int(link_map[k])
        q_k = 0.0
        if jtype != FIXED and jidx >= 0:
            multiplier, offset = offsets[k]
            q_k = multiplier * q[jidx] + offset
            tau_x[jidx] += multiplier * f_x[k, get_joint_s_index(jtype)]
        if parent >= 0 and parent != k:
            R, p = build_joint_transform(fixed_transforms[k], jtype, q_k)
            f_x[parent] += spatial_transform_transpose_multiply(R, p, f_x[k])
    return tau + tau_x, v, a, f + f_x


def _numpy_fwd_bwd(kp, q_np, qd_np, qdd_np, tau_bar_np, f_ext_np=None, gravity=-9.81):
    """NumPy reference forward and adjoint for a batch (optionally with wrenches)."""
    from curobo.tests._src.robot.dynamics.rnea_numpy_reference import (
        rnea_backward_from_kinematics_params,
    )

    taus, gqs, gqds, gqdds = [], [], [], []
    for b in range(q_np.shape[0]):
        if f_ext_np is None:
            tau, v, a, f = rnea_from_kinematics_params(
                q_np[b], qd_np[b], qdd_np[b], kp, gravity
            )
        else:
            tau, v, a, f = _numpy_rnea_with_f_ext(
                kp, q_np[b], qd_np[b], qdd_np[b], f_ext_np[b], gravity
            )
        gq, gqd, gqdd = rnea_backward_from_kinematics_params(
            tau_bar_np[b], q_np[b], qd_np[b], qdd_np[b], v, a, f, kp, gravity
        )
        taus.append(tau)
        gqs.append(gq)
        gqds.append(gqd)
        gqdds.append(gqdd)
    return np.stack(taus), np.stack(gqs), np.stack(gqds), np.stack(gqdds)


def _assert_fwd_bwd_close(cuda_out, ref_out, label):
    """Compare (tau, grad_q, grad_qd, grad_qdd) with the parity tolerance."""
    names = ["tau", "grad_q", "grad_qd", "grad_qdd"]
    for name, got, want in zip(names, cuda_out[:4], ref_out[:4]):
        np.testing.assert_allclose(got, want, err_msg=f"{label}: {name}", **PARITY_TOL)


class TestExternalForces:
    """External wrenches: forward torques and all four gradients."""

    @pytest.mark.parametrize("robot_yml", PARITY_ROBOTS)
    def test_forward_backward_vs_numpy(self, robot_yml):
        """tau(f_ext) and grad_q/qd/qdd under f_ext match the NumPy reference."""
        kp, device_cfg = _load_robot(robot_yml)
        batch_size = 4
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=7)
        rng = np.random.RandomState(11)
        f_ext = rng.randn(batch_size, kp.num_links, 6)

        dynamics = _make_dynamics(kp, device_cfg, batch_size)
        cuda_out = _run_cuda_fwd_bwd(dynamics, device_cfg.device, q, qd, qdd, tau_bar, f_ext)
        ref_out = _numpy_fwd_bwd(kp, q, qd, qdd, tau_bar, f_ext)
        _assert_fwd_bwd_close(cuda_out, ref_out, f"{robot_yml} f_ext")

        # The wrench term must change the torques (guards against f_ext being ignored).
        tau_no_ext = _run_cuda_fwd_bwd(dynamics, device_cfg.device, q, qd, qdd, tau_bar)[0]
        assert np.abs(cuda_out[0] - tau_no_ext).max() > 1e-2

    @pytest.mark.parametrize("robot_yml", PARITY_ROBOTS)
    def test_grad_f_ext_vs_finite_differences(self, robot_yml):
        """grad_f_ext equals the finite-difference VJP (tau is linear in f_ext)."""
        kp, device_cfg = _load_robot(robot_yml)
        num_links = kp.num_links
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, 1, seed=3)
        rng = np.random.RandomState(5)
        f_ext = rng.randn(1, num_links, 6)

        dynamics = _make_dynamics(kp, device_cfg, 1)
        grad_f_ext = _run_cuda_fwd_bwd(
            dynamics, device_cfg.device, q, qd, qdd, tau_bar, f_ext
        )[4]

        # Central differences over every wrench component, in one batched forward call.
        step = 1.0
        n_pert = num_links * 6
        f_batch = np.repeat(f_ext, 2 * n_pert, axis=0)
        for idx in range(n_pert):
            k, i = divmod(idx, 6)
            f_batch[2 * idx, k, i] += step
            f_batch[2 * idx + 1, k, i] -= step
        dyn_fd = _make_dynamics(kp, device_cfg, 2 * n_pert)
        tau_fd = _run_cuda_fwd_bwd(
            dyn_fd,
            device_cfg.device,
            np.repeat(q, 2 * n_pert, axis=0),
            np.repeat(qd, 2 * n_pert, axis=0),
            np.repeat(qdd, 2 * n_pert, axis=0),
            np.repeat(tau_bar, 2 * n_pert, axis=0),
            f_batch,
        )[0]
        dtau = (tau_fd[0::2] - tau_fd[1::2]) / (2.0 * step)  # [n_pert, num_dof]
        grad_fd = (dtau @ tau_bar[0]).reshape(num_links, 6)

        np.testing.assert_allclose(
            grad_f_ext[0], grad_fd, err_msg=f"{robot_yml}: grad_f_ext vs FD", **PARITY_TOL
        )
        assert np.abs(grad_f_ext).max() > 1e-3


class TestTreeParallel:
    """Tree-parallel launches (threads_per_batch > 1) against serial and NumPy."""

    @pytest.mark.parametrize("batch_size", TREE_BATCH_SIZES)
    @pytest.mark.parametrize("threads_per_batch", TREE_THREADS)
    @pytest.mark.parametrize("robot_yml", ["franka.yml", "dual_ur10e.yml", "unitree_g1.yml"])
    def test_forward_backward(self, robot_yml, threads_per_batch, batch_size):
        """Every thread count gives the NumPy reference and the serial result."""
        kp, device_cfg = _load_robot(robot_yml)
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=batch_size)
        label = f"{robot_yml} tpb={threads_per_batch} batch={batch_size}"

        dyn_tree = _make_dynamics(kp, device_cfg, batch_size, threads_per_batch=threads_per_batch)
        tree_out = _run_cuda_fwd_bwd(dyn_tree, device_cfg.device, q, qd, qdd, tau_bar)
        ref_out = _numpy_fwd_bwd(kp, q, qd, qdd, tau_bar)
        _assert_fwd_bwd_close(tree_out, ref_out, label + " vs numpy")

        dyn_serial = _make_dynamics(kp, device_cfg, batch_size, threads_per_batch=1)
        serial_out = _run_cuda_fwd_bwd(dyn_serial, device_cfg.device, q, qd, qdd, tau_bar)
        _assert_fwd_bwd_close(tree_out, serial_out, label + " vs serial")

    @pytest.mark.parametrize("threads_per_batch", [4, 8, 16, 32])
    def test_external_forces(self, threads_per_batch):
        """Wrench gradients also hold on the tree-parallel path (branched robot)."""
        kp, device_cfg = _load_robot("dual_ur10e.yml")
        batch_size = 5
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=21)
        rng = np.random.RandomState(22)
        f_ext = rng.randn(batch_size, kp.num_links, 6)

        dyn_tree = _make_dynamics(kp, device_cfg, batch_size, threads_per_batch=threads_per_batch)
        tree_out = _run_cuda_fwd_bwd(dyn_tree, device_cfg.device, q, qd, qdd, tau_bar, f_ext)
        dyn_serial = _make_dynamics(kp, device_cfg, batch_size, threads_per_batch=1)
        serial_out = _run_cuda_fwd_bwd(dyn_serial, device_cfg.device, q, qd, qdd, tau_bar, f_ext)

        label = f"dual_ur10e tpb={threads_per_batch} f_ext"
        _assert_fwd_bwd_close(tree_out, serial_out, label)
        np.testing.assert_allclose(
            tree_out[4], serial_out[4], err_msg=label + ": grad_f_ext", **PARITY_TOL
        )
        ref_out = _numpy_fwd_bwd(kp, q, qd, qdd, tau_bar, f_ext)
        _assert_fwd_bwd_close(tree_out, ref_out, label + " vs numpy")


class TestMimicJoints:
    """Mimic joints (multiplier and offset) on a chain with real inertias."""

    @staticmethod
    def _mimic_params(robot_yml="ur10e.yml", multiplier=-1.5, offset=0.5):
        """Make the link driven by joint 2 mimic joint 1 on a clone of the robot's params."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import FIXED

        kp, device_cfg = _load_robot(robot_yml)
        kp = kp.clone()
        joint_map = kp.joint_map.cpu().numpy()
        joint_map_type = kp.joint_map_type.cpu().numpy()
        movable = [k for k in range(kp.num_links) if joint_map_type[k] != FIXED and joint_map[k] >= 0]
        k_src = next(k for k in movable if joint_map[k] == 1)
        k_mimic = next(k for k in movable if joint_map[k] == 2)
        kp.joint_map[k_mimic] = int(joint_map[k_src])
        kp.joint_offset_map[2 * k_mimic] = multiplier
        kp.joint_offset_map[2 * k_mimic + 1] = offset
        return kp, device_cfg

    def test_forward_backward_vs_numpy(self):
        """Torques and gradients of a mimic chain match the NumPy reference."""
        kp, device_cfg = self._mimic_params()
        batch_size = 6
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=31)
        dynamics = _make_dynamics(kp, device_cfg, batch_size)
        cuda_out = _run_cuda_fwd_bwd(dynamics, device_cfg.device, q, qd, qdd, tau_bar)
        ref_out = _numpy_fwd_bwd(kp, q, qd, qdd, tau_bar)
        _assert_fwd_bwd_close(cuda_out, ref_out, "ur10e mimic")

        # Joint 2 no longer drives a link: its torque and gradients are zero,
        # and the mimic must leave a trace in the torques of joint 1.
        for arr in cuda_out[:4]:
            assert np.abs(arr[:, 2]).max() == 0.0
        plain_kp, _ = _load_robot("ur10e.yml")
        plain = _make_dynamics(plain_kp, device_cfg, batch_size)
        plain_out = _run_cuda_fwd_bwd(plain, device_cfg.device, q, qd, qdd, tau_bar)
        assert np.abs(cuda_out[0][:, 1] - plain_out[0][:, 1]).max() > 1e-2


class TestBatchShapes:
    """Large and warp-unaligned batches, forward and backward."""

    def test_large_batch_forward_backward(self):
        """Batch 1000 on Franka: a sample of rows matches the NumPy reference."""
        kp, device_cfg = _load_robot("franka.yml")
        batch_size = 1000
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=41)
        dynamics = _make_dynamics(kp, device_cfg, batch_size)
        cuda_out = _run_cuda_fwd_bwd(dynamics, device_cfg.device, q, qd, qdd, tau_bar)
        assert all(np.isfinite(arr).all() for arr in cuda_out[:4])

        rows = np.array([0, 1, 31, 32, 33, 511, 512, 998, 999])
        ref_out = _numpy_fwd_bwd(kp, q[rows], qd[rows], qdd[rows], tau_bar[rows])
        _assert_fwd_bwd_close([arr[rows] for arr in cuda_out[:4]], ref_out, "franka batch 1000")

    @pytest.mark.parametrize("batch_size", [1, 5, 33, 129])
    def test_unaligned_batches(self, batch_size):
        """Batch sizes that do not fill a warp (UR10e, serial path)."""
        kp, device_cfg = _load_robot("ur10e.yml")
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=50 + batch_size)
        dynamics = _make_dynamics(kp, device_cfg, batch_size)
        cuda_out = _run_cuda_fwd_bwd(dynamics, device_cfg.device, q, qd, qdd, tau_bar)
        ref_out = _numpy_fwd_bwd(kp, q, qd, qdd, tau_bar)
        _assert_fwd_bwd_close(cuda_out, ref_out, f"ur10e batch {batch_size}")

    @pytest.mark.parametrize("robot_yml,batch_size", [("franka.yml", 32), ("unitree_g1.yml", 8)])
    def test_partial_last_block(self, robot_yml, batch_size):
        """A last block with fewer active batch elements than links (serial path).

        The backward kernel loads the link inertias into block-shared memory
        cooperatively; every row of a partially filled last block must still
        see all of them.
        """
        kp, device_cfg = _load_robot(robot_yml)
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=50 + batch_size)
        dynamics = _make_dynamics(kp, device_cfg, batch_size, threads_per_batch=1)
        cuda_out = _run_cuda_fwd_bwd(dynamics, device_cfg.device, q, qd, qdd, tau_bar)
        ref_out = _numpy_fwd_bwd(kp, q, qd, qdd, tau_bar)
        _assert_fwd_bwd_close(
            cuda_out, ref_out, f"{robot_yml} batch {batch_size} serial, partial last block"
        )


@pytest.mark.skipif(not PINOCCHIO_AVAILABLE, reason="Pinocchio not installed")
class TestGradientsVsPinocchio:
    """Gradients against Pinocchio's analytical RNEA derivatives (independent route)."""

    @staticmethod
    def _pin_force_vector(pin_model, kp, perm, f_ext_b):
        """cuRobo wrenches [torque, force] per link -> Pinocchio forces per joint.

        Wrenches on fixed links are not representable per joint in Pinocchio and
        must be zero in f_ext_b.
        """
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import FIXED

        joint_map = kp.joint_map.cpu().numpy()
        joint_map_type = kp.joint_map_type.cpu().numpy()
        fext = pin.StdVec_Force()
        for _ in range(pin_model.njoints):
            fext.append(pin.Force.Zero())
        for k in range(kp.num_links):
            if joint_map_type[k] == FIXED or joint_map[k] < 0:
                assert np.all(f_ext_b[k] == 0.0)
                continue
            pin_joint = int(perm[int(joint_map[k])]) + 1
            fext[pin_joint] = pin.Force(f_ext_b[k, 3:6], f_ext_b[k, 0:3])
        return fext

    @pytest.mark.parametrize("with_f_ext", [False, True])
    @pytest.mark.parametrize("robot_yml", PARITY_ROBOTS)
    def test_grads(self, robot_yml, with_f_ext):
        """grad_q/qd/qdd = (dtau/dq, dtau/dv, dtau/da)^T tau_bar from Pinocchio."""
        from curobo.tests._src.robot.dynamics.rnea_numpy_reference import FIXED

        kp, device_cfg, pin_model, pin_data, perm = TestCUDAvsPinocchio._load_pin(robot_yml)
        inv_perm = np.argsort(perm)
        batch_size = 3
        q, qd, qdd, tau_bar = _random_inputs(kp.num_dof, batch_size, seed=61)
        f_ext = None
        if with_f_ext:
            rng = np.random.RandomState(62)
            f_ext = rng.randn(batch_size, kp.num_links, 6)
            joint_map = kp.joint_map.cpu().numpy()
            joint_map_type = kp.joint_map_type.cpu().numpy()
            for k in range(kp.num_links):
                if joint_map_type[k] == FIXED or joint_map[k] < 0:
                    f_ext[:, k, :] = 0.0

        dynamics = _make_dynamics(kp, device_cfg, batch_size)
        cuda_out = _run_cuda_fwd_bwd(dynamics, device_cfg.device, q, qd, qdd, tau_bar, f_ext)

        for b in range(batch_size):
            args = [pin_model, pin_data, q[b][inv_perm], qd[b][inv_perm], qdd[b][inv_perm]]
            if with_f_ext:
                args.append(self._pin_force_vector(pin_model, kp, perm, f_ext[b]))
            tau_pin = np.array(pin.rnea(*args))[perm]
            dtau_dq, dtau_dv, dtau_da = pin.computeRNEADerivatives(*args)
            tau_bar_pin = tau_bar[b][inv_perm]
            label = f"{robot_yml} f_ext={with_f_ext} b={b}"
            np.testing.assert_allclose(cuda_out[0][b], tau_pin, err_msg=label + ": tau", **PARITY_TOL)
            for name, got, mat in zip(
                ["grad_q", "grad_qd", "grad_qdd"], cuda_out[1:4], (dtau_dq, dtau_dv, dtau_da)
            ):
                want = (np.asarray(mat).T @ tau_bar_pin)[perm]
                np.testing.assert_allclose(got[b], want, err_msg=f"{label}: {name}", **PARITY_TOL)


# ---------------------------------------------------------------------------
# Compiled (pybind) backend: launch geometry parity with the cuda.core helper
# ---------------------------------------------------------------------------

try:
    from curobo._src.curobolib.backends.pybind import dynamics as _compiled_dynamics

    COMPILED_DYNAMICS_AVAILABLE = hasattr(_compiled_dynamics, "rnea_launch_config")
except ImportError:
    COMPILED_DYNAMICS_AVAILABLE = False


@pytest.mark.skipif(not COMPILED_DYNAMICS_AVAILABLE, reason="compiled dynamics extension not built")
class TestCompiledLaunchConfig:
    """The compiled launcher lays out blocks exactly like DynamicsLaunchCfg."""

    LINK_COUNTS = [10, 13, 22, 56, 120]
    BATCH_SIZES = [1, 7, 31, 32, 33, 128, 1000, 4096]

    @pytest.mark.parametrize("backward", [False, True])
    @pytest.mark.parametrize("threads_per_batch", TREE_THREADS)
    def test_matches_cuda_core_helper(self, threads_per_batch, backward):
        """Grid, block and shared-memory size agree for every robot size and batch."""
        from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import (
            DynamicsLaunchCfg,
        )

        calculate = (
            DynamicsLaunchCfg.calculate_backward_geometry
            if backward
            else DynamicsLaunchCfg.calculate_forward_geometry
        )
        for num_links in self.LINK_COUNTS:
            for batch_size in self.BATCH_SIZES:
                expected = tuple(int(v) for v in calculate(batch_size, num_links, threads_per_batch))
                got = tuple(
                    int(v)
                    for v in _compiled_dynamics.rnea_launch_config(
                        batch_size, num_links, threads_per_batch, backward
                    )
                )
                assert got == expected, (
                    f"links={num_links} batch={batch_size} tpb={threads_per_batch} "
                    f"backward={backward}: compiled {got} vs cuda.core {expected}"
                )

    def test_rejects_oversized_robot(self):
        """A robot whose single batch element exceeds the shared-memory cap is refused."""
        with pytest.raises(RuntimeError):
            _compiled_dynamics.rnea_launch_config(1, 2000, 1, False)
