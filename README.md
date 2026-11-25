<!-- SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Modified 2026 by Tomas Thoresen: section on the AMD ROCm/HIP (gfx1151) port added. -->
# cuRobo

*CUDA Accelerated Robot Library*

**[Documentation](https://nvlabs.github.io/curobo) | [Paper](https://arxiv.org/abs/2611.06913)**

> [!NOTE]
> cuRoboV2 is a significant rewrite and the public API has changed from cuRobo v1.
> If you depend on the v1 API, pin to the [`v0.7.8`](https://github.com/NVlabs/curobo/tree/v0.7.8) tag.

cuRobo is a CUDA-accelerated library for robot motion generation, built on
PyTorch, CUDA, and Warp. It provides GPU-parallel algorithms for
forward/inverse kinematics, collision checking, trajectory optimization,
geometric planning, GPU-native perception, and whole-body motion generation,
scaling from single-arm manipulators to high-DoF humanoids.

Key capabilities:
- **Dynamics-aware trajectory optimization** with B-spline representation enforcing smoothness and torque limits
- **GPU-native ESDF perception** that generates dense signed distance fields from depth images, up to 10x faster than state-of-the-art
- **Scalable whole-body computation** including topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision for high-DoF robots
- **Collision-free motion generation** combining IK, geometric planning, and trajectory optimization.

## AMD ROCm support (this fork)

This fork runs cuRobo on AMD ROCm (HIP), tested on the AMD Radeon 8060S
(gfx1151). It is not an NVIDIA release and is not endorsed by NVIDIA. Upstream
is [NVlabs/curobo](https://github.com/NVlabs/curobo); branch
`amd-gfx1151-port` is upstream `main` plus the changes described here.

cuRobo compiles its kernels at run time and launches them through NVIDIA's
`cuda.core` package, which targets the CUDA driver. This fork adds
`curobo/_src/curobolib/backends/hip_core`, an implementation of the part of
`cuda.core` that cuRobo uses, on top of hipRTC and the HIP runtime that a ROCm
build of PyTorch loads. The kernel sources and the kernel backend are
upstream's, unchanged.

- `import curobo` registers an import finder after all others. It serves
  `cuda`, `cuda.core`, `cuda.pathfinder`, `cuda.bindings` and
  `cuda.bindings.runtime` only when no `cuda` package is installed and PyTorch
  is a ROCm build. With NVIDIA's packages installed, or with a CUDA build of
  PyTorch, nothing changes.
- A header included ahead of every kernel source gives the warp builtins
  (`__ballot_sync`, `__shfl_sync` and the others) CUDA's 32-bit lane-mask
  semantics on HIP, and keeps the round-to-nearest intrinsics out of fused
  multiply-adds.
- `is_cuda_graph_reset_available()` returns `False` on a PyTorch build without
  CUDA.

### Tested configuration

AMD Ryzen AI Max+ 395 with Radeon 8060S (gfx1151), Ubuntu 24.04.

| Item | Version |
|---|---|
| ROCm | 7.14.1 under `/opt/rocm` |
| Python | 3.12 |
| PyTorch | 2.14.0+rocm7.14 (`download.pytorch.org/whl/rocm7.14`) |
| Warp | [tomasthoresen/warp](https://github.com/tomasthoresen/warp/tree/tomas/halo-v1.17.0), branch `tomas/halo-v1.17.0` |

cuRobo depends on NVIDIA Warp. The `warp-lang` package on PyPI has no ROCm
backend; the Warp branch above is a ROCm/HIP port of Warp 1.17.0 and has to be
installed before cuRobo so that it satisfies the `warp-lang` dependency.

### Use

Install the ROCm build of PyTorch and the Warp branch into a Python
environment, then install this repository with `pip install -e .`. Nothing is
compiled at install time. Warp and hipRTC load ROCm libraries at run time; a
ROCm install without an `ld.so.conf.d` entry needs its library directory on
`LD_LIBRARY_PATH`.

```bash
python -m curobo.examples.getting_started.forward_kinematics --test
python -m curobo.examples.getting_started.inverse_kinematics --test
python -m curobo.examples.getting_started.motion_planning --test
```

Each command exits with status 0 on success.

### Limitations

- `torch.compile` did not finish compiling cuRobo's solvers on the tested
  PyTorch build. Set `curobo.runtime.torch_compile = False` before creating a
  solver, or set `TORCHDYNAMO_DISABLE=1` in the environment.
- The 64-lane warp mode of HIP is not supported; the kernels' lane masks are
  32 bits wide.
- `kernel_lbfgs_step_compact` uses NVIDIA tensor-core intrinsics and does not
  compile for HIP. cuRobo's solvers do not use it.
- `hip_core` checks the number and size of kernel arguments before a launch
  and raises on a mismatch.

License: Apache-2.0, as upstream (`LICENSE`). Robot assets are covered by
`LICENSE_ASSETS`.

## C++ Runtime

Start with [AGENTS.md](AGENTS.md) for the Python-first integration workflow: validate the
robot and motion quality, allocate collision caches, warm up once, and reuse solvers.

The [`cpp_runtime/`](cpp_runtime/README.md) directory contains backend-neutral
C++ APIs for robot kinematics, goalset inverse kinematics, paired-candidate
time-optimal B-spline seed fitting, joint-space trajectory optimization with
timestep refinement, paired-lane bidirectional RRT over IK solutions, composed
joint-space motion planning, single-pose IK-to-RRT-to-TrajOpt planning, goalset
grasp approach/insertion/lift, single-scene collision data, voxel sphere fitting,
grasped-object attachment, and the block-sparse mapper. Python remains the source
of truth for production cuRobo and Warp kernels, while C++-specific export glue
lives under `cpp_runtime/robot/kernels/`. Generated architecture-specific artifacts
are not committed.

## Citation

If you found this work useful, please cite cuRoboV2-Fast,

```
@misc{curobo_v2_fast,
      title={cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning},
      author={Balakumar Sundaralingam and Adithyavairavan Murali and Stan Birchfield},
      year={2026},
      eprint={2611.06913},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.06913}
}
```

## Contributing

Contributions are welcome. Bugs: [open an issue](https://github.com/NVlabs/curobo/issues). General usage questions: [GitHub Discussions](https://github.com/NVlabs/curobo/discussions). For pull requests, please read [`CONTRIBUTING.md`](CONTRIBUTING.md). All commits must include a DCO sign-off (`git commit -s`).

## License

cuRobo is released under the [Apache 2.0 license](LICENSE).

The example robot assets bundled in this repository are provided under their respective licenses. See [LICENSE_ASSETS](LICENSE_ASSETS) for details.

| Asset directory | License | Local license file |
| --- | --- | --- |
| [franka_description](curobo/content/assets/robot/franka_description/) | Apache-2.0 | [LICENSE](curobo/content/assets/robot/franka_description/LICENSE) |
| [g1](curobo/content/assets/robot/g1/) | BSD-3-Clause | [LICENSE](curobo/content/assets/robot/g1/LICENSE) |
| [ur_description](curobo/content/assets/robot/ur_description/) | BSD-3-Clause | [LICENSE](curobo/content/assets/robot/ur_description/LICENSE) |

## Third-Party Software

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

Dependency licenses and attributions are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials.  Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses.  You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.**
