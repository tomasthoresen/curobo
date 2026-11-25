# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for the HIP stand-in that serves ``cuda.core`` on a ROCm build of PyTorch."""

# Standard Library
import ctypes
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

# Third Party
import numpy as np
import pytest
import torch

from curobo._src.curobolib.backends import hip_core
from curobo._src.curobolib.backends.hip_core import _core, _hip, _pathfinder, _program

_REPOSITORY_ROOT = Path(__file__).parents[4]

requires_hip = pytest.mark.skipif(
    not (getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()),
    reason="needs a ROCm build of PyTorch with a device",
)

_NVRTC_TYPEDEFS = """
// Define standard integer types for NVRTC
typedef signed char        int8_t;
typedef short              int16_t;
typedef int                int32_t;
typedef long long          int64_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", dedent(script)],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )


def test_fixed_width_typedefs_are_removed_and_others_kept() -> None:
    source = (
        _NVRTC_TYPEDEFS
        + "typedef float real_t;\nstruct A { int32_t x; };\n"
        + "typedef struct Foo int32_t;\ntypedef unsigned short uint16_t;  // comment\n"
    )
    stripped = _program._FIXED_WIDTH_TYPEDEF.sub("", source)
    assert "typedef struct Foo int32_t;" in stripped
    assert "uint16_t;" not in stripped
    assert stripped.count("\n") == source.count("\n")
    assert "int64_t;" not in stripped
    assert "uint8_t;" not in stripped
    assert "typedef float real_t;" in stripped
    assert "struct A { int32_t x; };" in stripped


def test_launch_config_normalizes_dimensions() -> None:
    config = _core.LaunchConfig(grid=5, block=(32, 2), shmem_size=None)
    assert config.grid == (5, 1, 1)
    assert config.block == (32, 2, 1)
    assert config.shmem_size == 0
    with pytest.raises(ValueError):
        _core.LaunchConfig(grid=0, block=32)
    for grid in (1.5, True, 2**32 + 1, (4, 0)):
        with pytest.raises(ValueError):
            _core.LaunchConfig(grid=grid, block=32)
    for shmem_size in (-1, 2**31, 1.5, "8", True):
        with pytest.raises(ValueError):
            _core.LaunchConfig(grid=1, block=32, shmem_size=shmem_size)
    assert _core.LaunchConfig(grid=1, block=32, shmem_size=np.int64(64)).shmem_size == 64


def test_pathfinder_serves_runtime_headers_only() -> None:
    directory = _pathfinder.find_nvidia_header_directory("nvrtc")
    assert directory == _pathfinder.find_nvidia_header_directory("cudart")
    for header in ("nvrtc.h", "cuda_runtime.h", "cuda_fp16.h", "hip_compat.h"):
        assert (Path(directory) / header).is_file()
    assert _pathfinder.find_nvidia_header_directory("nvcc") is None


def test_an_installed_cuda_package_takes_precedence() -> None:
    """The finder is consulted last, so a real ``cuda`` package is never shadowed."""
    result = _run(
        """
        import importlib.abc, importlib.machinery, sys, types

        class RealCuda(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in ("cuda", "cuda.core"):
                    return importlib.machinery.ModuleSpec(
                        fullname, self, is_package=fullname == "cuda"
                    )
                return None
            def create_module(self, spec):
                module = types.ModuleType(spec.name)
                module.marker = "real"
                return module
            def exec_module(self, module):
                pass

        sys.meta_path.insert(0, RealCuda())
        import curobo
        import cuda.core
        from curobo._src.curobolib.backends import hip_core
        assert cuda.core.marker == "real"
        assert not hip_core.is_active()
        """
    )
    assert result.returncode == 0, result.stderr


def test_an_installed_cuda_package_is_not_completed_from_hip_core() -> None:
    """A ``cuda`` package without ``cuda.core`` stays without it."""
    result = _run(
        """
        import importlib.abc, importlib.machinery, sys, types

        class RealCuda(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cuda":
                    return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
                return None
            def create_module(self, spec):
                return types.ModuleType(spec.name)
            def exec_module(self, module):
                pass

        sys.meta_path.insert(0, RealCuda())
        import curobo
        for name in ("cuda.core", "cuda.pathfinder", "cuda.bindings.runtime"):
            try:
                __import__(name)
            except ImportError:
                continue
            raise AssertionError(f"{name} resolved to {sys.modules[name]!r}")
        """
    )
    assert result.returncode == 0, result.stderr


def test_finder_declines_when_torch_reports_no_hip_version() -> None:
    result = _run(
        """
        import torch
        torch.version.hip = None
        import curobo
        try:
            import cuda.core
        except ImportError:
            pass
        else:
            raise AssertionError("cuda.core resolved without a ROCm build of PyTorch")
        """
    )
    assert result.returncode == 0, result.stderr


def test_finder_declines_when_torch_cannot_be_loaded(tmp_path) -> None:
    (tmp_path / "torch").mkdir()
    (tmp_path / "torch" / "__init__.py").write_text('raise OSError("library not found")\n')
    script = dedent(
        """
        import curobo
        for name in ("cuda", "cuda.core", "cuda.pathfinder", "cuda.bindings.runtime"):
            try:
                __import__(name)
            except ImportError:
                continue
            raise AssertionError(f"{name} did not raise ImportError")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{_REPOSITORY_ROOT}"},
    )
    assert result.returncode == 0, result.stderr


def test_finder_declines_without_a_rocm_torch() -> None:
    result = _run(
        """
        import curobo
        from curobo._src.curobolib.backends import hip_core
        hip_core.torch_is_rocm = lambda: False
        try:
            import cuda.core
        except ImportError:
            pass
        else:
            import cuda
            raise AssertionError(f"cuda resolved to {cuda!r}")
        """
    )
    assert result.returncode == 0, result.stderr


_SERVED_NAMES = {
    "cuda": "_cuda",
    "cuda.core": "_core",
    "cuda.pathfinder": "_pathfinder",
    "cuda.bindings": "_bindings",
    "cuda.bindings.runtime": "_runtime",
}


def _run_with_stub_torch(tmp_path, torch_source: str, code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a new interpreter where ``torch`` is a one-file stand-in."""
    (tmp_path / "torch").mkdir(exist_ok=True)
    (tmp_path / "torch" / "__init__.py").write_text(dedent(torch_source))
    return subprocess.run(
        [sys.executable, "-c", dedent(code)],
        cwd=_REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": f"{tmp_path}{os.pathsep}{_REPOSITORY_ROOT}"},
    )


_ROCM_TORCH = """
import types
version = types.SimpleNamespace(hip="7.14")
"""


@pytest.mark.parametrize("name", sorted(_SERVED_NAMES))
def test_each_cuda_name_resolves_to_hip_core(tmp_path, name: str) -> None:
    """Every served name is importable by its full name as the first ``cuda`` import."""
    expected = "curobo._src.curobolib.backends.hip_core." + _SERVED_NAMES[name]
    result = _run_with_stub_torch(
        tmp_path,
        _ROCM_TORCH,
        f"""
        import importlib
        import curobo
        module = importlib.import_module({name!r})
        assert module.__name__ == {expected!r}, module.__name__
        """,
    )
    assert result.returncode == 0, result.stderr


def test_served_names_are_exactly_the_documented_five() -> None:
    assert dict(hip_core._PROVIDED) == {
        name: "curobo._src.curobolib.backends.hip_core." + target
        for name, target in _SERVED_NAMES.items()
    }


@pytest.mark.parametrize(
    "torch_source",
    [
        "import types\nversion = types.SimpleNamespace()\n",
        "pass\n",
    ],
    ids=["version-without-hip", "no-version"],
)
def test_finder_declines_when_torch_has_no_hip_attribute(tmp_path, torch_source: str) -> None:
    result = _run_with_stub_torch(
        tmp_path,
        torch_source,
        """
        import curobo
        for name in ("cuda", "cuda.core", "cuda.pathfinder", "cuda.bindings",
                     "cuda.bindings.runtime"):
            try:
                __import__(name)
            except ImportError:
                continue
            raise AssertionError(f"{name} did not raise ImportError")
        """,
    )
    assert result.returncode == 0, result.stderr


def test_install_registers_the_finder_once_and_last() -> None:
    result = _run(
        """
        import sys
        import curobo
        from curobo._src.curobolib.backends import hip_core
        hip_core.install()
        hip_core.install()
        finders = [f for f in sys.meta_path if isinstance(f, hip_core._HipCoreFinder)]
        assert len(finders) == 1, finders
        assert sys.meta_path[-1] is finders[0], sys.meta_path
        """
    )
    assert result.returncode == 0, result.stderr


def test_an_installed_cuda_package_is_resolved_without_importing_torch() -> None:
    result = _run(
        """
        import importlib.abc, importlib.machinery, sys, types

        class RealCuda(importlib.abc.MetaPathFinder, importlib.abc.Loader):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cuda":
                    return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
                return None
            def create_module(self, spec):
                return types.ModuleType(spec.name)
            def exec_module(self, module):
                pass

        sys.meta_path.insert(0, RealCuda())
        import curobo
        for name in ("cuda.core", "cuda.pathfinder", "cuda.bindings", "cuda.bindings.runtime"):
            try:
                __import__(name)
            except ImportError:
                continue
            raise AssertionError(f"{name} resolved")
        assert "torch" not in sys.modules
        """
    )
    assert result.returncode == 0, result.stderr


@requires_hip
def test_cuda_names_resolve_to_hip_core() -> None:
    result = _run(
        """
        import curobo
        import cuda.bindings.runtime as cudart
        import cuda.pathfinder
        from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch
        from curobo._src.curobolib.backends import hip_core
        assert hip_core.is_active()
        assert cudart.cudaError_t.cudaSuccess == 0
        assert cuda.pathfinder.find_nvidia_header_directory("nvrtc") is not None
        """
    )
    assert result.returncode == 0, result.stderr


_MASK_KERNEL = """
extern "C" __global__ void mask_probe(unsigned *ballot, unsigned *match, int *all, int *any,
                                      int group, int scale) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned group_mask = ((group >= 32) ? 0xffffffffu : ((1u << group) - 1u))
                              << ((lane / group) * group);
  const int key = static_cast<int>(lane % 4u);
  ballot[threadIdx.x] = __ballot_sync(group_mask, lane % 2u == 0u);
  match[threadIdx.x] = __match_any_sync(group_mask, key * scale);
  all[threadIdx.x] = __all_sync(group_mask, lane != 5u);
  any[threadIdx.x] = __any_sync(group_mask, lane == 5u);
  __syncwarp(group_mask);
}
"""


def _compile(source: str, name: str):
    from cuda.core import Program, ProgramOptions

    program = Program(_NVRTC_TYPEDEFS + source, code_type="c++", options=ProgramOptions())
    return program.compile("cubin", name_expressions=(name,)).get_kernel(name)


@requires_hip
@pytest.mark.parametrize("group", [4, 16, 32])
def test_partial_masks_restrict_ballot_match_all_any(group: int) -> None:
    """A 32-bit mask naming one lane group limits the result to that group, as on CUDA."""
    import curobo  # noqa: F401  (installs the finder)
    from cuda.core import Device, LaunchConfig, launch

    kernel = _compile(_MASK_KERNEL, "mask_probe")
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    ballot = torch.zeros(32, dtype=torch.int32, device="cuda")
    match = torch.zeros(32, dtype=torch.int32, device="cuda")
    all_ = torch.zeros(32, dtype=torch.int32, device="cuda")
    any_ = torch.zeros(32, dtype=torch.int32, device="cuda")
    launch(
        stream,
        LaunchConfig(grid=1, block=32),
        kernel,
        ballot.data_ptr(),
        match.data_ptr(),
        all_.data_ptr(),
        any_.data_ptr(),
        group,
        np.int32(1),
    )
    torch.cuda.synchronize()
    ballot_np = [int(v) for v in ballot.cpu().numpy().view(np.uint32)]
    match_np = [int(v) for v in match.cpu().numpy().view(np.uint32)]
    for lane in range(32):
        first = (lane // group) * group
        lanes = range(first, first + group)
        mask = sum(1 << other for other in lanes)
        assert ballot_np[lane] == sum(1 << other for other in lanes if other % 2 == 0)
        assert match_np[lane] == sum(1 << other for other in lanes if other % 4 == lane % 4)
        assert match_np[lane] & ~mask == 0
        assert int(all_[lane]) == int(5 not in lanes)
        assert int(any_[lane]) == int(5 in lanes)


class _Stream:
    def __init__(self, handle: int) -> None:
        self._handle = handle

    def __cuda_stream__(self):
        return (0, self._handle)


_AXPY_KERNEL = """
extern "C" __global__ void axpy(float *out, const float *x, float a, int n, int64_t offset) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a * x[i] + static_cast<float>(offset);
}
"""


@requires_hip
def test_launch_marshals_arguments_and_checks_them() -> None:
    import curobo  # noqa: F401
    from cuda.core import Device, LaunchConfig, launch

    kernel = _compile(_AXPY_KERNEL, "axpy")
    assert kernel.num_arguments == 5
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    x = torch.arange(100, dtype=torch.float32, device="cuda")
    out = torch.zeros_like(x)
    config = LaunchConfig(grid=4, block=32)
    launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float32(2.5), 100, 3)
    torch.cuda.synchronize()
    assert torch.equal(out, 2.5 * x + 3.0)

    with pytest.raises(TypeError, match="takes 5 arguments"):
        launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float32(2.5), 100)
    with pytest.raises(TypeError, match="Python float"):
        launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), 2.5, 100, 3)
    out.zero_()
    with pytest.raises(TypeError, match="8 bytes"):
        launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float64(2.5), 100, 3)
    with pytest.raises(TypeError, match="8 bytes"):
        launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), ctypes.c_double(2.5), 100, 3)
    with pytest.raises(OverflowError):
        launch(
            stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float32(2.5), -(2**31) - 1, 3
        )
    torch.cuda.synchronize()
    assert not out.any(), "a rejected launch ran"
    with pytest.raises(TypeError, match="bool"):
        launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float32(2.5), True, 3)
    with pytest.raises(OverflowError):
        launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float32(2.5), 2**32, 3)
    with pytest.raises(OverflowError):
        launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float32(2.5), 100, 2**64)
    launch(stream, config, kernel, out.data_ptr(), x.data_ptr(), np.float32(2.5), 100, -3)
    torch.cuda.synchronize()
    assert torch.equal(out, 2.5 * x - 3.0)


@requires_hip
def test_compile_error_carries_the_compiler_log() -> None:
    import curobo  # noqa: F401

    with pytest.raises(RuntimeError, match="undeclared_symbol"):
        _compile('extern "C" __global__ void broken() { undeclared_symbol = 1; }', "broken")


@requires_hip
def test_reported_failures_leave_no_pending_error() -> None:
    """A failure raised here is not reported again by the next reader of the last error."""
    import curobo  # noqa: F401
    import cuda.bindings.runtime as cudart
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

    program = Program(_NVRTC_TYPEDEFS + _AXPY_KERNEL, code_type="c++", options=ProgramOptions())
    code = program.compile("cubin", name_expressions=("axpy",))
    kernel = code.get_kernel("axpy")
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    x = torch.arange(8, dtype=torch.float32, device="cuda")
    out = torch.zeros_like(x)
    cudart.cudaGetLastError()

    with pytest.raises(RuntimeError, match="launch of axpy"):
        launch(
            stream,
            LaunchConfig(grid=1, block=1 << 20),
            kernel,
            out.data_ptr(),
            x.data_ptr(),
            np.float32(1.0),
            8,
            0,
        )
    assert cudart.cudaGetLastError()[0] == cudart.cudaError_t.cudaSuccess
    with pytest.raises(RuntimeError, match="no_such_kernel"):
        code.get_kernel("no_such_kernel")
    assert cudart.cudaGetLastError()[0] == cudart.cudaError_t.cudaSuccess
    torch.ones(3, device="cuda").sum().item()


@requires_hip
def test_other_warp_sizes_are_refused() -> None:
    import curobo  # noqa: F401
    from cuda.core import Program, ProgramOptions

    program = Program(_AXPY_KERNEL, options=ProgramOptions(extra=["-mwavefrontsize64"]))
    with pytest.raises(NotImplementedError, match="32-lane"):
        program.compile("cubin", name_expressions=("axpy",))
    narrow = Program(_AXPY_KERNEL, options=ProgramOptions(extra=["-mno-wavefrontsize64"]))
    kernel = narrow.compile("cubin", name_expressions=("axpy",)).get_kernel("axpy")
    assert kernel.num_arguments == 5
    other = Program(_AXPY_KERNEL, options=ProgramOptions(extra=["--offload-arch=gfx906"]))
    with pytest.raises(NotImplementedError, match="another target"):
        other.compile("cubin", name_expressions=("axpy",))


@requires_hip
def test_a_device_with_another_warp_size_is_refused(monkeypatch) -> None:
    import curobo  # noqa: F401
    from cuda.core import Program, ProgramOptions

    monkeypatch.setattr(_hip, "warp_size", lambda: 64)
    with pytest.raises(NotImplementedError, match="32-lane"):
        Program(_AXPY_KERNEL, options=ProgramOptions()).compile(
            "cubin", name_expressions=("axpy",)
        )


@requires_hip
def test_an_error_pending_before_a_kernel_load_stays_pending() -> None:
    import curobo  # noqa: F401
    import cuda.bindings.runtime as cudart

    cudart.cudaGetLastError()
    assert _hip.runtime().hipSetDevice(4321) != 0
    try:
        _compile(_AXPY_KERNEL, "axpy")
        assert cudart.cudaGetLastError()[0] != cudart.cudaError_t.cudaSuccess
    finally:
        cudart.cudaGetLastError()


@requires_hip
def test_a_python_float_reaches_a_double_parameter() -> None:
    import curobo  # noqa: F401
    from cuda.core import Device, LaunchConfig, launch

    kernel = _compile(
        'extern "C" __global__ void put(double *out, double value) {'
        " out[threadIdx.x] = value; }",
        "put",
    )
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    out = torch.zeros(4, dtype=torch.float64, device="cuda")
    launch(stream, LaunchConfig(grid=1, block=4), kernel, out.data_ptr(), 0.1)
    torch.cuda.synchronize()
    assert out.tolist() == [0.1] * 4


_SHUFFLE_KERNEL = """
extern "C" __global__ void shuffles(int *pair, int *first, int *down, int *up, unsigned *ballot) {
  const int lane = threadIdx.x & 31;
  const int signed_mask = -1;
  pair[threadIdx.x] = __shfl_xor_sync(signed_mask, lane, 1);
  first[threadIdx.x] = __shfl_sync(0xffffffffu, lane, 0, 8);
  down[threadIdx.x] = __shfl_down_sync(0xffffffffu, lane, 1, 4);
  up[threadIdx.x] = __shfl_up_sync(0xffffffffu, lane, 1, 4);
  ballot[threadIdx.x] = __ballot_sync(0xffff, lane % 2 == 0);
  __syncwarp(signed_mask);
}
"""


@requires_hip
def test_shuffles_and_signed_masks() -> None:
    import curobo  # noqa: F401
    from cuda.core import Device, LaunchConfig, launch

    kernel = _compile(_SHUFFLE_KERNEL, "shuffles")
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    outputs = [torch.full((32,), -7, dtype=torch.int32, device="cuda") for _ in range(5)]
    launch(stream, LaunchConfig(grid=1, block=32), kernel, *[o.data_ptr() for o in outputs])
    torch.cuda.synchronize()
    pair, first, down, up, ballot = (o.tolist() for o in outputs)
    for lane in range(32):
        assert pair[lane] == lane ^ 1
        assert first[lane] == (lane // 8) * 8
        assert down[lane] == (lane + 1 if lane % 4 != 3 else lane)
        assert up[lane] == (lane - 1 if lane % 4 != 0 else lane)
        assert ballot[lane] == 0x5555


_SIGNED_GROUP_KERNEL = """
extern "C" __global__ void signed_groups(unsigned *match, int *all, int *any, int *down, int *up,
                                         int *first, int *crossed) {
  const int lane = threadIdx.x & 31;
  const int group_mask = static_cast<int>(0xffu << ((lane / 8) * 8));
  match[threadIdx.x] = __match_any_sync(group_mask, lane % 2);
  all[threadIdx.x] = __all_sync(group_mask, lane != 5);
  any[threadIdx.x] = __any_sync(group_mask, lane == 5);
  down[threadIdx.x] = __shfl_down_sync(group_mask, lane, 1, 4);
  up[threadIdx.x] = __shfl_up_sync(group_mask, lane, 1, 4);
  first[threadIdx.x] = __shfl_sync(group_mask, lane, 0, 8);
  crossed[threadIdx.x] = __shfl_xor_sync(0xffffffffu, lane, 4, 4);
}
"""


@requires_hip
def test_signed_group_masks_and_shuffle_widths() -> None:
    import curobo  # noqa: F401
    from cuda.core import Device, LaunchConfig, launch

    kernel = _compile(_SIGNED_GROUP_KERNEL, "signed_groups")
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    outputs = [torch.full((32,), -7, dtype=torch.int32, device="cuda") for _ in range(7)]
    launch(stream, LaunchConfig(grid=1, block=32), kernel, *[o.data_ptr() for o in outputs])
    torch.cuda.synchronize()
    match = [int(v) for v in outputs[0].cpu().numpy().view(np.uint32)]
    all_, any_, down, up, first, crossed = (o.tolist() for o in outputs[1:])
    for lane in range(32):
        lanes = range((lane // 8) * 8, (lane // 8) * 8 + 8)
        assert match[lane] == sum(1 << other for other in lanes if other % 2 == lane % 2)
        assert all_[lane] == int(5 not in lanes)
        assert any_[lane] == int(5 in lanes)
        assert down[lane] == (lane + 1 if lane % 4 != 3 else lane)
        assert up[lane] == (lane - 1 if lane % 4 != 0 else lane)
        assert first[lane] == (lane // 8) * 8
        # A shuffle may read an earlier group of ``width`` lanes but not a later one.
        assert crossed[lane] == (lane - 4 if lane % 8 >= 4 else lane)


_ROUNDED_FORMS = [
    ("float", "__fadd_rn(a[i] * b[i], c[i])"),
    ("float", "__fmul_rn(a[i], b[i]) + c[i]"),
    ("float", "__fsub_rn(a[i] * b[i], -c[i])"),
    ("float", "__fsub_rn(c[i], -(a[i] * b[i]))"),
    ("double", "__dadd_rn(a[i] * b[i], c[i])"),
    ("double", "__dmul_rn(a[i], b[i]) + c[i]"),
    ("double", "__dsub_rn(a[i] * b[i], -c[i])"),
]


@requires_hip
@pytest.mark.parametrize("ctype,expression", _ROUNDED_FORMS)
def test_each_rounded_intrinsic_blocks_contraction(ctype: str, expression: str) -> None:
    """One rounded intrinsic next to an ordinary operation is enough to round twice."""
    import curobo  # noqa: F401
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

    dtype = np.float32 if ctype == "float" else np.float64
    source = """
    extern "C" __global__ void form(T *out, const T *a, const T *b, const T *c, int n) {
      const int i = blockIdx.x * blockDim.x + threadIdx.x;
      if (i < n) out[i] = EXPRESSION;
    }
    """
    rng = np.random.default_rng(11)
    n = 4096
    a_np = rng.uniform(0.5, 2.0, n).astype(dtype)
    b_np = rng.uniform(0.5, 2.0, n).astype(dtype)
    c_np = (-(a_np * b_np) * rng.uniform(0.999, 1.001, n)).astype(dtype)
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    results = {}
    for name, body in (("rounded", expression), ("plain", "a[i] * b[i] + c[i]")):
        program = Program(
            source.replace("T ", ctype + " ").replace("EXPRESSION", body),
            options=ProgramOptions(fma=True, ftz=True),
        )
        kernel = program.compile("cubin", name_expressions=("form",)).get_kernel("form")
        a, b, c = (torch.from_numpy(v).cuda() for v in (a_np, b_np, c_np))
        out = torch.zeros(n, dtype=a.dtype, device="cuda")
        launch(
            stream,
            LaunchConfig(grid=n // 256, block=256),
            kernel,
            out.data_ptr(),
            a.data_ptr(),
            b.data_ptr(),
            c.data_ptr(),
            n,
        )
        torch.cuda.synchronize()
        results[name] = out.cpu().numpy()
    two_roundings = (a_np * b_np).astype(dtype) + c_np
    assert np.any(results["plain"] != two_roundings), "inputs do not show contraction"
    np.testing.assert_array_equal(results["rounded"], two_roundings)


@requires_hip
def test_compile_errors_name_the_callers_line() -> None:
    import curobo  # noqa: F401
    from cuda.core import Program, ProgramOptions

    source = 'extern "C" __global__ void broken() {\n\n  undeclared_symbol = 1;\n}\n'
    with pytest.raises(RuntimeError, match=r":3:\d+: error"):
        Program(source, options=ProgramOptions()).compile("cubin", name_expressions=("broken",))


@requires_hip
def test_kernel_load_leaves_no_pending_error() -> None:
    """Reading a kernel's parameter table must not leave an error for the next caller."""
    import curobo  # noqa: F401
    import cuda.bindings.runtime as cudart

    cudart.cudaGetLastError()
    _compile(_AXPY_KERNEL, "axpy")
    assert cudart.cudaGetLastError()[0] == cudart.cudaError_t.cudaSuccess
    torch.zeros(4, device="cuda").sum().item()


_ROUNDED_KERNEL = """
extern "C" __global__ void rounded(float *separate, float *fused, const float *a, const float *b,
                                   const float *c, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  separate[i] = __fadd_rn(c[i], __fmul_rn(a[i], b[i]));
  fused[i] = a[i] * b[i] + c[i];
}
"""


@requires_hip
def test_rounded_intrinsics_are_not_contracted() -> None:
    """``__fmul_rn`` followed by ``__fadd_rn`` rounds twice, even when contraction is on."""
    import curobo  # noqa: F401
    from cuda.core import Device, LaunchConfig, Program, ProgramOptions, launch

    program = Program(_ROUNDED_KERNEL, code_type="c++", options=ProgramOptions(fma=True, ftz=True))
    kernel = program.compile("cubin", name_expressions=("rounded",)).get_kernel("rounded")
    stream = Device().create_stream(_Stream(torch.cuda.current_stream().cuda_stream))
    rng = np.random.default_rng(7)
    n = 4096
    a_np = rng.uniform(0.5, 2.0, n).astype(np.float32)
    b_np = rng.uniform(0.5, 2.0, n).astype(np.float32)
    c_np = (-(a_np * b_np) * rng.uniform(0.999, 1.001, n)).astype(np.float32)
    a, b, c = (torch.from_numpy(v).cuda() for v in (a_np, b_np, c_np))
    separate = torch.zeros(n, dtype=torch.float32, device="cuda")
    fused = torch.zeros(n, dtype=torch.float32, device="cuda")
    launch(
        stream,
        LaunchConfig(grid=n // 256, block=256),
        kernel,
        separate.data_ptr(),
        fused.data_ptr(),
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        n,
    )
    torch.cuda.synchronize()
    expected = (a_np * b_np).astype(np.float32) + c_np
    exact = (a_np.astype(np.float64) * b_np.astype(np.float64) + c_np.astype(np.float64)).astype(
        np.float32
    )
    assert np.any(expected != exact), "inputs do not distinguish fused from separate rounding"
    np.testing.assert_array_equal(separate.cpu().numpy(), expected)
    np.testing.assert_array_equal(fused.cpu().numpy(), exact)
