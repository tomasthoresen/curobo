# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for what the HIP stand-in hands to the runtime when a kernel is launched."""

# Standard Library
import ctypes
from typing import Any, List, Tuple

# Third Party
import numpy as np
import pytest
import torch

from curobo._src.curobolib.backends.hip_core import _hip, _launch, _program

requires_hip = pytest.mark.skipif(
    not (getattr(torch.version, "hip", None) is not None and torch.cuda.is_available()),
    reason="needs a ROCm build of PyTorch with a device",
)


class _RecordingRuntime:
    """Stands in for the HIP runtime library and records each launch call."""

    def __init__(self) -> None:
        self.calls: List[Tuple[Any, ...]] = []

    def hipModuleLaunchKernel(self, function, gx, gy, gz, bx, by, bz, shmem, stream, args, extra):
        sizes = function.sizes
        values = [ctypes.string_at(args[i], sizes[i]) for i in range(len(sizes))]
        self.calls.append(((gx, gy, gz), (bx, by, bz), shmem, stream.value, values))
        return 0


class _FakeKernel:
    """A kernel as :func:`launch` sees it: a name, parameter sizes and a function handle."""

    def __init__(self, *sizes: int) -> None:
        self._name = "recorded"
        self._param_sizes = tuple(sizes)
        self._function = ctypes.c_void_p(1)
        self._function.sizes = self._param_sizes


class _ForeignStream:
    def __init__(self, handle: int) -> None:
        self._handle = handle

    def __cuda_stream__(self) -> Tuple[int, int]:
        return (0, self._handle)


class _WithHandle:
    def __init__(self, handle: int) -> None:
        self.handle = handle


@pytest.fixture
def recorded(monkeypatch) -> _RecordingRuntime:
    runtime = _RecordingRuntime()
    monkeypatch.setattr(_hip, "runtime", lambda: runtime)
    return runtime


def test_launch_passes_all_dimensions_and_shared_memory(recorded) -> None:
    config = _launch.LaunchConfig(grid=(2, 3, 5), block=(7, 11, 13), shmem_size=4096)
    _launch.launch(_launch.Stream(0x1234), config, _FakeKernel())
    assert recorded.calls == [((2, 3, 5), (7, 11, 13), 4096, 0x1234, [])]


def test_launch_uses_the_stream_of_a_foreign_stream_object(recorded) -> None:
    _launch.launch(_ForeignStream(0xABCD00), _launch.LaunchConfig(grid=1, block=1), _FakeKernel())
    assert recorded.calls[0][3] == 0xABCD00


def test_unsigned_and_handle_arguments_reach_the_kernel(recorded) -> None:
    kernel = _FakeKernel(4, 8, 8, 1)
    config = _launch.LaunchConfig(grid=1, block=1)
    _launch.launch(
        _launch.Stream(1), config, kernel, 0xFFFFFFFF, 2**64 - 1, _WithHandle(0x7F00DEADBEEF), 255
    )
    values = recorded.calls[0][4]
    assert values[0] == np.uint32(0xFFFFFFFF).tobytes()
    assert values[1] == np.uint64(2**64 - 1).tobytes()
    assert values[2] == np.uint64(0x7F00DEADBEEF).tobytes()
    assert values[3] == b"\xff"


def test_an_object_without_an_integer_handle_is_refused(recorded) -> None:
    config = _launch.LaunchConfig(grid=1, block=1)
    for value in (_WithHandle("0x10"), object(), "text", None):
        with pytest.raises(TypeError, match="unsupported type"):
            _launch.launch(_launch.Stream(1), config, _FakeKernel(8), value)
    assert recorded.calls == []


def test_a_wrong_number_of_arguments_is_refused_before_the_launch(recorded) -> None:
    config = _launch.LaunchConfig(grid=1, block=1)
    for arguments in ((), (1,), (1, 2, 3)):
        with pytest.raises(TypeError, match="takes 2 arguments"):
            _launch.launch(_launch.Stream(1), config, _FakeKernel(8, 4), *arguments)
    assert recorded.calls == []


@pytest.mark.parametrize("dimensions", [(), (1, 1, 1, 1), [2, 2, 2, 2, 2]])
def test_launch_config_refuses_other_than_one_to_three_dimensions(dimensions) -> None:
    with pytest.raises(ValueError, match="grid"):
        _launch.LaunchConfig(grid=dimensions, block=1)
    with pytest.raises(ValueError, match="block"):
        _launch.LaunchConfig(grid=1, block=dimensions)


def test_cluster_and_cooperative_launches_are_refused() -> None:
    with pytest.raises(NotImplementedError):
        _launch.LaunchConfig(grid=1, block=1, cluster=(2, 1, 1))
    with pytest.raises(NotImplementedError):
        _launch.LaunchConfig(grid=1, block=1, cooperative_launch=True)


_GRID_KERNEL = """
extern "C" __global__ void mark(int *seen, int grid_x, int grid_y, int block_x, int block_y,
                                int block_z) {
  const int block = blockIdx.x + grid_x * (blockIdx.y + grid_y * blockIdx.z);
  const int thread = threadIdx.x + block_x * (threadIdx.y + block_y * threadIdx.z);
  atomicAdd(&seen[block * (block_x * block_y * block_z) + thread], 1);
}
"""


@requires_hip
def test_every_thread_of_a_three_dimensional_launch_runs_once() -> None:
    options = _program.ProgramOptions(std="c++17", fma=True)
    kernel = _program.Program(_GRID_KERNEL, "c++", options).compile("cubin").get_kernel("mark")
    grid, block = (2, 3, 2), (4, 2, 3)
    count = int(np.prod(grid) * np.prod(block))
    seen = torch.zeros(count + 64, dtype=torch.int32, device="cuda")
    device = _launch.Device()
    with pytest.raises(NotImplementedError):
        device.create_stream(None)
    stream = device.create_stream(_ForeignStream(torch.cuda.current_stream().cuda_stream))
    assert stream.handle == torch.cuda.current_stream().cuda_stream
    _launch.launch(
        stream,
        _launch.LaunchConfig(grid=grid, block=block),
        kernel,
        seen.data_ptr(),
        grid[0],
        grid[1],
        block[0],
        block[1],
        block[2],
    )
    stream.sync()
    result = seen.cpu().numpy()
    assert (result[:count] == 1).all(), result
    assert not result[count:].any(), result
