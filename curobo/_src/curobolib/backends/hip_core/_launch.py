# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Stand-in for the launch half of ``cuda.core``: devices, streams, launch configuration."""

# Standard Library
import ctypes
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

# Third Party
import numpy as np

from curobo._src.curobolib.backends.hip_core import _hip
from curobo._src.curobolib.backends.hip_core._program import Kernel

_MAX_DIMENSION = 2**31 - 1  # launch dimensions are passed to the runtime as 32-bit values


@dataclass
class LaunchConfig:
    """Grid, block and dynamic shared-memory size of one kernel launch."""

    grid: Union[int, Tuple[int, ...]] = 1
    block: Union[int, Tuple[int, ...]] = 1
    shmem_size: int = 0
    cluster: Any = None
    cooperative_launch: bool = False

    def __post_init__(self) -> None:
        self.grid = _dim3(self.grid, "grid")
        self.block = _dim3(self.block, "block")
        size = 0 if self.shmem_size is None else self.shmem_size
        if (
            not isinstance(size, (int, np.integer))
            or isinstance(size, bool)
            or not 0 <= size <= _MAX_DIMENSION
        ):
            raise ValueError(f"hip_core: invalid shared-memory size {self.shmem_size!r}")
        self.shmem_size = int(size)
        if self.cluster is not None or self.cooperative_launch:
            raise NotImplementedError(
                "hip_core: cluster and cooperative launches are not available"
            )


def _dim3(value: Union[int, Sequence[int]], what: str) -> Tuple[int, int, int]:
    values = value if isinstance(value, (tuple, list)) else (value,)
    if not all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
        raise ValueError(f"hip_core: invalid {what} dimensions {value!r}")
    dims = tuple(int(v) for v in values)
    if not 1 <= len(dims) <= 3 or any(not 1 <= v <= _MAX_DIMENSION for v in dims):
        raise ValueError(f"hip_core: invalid {what} dimensions {value!r}")
    return dims + (1,) * (3 - len(dims))


class Stream:
    """A HIP stream owned elsewhere, for example by PyTorch."""

    def __init__(self, handle: int) -> None:
        self._handle = int(handle)
        self._c_handle = ctypes.c_void_p(self._handle)

    @property
    def handle(self) -> int:
        """Native stream handle."""
        return self._handle

    def __cuda_stream__(self) -> Tuple[int, int]:
        return (0, self._handle)

    def sync(self) -> None:
        """Block until the stream's queued work has finished."""
        _hip.check(_hip.runtime().hipStreamSynchronize(self._c_handle), "hipStreamSynchronize")


class Device:
    """The HIP device PyTorch is using."""

    def __init__(self, device_id: Optional[int] = None) -> None:
        import torch

        _hip.runtime()
        self._device_id = torch.cuda.current_device() if device_id is None else int(device_id)

    @property
    def device_id(self) -> int:
        """Device ordinal."""
        return self._device_id

    @property
    def arch(self) -> str:
        """Offload architecture of the device, for example ``gfx1151``."""
        return _hip.arch_name()

    def set_current(self, ctx: Any = None) -> None:
        """Do nothing: PyTorch already made the device's context current."""
        return None

    def create_stream(self, obj: Any = None, options: Any = None) -> Stream:
        """Wrap an existing stream that exposes ``__cuda_stream__``."""
        if obj is None:
            raise NotImplementedError("hip_core: only existing streams can be wrapped")
        _, handle = obj.__cuda_stream__()
        return Stream(handle)

    def sync(self) -> None:
        """Block until all device work has finished."""
        import torch

        torch.cuda.synchronize(self._device_id)


def _marshal(value: Any, size: int, index: int, kernel: Kernel) -> Any:
    """Convert one launch argument the way ``cuda.core`` does, and check it against the kernel.

    A Python ``int`` is passed as a pointer-sized integer, a Python ``float`` as a double and a
    ``bool`` as one byte; NumPy scalars and ctypes values keep their own type. The runtime reads
    as many bytes as the kernel parameter has. A Python ``int`` may therefore be wider than its
    parameter as long as its value fits; every other value must have the parameter's size.
    """
    where = f"argument {index} of {kernel._name}"
    if isinstance(value, np.generic):
        if value.dtype.itemsize != size:
            raise TypeError(
                f"hip_core: {where} has {value.dtype.itemsize} bytes ({value.dtype}) but the "
                f"kernel parameter has {size}"
            )
        return ctypes.create_string_buffer(value.tobytes(), size)
    if isinstance(value, bool):
        if size != 1:
            raise TypeError(
                f"hip_core: {where} is a bool (1 byte) but the kernel parameter has {size} bytes"
            )
        return ctypes.c_bool(value)
    if isinstance(value, int):
        return _marshal_int(value, size, where)
    if isinstance(value, float):
        if size != 8:
            raise TypeError(
                f"hip_core: {where} is a Python float (8 bytes) but the kernel parameter has "
                f"{size}; pass a numpy.float32"
            )
        return ctypes.c_double(value)
    if isinstance(value, (ctypes._SimpleCData, ctypes.Structure, ctypes.Array)):
        if ctypes.sizeof(value) != size:
            raise TypeError(
                f"hip_core: {where} has {ctypes.sizeof(value)} bytes but the kernel parameter "
                f"has {size}"
            )
        return value
    handle = getattr(value, "handle", None)
    if isinstance(handle, int):
        return _marshal_int(handle, size, where)
    raise TypeError(f"hip_core: unsupported type {type(value).__name__} for {where}")


def _marshal_int(value: int, size: int, where: str) -> Any:
    if size > 8:
        raise TypeError(f"hip_core: {where} is an int but the kernel parameter has {size} bytes")
    bits = 8 * size
    if not -(1 << (bits - 1)) <= value < (1 << bits):
        raise OverflowError(f"hip_core: {where} = {value} does not fit in {size} bytes")
    return ctypes.c_uint64(value & 0xFFFFFFFFFFFFFFFF)


def launch(stream: Any, config: LaunchConfig, kernel: Kernel, *kernel_args: Any) -> None:
    """Queue a kernel on a stream."""
    sizes = kernel._param_sizes
    if len(kernel_args) != len(sizes):
        raise TypeError(
            f"hip_core: {kernel._name} takes {len(sizes)} arguments, {len(kernel_args)} given"
        )
    holders = [_marshal(value, sizes[i], i, kernel) for i, value in enumerate(kernel_args)]
    pointers = (ctypes.c_void_p * max(len(holders), 1))(
        *[ctypes.addressof(holder) for holder in holders]
    )
    if isinstance(stream, Stream):
        stream_handle = stream._c_handle
    else:
        stream_handle = ctypes.c_void_p(int(stream.__cuda_stream__()[1]))
    grid, block = config.grid, config.block
    code = _hip.runtime().hipModuleLaunchKernel(
        kernel._function,
        grid[0],
        grid[1],
        grid[2],
        block[0],
        block[1],
        block[2],
        config.shmem_size,
        stream_handle,
        pointers,
        None,
    )
    _hip.check(code, f"launch of {kernel._name}")
