# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Stand-in for ``cuda.bindings.runtime``: the few runtime calls cuRobo's backend makes."""

# Standard Library
import ctypes
from typing import Tuple

from curobo._src.curobolib.backends.hip_core import _hip


class cudaError_t:
    """Error codes compared against by callers."""

    cudaSuccess = 0


class cudaFuncAttribute:
    """Function attributes accepted by :func:`cudaFuncSetAttribute`."""

    cudaFuncAttributeMaxDynamicSharedMemorySize = 8
    cudaFuncAttributePreferredSharedMemoryCarveout = 9


def cudaGetLastError() -> Tuple[int]:
    """Return and clear the calling thread's last HIP error."""
    return (int(_hip.runtime().hipGetLastError()),)


def cudaStreamSynchronize(stream_handle: int) -> Tuple[int]:
    """Block until the stream's queued work has finished."""
    return (int(_hip.runtime().hipStreamSynchronize(ctypes.c_void_p(int(stream_handle)))),)


def cudaFuncSetAttribute(function_handle: int, attribute: int, value: int) -> Tuple[int]:
    """Accept a function attribute.

    CUDA requires an opt-in before a kernel may use more dynamic shared memory than the default
    limit. HIP has no such opt-in: a launch may request up to the device's shared-memory size,
    and a larger request fails at launch. The call therefore succeeds without effect.
    """
    return (cudaError_t.cudaSuccess,)
