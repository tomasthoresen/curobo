# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Handles to the HIP runtime and hipRTC libraries that PyTorch loaded.

The kernels must be compiled for, and launched through, the same HIP runtime that owns
PyTorch's device context and streams. A second copy of the runtime in the process would
have its own context. The libraries are therefore located from the running process.
"""

# Standard Library
import ctypes
import glob
import os
from functools import lru_cache
from typing import List, Optional


def _mapped_library_dir(name: str) -> Optional[str]:
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.rsplit(" ", 1)[-1].strip()
            if os.path.basename(path).startswith(name):
                return os.path.dirname(path)
    return None


@lru_cache(maxsize=None)
def library_dir() -> str:
    """Return the directory of the HIP runtime loaded by PyTorch."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("hip_core: PyTorch reports no available HIP device")
    torch.cuda.init()
    directory = _mapped_library_dir("libamdhip64.so")
    if directory is None:
        raise RuntimeError("hip_core: libamdhip64 is not loaded in this process")
    return directory


def _load(directory: str, stem: str) -> ctypes.CDLL:
    candidates = sorted(glob.glob(os.path.join(directory, stem + ".so*")), key=len)
    if not candidates:
        raise RuntimeError(f"hip_core: {stem} not found next to the HIP runtime in {directory}")
    return ctypes.CDLL(candidates[0])


@lru_cache(maxsize=None)
def runtime() -> ctypes.CDLL:
    """Return the HIP runtime library with the signatures this package uses."""
    lib = _load(library_dir(), "libamdhip64")
    lib.hipGetErrorString.restype = ctypes.c_char_p
    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetLastError.restype = ctypes.c_int
    lib.hipPeekAtLastError.restype = ctypes.c_int
    lib.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.hipLibraryLoadData.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    lib.hipLibraryGetKernel.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    lib.hipKernelGetParamInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.hipKernelGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib


@lru_cache(maxsize=None)
def compiler() -> ctypes.CDLL:
    """Return the hipRTC library with the signatures this package uses."""
    lib = _load(library_dir(), "libhiprtc")
    lib.hiprtcGetErrorString.restype = ctypes.c_char_p
    lib.hiprtcGetErrorString.argtypes = [ctypes.c_int]
    lib.hiprtcCreateProgram.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.hiprtcAddNameExpression.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hiprtcCompileProgram.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.hiprtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.hiprtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hiprtcGetLoweredName.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_char_p),
    ]
    lib.hiprtcGetCodeSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    lib.hiprtcGetCode.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hiprtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    return lib


def error_string(code: int) -> str:
    """Return the HIP runtime's text for an error code."""
    text = runtime().hipGetErrorString(int(code))
    return text.decode(errors="replace") if text else "unknown error"


def check(code: int, what: str) -> None:
    """Raise when a HIP runtime call failed.

    The runtime also records the failure as the thread's last error. It is reported here, so the
    record is cleared; otherwise the next caller that reads it, PyTorch included, would report
    this failure a second time as its own.
    """
    if code != 0:
        runtime().hipGetLastError()
        raise RuntimeError(f"hip_core: {what} failed with hipError {code} ({error_string(code)})")


@lru_cache(maxsize=None)
def arch_name() -> str:
    """Return the device's offload architecture, for example ``gfx1151``."""
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return properties.gcnArchName.split(":")[0]


@lru_cache(maxsize=None)
def warp_size() -> int:
    """Return the number of lanes in one warp on the device."""
    import torch

    return int(torch.cuda.get_device_properties(torch.cuda.current_device()).warp_size)


@lru_cache(maxsize=None)
def system_include_dirs() -> List[str]:
    """Return the include directories every hipRTC compile needs.

    These are the HIP headers shipped with the runtime and the host compiler's freestanding
    headers, which supply ``<stdint.h>`` and ``<stddef.h>`` to the kernel sources.
    """
    directories = []
    hip_include = os.path.join(os.path.dirname(library_dir()), "include")
    if os.path.isdir(os.path.join(hip_include, "hip")):
        directories.append(hip_include)
    else:
        rocm_path = os.environ.get("ROCM_PATH", "/opt/rocm")
        if os.path.isdir(os.path.join(rocm_path, "include", "hip")):
            directories.append(os.path.join(rocm_path, "include"))
        else:
            raise RuntimeError("hip_core: HIP headers not found; set ROCM_PATH")

    def version(path: str) -> int:
        try:
            return int(path.rstrip("/").split("/")[-2].split(".")[0])
        except ValueError:
            return -1

    host = sorted(glob.glob("/usr/lib/gcc/*/*/include"), key=version)
    if not host:
        raise RuntimeError(
            "hip_core: the host compiler's headers (/usr/lib/gcc/*/*/include) were not found; "
            "the kernels need <stdint.h> from them"
        )
    directories.append(host[-1])
    return directories
