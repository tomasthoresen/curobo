# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Stand-in for the compile half of ``cuda.core``: programs, options, object code, kernels."""

# Standard Library
import ctypes
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

from curobo._src.curobolib.backends.hip_core import _hip

_INCLUDE_DIR = Path(__file__).resolve().parent / "include"

# cuRobo's kernel cache prepends typedefs for the fixed-width integer types because NVRTC does
# not provide them. The host headers used by hipRTC do, with different underlying types.
_FIXED_WIDTH_TYPEDEF = re.compile(
    r"^[ \t]*typedef\s+(?:(?:signed|unsigned|char|short|int|long)\s+)+u?int(?:8|16|32|64)_t\s*;"
    r"[ \t]*(?://[^\n]*)?$",
    re.MULTILINE,
)
# "#line 1" keeps the compiler's line numbers those of the caller's source.
_PREAMBLE = "#include <hip/hip_runtime.h>\n#include <stdint.h>\n#line 1\n"
# The prelude maps CUDA's 32-bit lane masks onto HIP's builtins and is written for 32-lane warps.
_WARP_SIZE = 32
_MAX_PARAMETERS = 1024


@dataclass
class ProgramOptions:
    """Compile options, in ``cuda.core`` terms, translated to hipRTC flags."""

    std: Optional[str] = "c++17"
    arch: Optional[str] = None
    include_path: Union[str, Sequence[str], None] = None
    define_macro: Union[str, Sequence[Any], None] = None
    ftz: Optional[bool] = None
    fma: Optional[bool] = None
    prec_div: Optional[bool] = None
    prec_sqrt: Optional[bool] = None
    lineinfo: Optional[bool] = None
    debug: Optional[bool] = None
    device_code_optimize: Optional[bool] = None
    name: str = "curobo_kernel"
    extra: List[str] = field(default_factory=list)

    def as_flags(self) -> List[str]:
        """Return the hipRTC command line for these options."""
        flags = [f"--offload-arch={_hip.arch_name()}", "-DHIP_ENABLE_WARP_SYNC_BUILTINS"]
        if self.std:
            flags.append(f"--std={self.std}")
        if self.debug:
            flags += ["-O0", "-g"]
        else:
            flags += ["-O3", "-DNDEBUG"]
        if self.lineinfo and not self.debug:
            flags.append("-gline-tables-only")
        if self.fma is not None:
            # "fast-honor-pragmas" contracts like "fast" but leaves operations alone that the
            # prelude marks as not contractible (the round-to-nearest intrinsics).
            flags.append("-ffp-contract=fast-honor-pragmas" if self.fma else "-ffp-contract=off")
        if self.ftz:
            flags.append("-fgpu-flush-denormals-to-zero")
        # CUDA's reduced-precision division and square root options have no hipRTC counterpart
        # and are not translated. Value-changing fast-math stays off.
        flags += [
            "-fno-finite-math-only",
            "-fno-associative-math",
            "-fno-reciprocal-math",
            "-fno-strict-aliasing",
        ]
        for macro in _as_list(self.define_macro):
            if isinstance(macro, (tuple, list)):
                flags.append(f"-D{macro[0]}={macro[1]}")
            else:
                flags.append(f"-D{macro}")
        for directory in _hip.system_include_dirs():
            flags.append(f"-I{directory}")
        flags.append(f"-I{_INCLUDE_DIR}")
        for directory in _as_list(self.include_path):
            flags.append(f"-I{directory}")
        flags += ["-include", str(_INCLUDE_DIR / "hip_compat.h")]
        flags += list(self.extra)
        if any(str(flag).startswith(("--offload-arch", "-mcpu", "-march")) for flag in self.extra):
            raise NotImplementedError(
                "hip_core: kernels are compiled for the device PyTorch uses; another target "
                "cannot be selected"
            )
        wide = any(_selects_wide_warps(str(flag)) for flag in flags)
        if wide or _hip.warp_size() != _WARP_SIZE:
            raise NotImplementedError(
                f"hip_core: the kernels' lane masks need {_WARP_SIZE}-lane warps; this device or "
                "these options select another size"
            )
        return flags


def _selects_wide_warps(flag: str) -> bool:
    """Return whether a compiler flag asks for 64-lane warps."""
    if "wavefrontsize64" not in flag:
        return False
    return not (flag.startswith("-mno-") or flag.startswith("-wavefrontsize64"))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    return list(value)


class Kernel:
    """A compiled kernel entry point."""

    def __init__(self, name: str, lowered_name: bytes, library: "ObjectCode") -> None:
        hip = _hip.runtime()
        self._name = name
        self._library = library  # keeps the loaded code alive
        kernel = ctypes.c_void_p()
        _hip.check(
            hip.hipLibraryGetKernel(ctypes.byref(kernel), library._handle, lowered_name),
            f"hipLibraryGetKernel({name})",
        )
        sizes = []
        offset, size = ctypes.c_size_t(), ctypes.c_size_t()
        error_was_pending = hip.hipPeekAtLastError() != 0
        while len(sizes) < _MAX_PARAMETERS and (
            hip.hipKernelGetParamInfo(kernel, len(sizes), ctypes.byref(offset), ctypes.byref(size))
            == 0
        ):
            sizes.append(size.value)
        # The probe past the last parameter leaves an error behind that the next caller of
        # hipGetLastError, PyTorch included, would report as its own. If an error was already
        # pending, one stays pending for its owner to find; the runtime keeps a single code and
        # offers no way to put the earlier one back, so the code it reads is the probe's.
        if not error_was_pending:
            hip.hipGetLastError()
        function = ctypes.c_void_p()
        _hip.check(
            hip.hipKernelGetFunction(ctypes.byref(function), kernel), "hipKernelGetFunction"
        )
        self._param_sizes: Tuple[int, ...] = tuple(sizes)
        self._handle = function.value
        self._function = function

    @property
    def num_arguments(self) -> int:
        """Number of parameters of the kernel."""
        return len(self._param_sizes)

    @property
    def handle(self) -> int:
        """Native function handle."""
        return self._handle


class ObjectCode:
    """Device code produced by :meth:`Program.compile`.

    The code stays loaded for the life of the process. Kernels are compiled once and cached by
    the caller, so the number of loaded objects is bounded by the number of distinct kernels.
    """

    def __init__(self, code: bytes, lowered_names: dict) -> None:
        self._code = ctypes.create_string_buffer(code, len(code))
        self._lowered_names = lowered_names
        self._handle = ctypes.c_void_p()
        _hip.check(
            _hip.runtime().hipLibraryLoadData(
                ctypes.byref(self._handle), self._code, None, None, 0, None, None, 0
            ),
            "hipLibraryLoadData",
        )

    @property
    def code(self) -> bytes:
        """The code object's bytes."""
        return self._code.raw

    def get_kernel(self, name: str) -> Kernel:
        """Return the entry point registered under a name expression."""
        lowered = self._lowered_names.get(name)
        if lowered is None:
            lowered = name.encode()
        return Kernel(name, lowered, self)


class Program:
    """A translation unit to be compiled with hipRTC."""

    def __init__(
        self, code: str, code_type: str = "c++", options: Optional[ProgramOptions] = None
    ) -> None:
        if code_type != "c++":
            raise NotImplementedError(f"hip_core: code type {code_type!r} is not supported")
        self._source = _PREAMBLE + _FIXED_WIDTH_TYPEDEF.sub("", code)
        self._options = options if options is not None else ProgramOptions()

    def compile(
        self,
        target_type: str = "cubin",
        name_expressions: Sequence[str] = (),
        logs: Any = None,
    ) -> ObjectCode:
        """Compile to device code and return it loaded."""
        if target_type != "cubin":
            raise NotImplementedError(f"hip_core: target type {target_type!r} is not supported")
        rtc = _hip.compiler()
        program = ctypes.c_void_p()
        unit_name = (self._options.name or "curobo_kernel").encode()
        code = rtc.hiprtcCreateProgram(
            ctypes.byref(program), self._source.encode(), unit_name, 0, None, None
        )
        if code != 0:
            raise RuntimeError(f"hip_core: hiprtcCreateProgram failed ({_rtc_error(code)})")
        try:
            for expression in name_expressions:
                code = rtc.hiprtcAddNameExpression(program, expression.encode())
                if code != 0:
                    raise RuntimeError(
                        f"hip_core: name expression {expression!r} rejected ({_rtc_error(code)})"
                    )
            flags = self._options.as_flags()
            c_flags = (ctypes.c_char_p * len(flags))(*[flag.encode() for flag in flags])
            code = rtc.hiprtcCompileProgram(program, len(flags), c_flags)
            log = _program_log(rtc, program)
            if logs is not None and log:
                logs.write(log)
            if code != 0:
                raise RuntimeError(
                    f"hip_core: hipRTC compilation failed ({_rtc_error(code)}):\n{log}"
                )
            lowered_names = {}
            for expression in name_expressions:
                lowered = ctypes.c_char_p()
                code = rtc.hiprtcGetLoweredName(
                    program, expression.encode(), ctypes.byref(lowered)
                )
                if code != 0 or not lowered.value:
                    raise RuntimeError(
                        f"hip_core: no lowered name for {expression!r} ({_rtc_error(code)})"
                    )
                lowered_names[expression] = bytes(lowered.value)
            size = ctypes.c_size_t()
            code = rtc.hiprtcGetCodeSize(program, ctypes.byref(size))
            if code != 0 or size.value == 0:
                raise RuntimeError(f"hip_core: no device code produced ({_rtc_error(code)})")
            buffer = ctypes.create_string_buffer(size.value)
            code = rtc.hiprtcGetCode(program, buffer)
            if code != 0:
                raise RuntimeError(f"hip_core: device code not readable ({_rtc_error(code)})")
            binary = buffer.raw
        finally:
            rtc.hiprtcDestroyProgram(ctypes.byref(program))
        return ObjectCode(binary, lowered_names)


def _rtc_error(code: int) -> str:
    text = _hip.compiler().hiprtcGetErrorString(int(code))
    return text.decode(errors="replace") if text else f"hiprtc error {code}"


def _program_log(rtc: ctypes.CDLL, program: ctypes.c_void_p) -> str:
    size = ctypes.c_size_t()
    rtc.hiprtcGetProgramLogSize(program, ctypes.byref(size))
    if size.value <= 1:
        return ""
    buffer = ctypes.create_string_buffer(size.value + 1)
    rtc.hiprtcGetProgramLog(program, buffer)
    return buffer.value.decode(errors="replace")
