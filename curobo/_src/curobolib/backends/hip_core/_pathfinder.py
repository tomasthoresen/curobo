# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Stand-in for ``cuda.pathfinder``: header directories for the hipRTC compile."""

# Standard Library
from pathlib import Path
from typing import Optional

_INCLUDE_DIR = Path(__file__).resolve().parent / "include"


def find_nvidia_header_directory(libname: str) -> Optional[str]:
    """Return the include directory that stands in for a CUDA component's headers.

    The runtime and runtime-compiler headers map to this package's shim directory. Components
    with no HIP counterpart (for example the compiler's tensor-core headers) return ``None``.
    """
    if libname in ("cudart", "nvrtc"):
        return str(_INCLUDE_DIR)
    return None
