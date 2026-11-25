# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Stand-in for ``cuda.core``: compile with hipRTC, launch through the HIP runtime.

Only the names and behaviour cuRobo's kernel backend uses are provided.
"""

from curobo._src.curobolib.backends.hip_core._launch import Device, LaunchConfig, Stream, launch
from curobo._src.curobolib.backends.hip_core._program import (
    Kernel,
    ObjectCode,
    Program,
    ProgramOptions,
)

__all__ = [
    "Device",
    "Kernel",
    "LaunchConfig",
    "ObjectCode",
    "Program",
    "ProgramOptions",
    "Stream",
    "launch",
]
