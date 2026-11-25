# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""Stand-in for ``cuda.bindings`` on a ROCm build of PyTorch."""

from curobo._src.curobolib.backends.hip_core import _runtime as runtime  # noqa: F401

__path__ = []
