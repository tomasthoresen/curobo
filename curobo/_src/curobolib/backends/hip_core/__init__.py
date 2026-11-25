# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#

"""HIP implementation of the ``cuda.core`` surface used by cuRobo's kernel backend.

cuRobo compiles its kernels at run time and launches them through ``cuda.core``. On a ROCm
build of PyTorch that package does not exist. This package provides the same names on top of
hipRTC and the HIP runtime that PyTorch already loaded, so the kernel backend runs unchanged.

:func:`install` registers an import finder that answers ``cuda``, ``cuda.core``,
``cuda.pathfinder``, ``cuda.bindings`` and ``cuda.bindings.runtime`` only when no installed
``cuda`` package exists and PyTorch is a ROCm build. An installed ``cuda`` package always takes
precedence, as a whole: its missing submodules are not filled in from here.
"""

# Standard Library
import importlib.abc
import importlib.machinery
import sys

_PREFIX = __name__ + "._"
_PROVIDED = {
    "cuda": _PREFIX + "cuda",
    "cuda.core": _PREFIX + "core",
    "cuda.pathfinder": _PREFIX + "pathfinder",
    "cuda.bindings": _PREFIX + "bindings",
    "cuda.bindings.runtime": _PREFIX + "runtime",
}


def torch_is_rocm() -> bool:
    """Return whether the installed PyTorch is a ROCm (HIP) build."""
    try:
        import torch
    except Exception:  # a PyTorch that cannot be loaded for any reason is not a ROCm build
        return False
    return getattr(getattr(torch, "version", None), "hip", None) is not None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec):
        module = importlib.import_module(self._target)
        return module

    def exec_module(self, module) -> None:
        return None


class _HipCoreFinder(importlib.abc.MetaPathFinder):
    """Resolve the ``cuda`` names to this package when nothing else provides them."""

    def find_spec(self, fullname, path=None, target=None):
        target_name = _PROVIDED.get(fullname)
        if target_name is None:
            return None
        if fullname != "cuda":
            # Serve submodules only below this package's own ``cuda``, never below a real one.
            parent = sys.modules.get("cuda")
            if parent is None or parent.__name__ != _PROVIDED["cuda"]:
                return None
        if not torch_is_rocm():
            return None
        is_package = fullname in ("cuda", "cuda.bindings")
        spec = importlib.machinery.ModuleSpec(
            fullname, _AliasLoader(target_name), is_package=is_package
        )
        return spec


_FINDER = _HipCoreFinder()


def install() -> None:
    """Register the finder after all existing ones, so installed packages win."""
    if _FINDER not in sys.meta_path:
        sys.meta_path.append(_FINDER)


def is_active() -> bool:
    """Return whether ``cuda.core`` currently resolves to this package."""
    module = sys.modules.get("cuda.core")
    return module is not None and module.__name__ == _PROVIDED["cuda.core"]
