# SPDX-FileCopyrightText: Copyright (c) 2026 Tomas Thoresen
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the defaults of the runtime flags."""

from __future__ import annotations

import importlib.util

import pytest
import torch

import curobo._src.runtime as curobo_runtime


def _load_fresh_runtime():
    """Execute the runtime module again and return the new module object.

    The test session changes the flags of the imported module, so its defaults
    are read from a second copy that is not registered in ``sys.modules``.
    """
    spec = importlib.util.spec_from_file_location("_curobo_runtime_copy", curobo_runtime.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("hip_version", "expected"),
    [(None, True), ("7.14.0", False)],
    ids=["no-hip-version", "rocm-build"],
)
def test_torch_compile_default_follows_the_pytorch_build(
    hip_version: str | None, expected: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``torch_compile`` defaults to off on a ROCm build of PyTorch and to on otherwise."""
    monkeypatch.setattr(torch.version, "hip", hip_version, raising=False)
    assert _load_fresh_runtime().torch_compile is expected


def test_torch_compile_default_without_a_hip_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    """A PyTorch build whose version module has no ``hip`` attribute keeps the default on."""
    monkeypatch.delattr(torch.version, "hip", raising=False)
    assert _load_fresh_runtime().torch_compile is True


def test_torch_compile_default_can_be_overridden() -> None:
    """Assigning the flag after import replaces the default."""
    module = _load_fresh_runtime()
    module.torch_compile = True
    assert module.torch_compile is True
    module.torch_compile = False
    assert module.torch_compile is False
