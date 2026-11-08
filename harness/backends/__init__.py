"""Backend selection."""

from __future__ import annotations

from .base import Backend, Caps, ProfileJob, ProfileResult
from .sim import SimBackend


def make_backend(name: str, gpu: str):
    name = (name or "modal").lower()
    if name == "sim":
        return SimBackend(gpu)
    if name == "modal":
        from .modal_gpu import ModalBackend
        return ModalBackend(gpu)
    raise ValueError(f"unknown backend: {name}")


__all__ = ["Backend", "Caps", "ProfileJob", "ProfileResult", "SimBackend", "make_backend"]
