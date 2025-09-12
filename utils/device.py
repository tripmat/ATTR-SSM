"""
Device selection utilities to optionally run on GPU without changing defaults.

Default remains CPU; passing a flag can opt into CUDA/MPS when available.
"""

from typing import Literal
import torch


DeviceArg = Literal["cpu", "cuda", "mps", "auto"]


def resolve_device(preference: DeviceArg = "cpu") -> str:
    """Resolve a device preference into an available torch device string.

    - "cpu": always returns "cpu"
    - "cuda": returns "cuda" if available else falls back to "cpu"
    - "mps": returns "mps" if available else falls back to "cpu"
    - "auto": prefer CUDA > MPS > CPU
    """
    pref = (preference or "cpu").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        mps_ok = getattr(torch.backends, "mps", None)
        return "mps" if (mps_ok and mps_ok.is_available()) else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    mps_ok = getattr(torch.backends, "mps", None)
    if mps_ok and mps_ok.is_available():
        return "mps"
    return "cpu"


def device_summary(device: str) -> str:
    """Return a short human-readable summary for the selected device."""
    if device == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
            return f"cuda ({name})"
        except Exception:
            return "cuda"
    return device

