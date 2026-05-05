"""TRUST: trusted multi-view time-series classification."""

__all__ = ["TRUST"]


def __getattr__(name):
    if name == "TRUST":
        from .model import TRUST

        return TRUST
    raise AttributeError(name)
