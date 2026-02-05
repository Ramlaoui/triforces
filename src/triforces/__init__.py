from __future__ import annotations

try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("triforces")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except Exception:  # pragma: no cover
    __version__ = "0.1.0"
