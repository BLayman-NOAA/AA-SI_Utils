# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Minimal local/remote storage helpers for exe_temp intermediate stores.

The recipe executor publishes ``ExecutionContext.temp_dir`` as either a local
``Path`` or a ``StorageLocation`` (for ``gs://`` scratch). This module works on
those values *by duck typing* — it does not import the engine — so ``aa_si_utils``
keeps its optional relationship with ``aa_recipe_manager``. Remote work goes
through fsspec; local work stays on ``pathlib``/``shutil`` exactly as before.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
from pathlib import Path
from typing import Any

_URL_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]+://")
_LOCAL_PROTOCOLS = frozenset({"file", "local"})


def is_remote(value: Any) -> bool:
    """True when ``value`` denotes a non-local fsspec URL (or remote location)."""
    if value is None:
        return False
    # A StorageLocation-like object exposes ``is_local``.
    is_local = getattr(value, "is_local", None)
    if isinstance(is_local, bool):
        return not is_local
    if isinstance(value, Path):
        return False
    match = _URL_SCHEME_RE.match(str(value))
    if match is None:
        return False
    scheme = str(value)[: match.end() - 3].lower()
    return scheme not in _LOCAL_PROTOCOLS


def storage_options_of(value: Any) -> dict[str, Any] | None:
    """Return fsspec storage options carried by a location value, or None."""
    opts = getattr(value, "storage_options", None)
    return dict(opts) if opts else None


def join(base: Any, *parts: str) -> Path | str:
    """Join path segments. Returns a ``Path`` for local bases, a URL str for remote."""
    if is_remote(base):
        joined = str(base).rstrip("/")
        for part in parts:
            joined = joined + "/" + str(part).lstrip("/")
        return joined
    path = Path(os.fspath(base))
    for part in parts:
        path = path / part
    return path


def get_fs(url: Any, storage_options: dict[str, Any] | None = None) -> Any:
    """Return the fsspec filesystem for a remote URL."""
    import fsspec.core

    fs, _ = fsspec.core.url_to_fs(str(url), **(storage_options or {}))
    return fs


def makedirs(target: Any, storage_options: dict[str, Any] | None = None) -> None:
    """Create a directory for local targets; a no-op for object stores."""
    if is_remote(target):
        try:
            get_fs(target, storage_options).makedirs(str(target), exist_ok=True)
        except (NotImplementedError, OSError):
            pass
        return
    Path(os.fspath(target)).mkdir(parents=True, exist_ok=True)


def remove_store(target: Any, storage_options: dict[str, Any] | None = None) -> None:
    """Remove a store dir/file (local or remote); missing targets are a no-op."""
    if is_remote(target):
        try:
            get_fs(target, storage_options).rm(str(target), recursive=True)
        except FileNotFoundError:
            pass
        return
    path = Path(os.fspath(target))
    if not path.exists():
        return
    if path.is_dir():
        def _on_error(func, fpath, _exc_info):
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)

        shutil.rmtree(path, onerror=_on_error)
    else:
        path.unlink()
