# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: NOAA Fisheries
"""Minimal local/remote storage helpers for intermediate stores and data inputs.

The recipe executor publishes ``ExecutionContext.temp_dir`` as either a local
``Path`` or a ``StorageLocation`` (for ``gs://`` scratch). This module works on
those values *by duck typing* — it does not import the engine — so ``aa_si_utils``
keeps its optional relationship with ``aa_recipe_manager``. Remote work goes
through fsspec; local work stays on ``pathlib``/``shutil`` exactly as before.

It also provides the input-side helpers used to read ``gs://`` raw folders:
:func:`glob_url` lists them without downloading, and :func:`localized_file`
materializes exactly one remote file at a time in a genuinely local scratch
directory, deleting it on exit.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Iterator

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


def basename(value: Any) -> str:
    """Final path segment of a local path or a remote URL."""
    if is_remote(value):
        return str(value).rstrip("/").rsplit("/", 1)[-1]
    return Path(os.fspath(value)).name


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


def glob_url(
    base: Any,
    pattern: str,
    storage_options: dict[str, Any] | None = None,
) -> list[str]:
    """Sorted full URLs of objects under a remote *base* matching *pattern*.

    ``fs.glob`` strips the protocol from its results, so each match is put back
    together with ``unstrip_protocol`` — callers get URLs they can reopen.
    """
    fs = get_fs(base, storage_options)
    matches = fs.glob(str(base).rstrip("/") + "/" + pattern)
    return sorted(fs.unstrip_protocol(match) for match in matches)


@contextlib.contextmanager
def localized_file(
    url: str,
    storage_options: dict[str, Any] | None = None,
    companion_suffixes: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Download one remote file to a private local scratch dir; delete on exit.

    The scratch dir comes from :func:`tempfile.mkdtemp`, so it is genuinely
    local even when the pipeline's ``temp_dir`` points at a bucket, and it is
    unique per call — concurrent mapped/parallel instances cannot collide.

    The original basename is preserved: echopype derives store names from the
    file stem, and companion files (e.g. ``.bot``) are located by swapping the
    raw file's extension. Any suffix in *companion_suffixes* is downloaded
    alongside the main file when it exists remotely.

    Only the local copy is removed; the remote object is never touched.
    """
    fs = get_fs(url, storage_options)
    scratch = Path(tempfile.mkdtemp(prefix="aa_si_localized_"))
    try:
        local_path = scratch / basename(url)
        fs.get(str(url), str(local_path))

        # Companions (.bot, .idx) sit beside the raw file with its extension
        # swapped; echopype finds them by that convention, not by argument.
        stem, _, _ = str(url).rpartition(".")
        for suffix in companion_suffixes:
            companion_url = stem + suffix
            if fs.exists(companion_url):
                fs.get(companion_url, str(scratch / basename(companion_url)))

        yield local_path
    finally:
        _rmtree_local(scratch)


def _rmtree_local(path: Path) -> None:
    """Recursively remove a local directory, clearing read-only bits (Windows)."""
    def _on_error(func, fpath, _exc_info):
        os.chmod(fpath, stat.S_IWRITE)
        func(fpath)

    shutil.rmtree(path, onerror=_on_error, ignore_errors=False)


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
        _rmtree_local(path)
    else:
        path.unlink()
