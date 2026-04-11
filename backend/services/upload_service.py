"""Staging service for browser-side file uploads.

Files are written under ``<sessions_dir>/.uploads/<group_id>/<filename>``.
A group represents a single NewSession draft and may contain multiple files
(e.g. an IR ``.xml`` + ``.bin`` pair, or several input tensors). Groups are
created server-side; clients must reuse a known group_id to add more files.

A background sweeper deletes groups whose mtime is older than ``ttl_seconds``.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Awaitable, Callable, Iterable, Optional, Union


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_FORBIDDEN = {"", ".", ".."}
_MAX_FILENAME = 200


@dataclass
class StagedFile:
    staged_path: str
    original_filename: str
    size: int
    group_id: str


class UploadError(Exception):
    """Base class for upload errors. ``status`` mirrors the desired HTTP code."""

    status: int = 400

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


class UploadService:
    """Manage staged uploads under <sessions_dir>/.uploads/."""

    def __init__(
        self,
        sessions_dir: Path,
        max_upload_bytes: int = 5 << 30,
        max_group_bytes: int = 10 << 30,
        ttl_seconds: int = 6 * 3600,
    ) -> None:
        self.sessions_dir = Path(sessions_dir)
        self.uploads_dir = self.sessions_dir / ".uploads"
        self.max_upload_bytes = max_upload_bytes
        self.max_group_bytes = max_group_bytes
        self.ttl_seconds = ttl_seconds
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ utils

    @staticmethod
    def sanitize_filename(name: str) -> str:
        """Return a safe basename or raise ``UploadError`` on bad input."""
        if not isinstance(name, str):
            raise UploadError("Filename must be a string")
        # Strip any directory components — only the basename ever lands on disk.
        base = os.path.basename(name.replace("\\", "/"))
        if base in _FORBIDDEN:
            raise UploadError(f"Invalid filename: {name!r}")
        if base.startswith("."):
            raise UploadError(f"Hidden filenames are not allowed: {name!r}")
        if any(ch in base for ch in ("\x00", "/", "\\")):
            raise UploadError(f"Filename contains forbidden characters: {name!r}")
        if any(ord(ch) < 32 for ch in base):
            raise UploadError(f"Filename contains control characters: {name!r}")
        if len(base.encode("utf-8")) > _MAX_FILENAME:
            raise UploadError(f"Filename exceeds {_MAX_FILENAME} bytes: {name!r}")
        return base

    def _group_dir(self, group_id: str) -> Path:
        if not _UUID_RE.match(group_id):
            raise UploadError(f"Malformed group_id: {group_id!r}")
        return self.uploads_dir / group_id

    def create_group(self) -> str:
        gid = str(uuid.uuid4())
        (self.uploads_dir / gid).mkdir(parents=True, exist_ok=False)
        return gid

    def get_or_create_group(self, group_id: Optional[str]) -> str:
        if not group_id:
            return self.create_group()
        gdir = self._group_dir(group_id)
        if not gdir.exists():
            raise UploadError(f"Unknown group_id: {group_id}", status=404)
        return group_id

    def list_groups(self) -> list[dict]:
        if not self.uploads_dir.exists():
            return []
        out: list[dict] = []
        for entry in sorted(self.uploads_dir.iterdir()):
            if not entry.is_dir() or not _UUID_RE.match(entry.name):
                continue
            files = []
            total = 0
            for f in entry.iterdir():
                if f.is_file():
                    sz = f.stat().st_size
                    total += sz
                    files.append({"name": f.name, "size": sz})
            out.append({
                "group_id": entry.name,
                "files": files,
                "total_size": total,
                "mtime": entry.stat().st_mtime,
            })
        return out

    def delete_group(self, group_id: str) -> None:
        gdir = self._group_dir(group_id)
        if not gdir.exists():
            raise UploadError(f"Unknown group_id: {group_id}", status=404)
        shutil.rmtree(gdir, ignore_errors=False)

    # ----------------------------------------------------------------- write

    def _resolve_target(self, group_id: str, filename: str) -> Path:
        """Return absolute target path inside ``group_id`` after traversal check."""
        group_dir = self._group_dir(group_id).resolve()
        target = (group_dir / filename).resolve()
        try:
            target.relative_to(group_dir)
        except ValueError as exc:
            raise UploadError(f"Path traversal rejected: {filename!r}") from exc
        return target

    def _check_disk_space(self, required: int) -> None:
        try:
            free = shutil.disk_usage(self.sessions_dir).free
        except OSError:
            return
        if free < int(required * 1.1):
            raise UploadError(
                f"Insufficient disk space (need ~{required} bytes, free {free}).",
                status=507,
            )

    async def stage_upload(
        self,
        *,
        filename: str,
        reader: Callable[[int], Awaitable[bytes]],
        group_id: Optional[str],
        content_length: Optional[int] = None,
        chunk_size: int = 1 << 20,
    ) -> StagedFile:
        """Stream bytes from ``reader`` (e.g. UploadFile.read) to disk.

        ``reader`` is an async callable taking a max byte count and returning
        the next chunk; an empty bytes value signals EOF.
        """
        sanitized = self.sanitize_filename(filename)

        if content_length is not None:
            if content_length < 0:
                raise UploadError("Negative content length")
            if content_length > self.max_upload_bytes:
                raise UploadError(
                    f"File too large: {content_length} > {self.max_upload_bytes} bytes",
                    status=413,
                )
            self._check_disk_space(content_length)

        gid = self.get_or_create_group(group_id)
        target = self._resolve_target(gid, sanitized)

        # Enforce per-group quota by including bytes already on disk.
        existing_total = sum(
            f.stat().st_size for f in self._group_dir(gid).iterdir() if f.is_file()
        )

        written = 0
        try:
            with open(target, "wb") as fh:
                while True:
                    chunk = await reader(chunk_size)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > self.max_upload_bytes:
                        raise UploadError(
                            f"File exceeded max_upload_bytes ({self.max_upload_bytes})",
                            status=413,
                        )
                    if existing_total + written > self.max_group_bytes:
                        raise UploadError(
                            f"Group exceeded max_group_bytes ({self.max_group_bytes})",
                            status=413,
                        )
                    fh.write(chunk)
        except UploadError:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        except OSError as exc:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            raise UploadError(f"Write failed: {exc}", status=500) from exc

        try:
            os.chmod(target, 0o644)
        except OSError:
            pass

        # Bump the group dir mtime so the TTL sweeper sees recent activity.
        try:
            os.utime(self._group_dir(gid), None)
        except OSError:
            pass

        return StagedFile(
            staged_path=str(target),
            original_filename=sanitized,
            size=written,
            group_id=gid,
        )

    # ------------------------------------------------------------ TTL sweep

    def sweep_expired(self, now: Optional[float] = None) -> int:
        """Delete groups whose mtime is older than ``ttl_seconds``. Returns count."""
        if not self.uploads_dir.exists():
            return 0
        cutoff = (now if now is not None else time.time()) - self.ttl_seconds
        removed = 0
        for entry in self.uploads_dir.iterdir():
            if not entry.is_dir() or not _UUID_RE.match(entry.name):
                continue
            try:
                mtime = entry.stat().st_mtime
            except OSError:
                continue
            if mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
                removed += 1
        return removed

    async def run_sweeper(self, interval_seconds: int = 1800) -> None:
        """Background task: sweep expired groups every ``interval_seconds``."""
        while True:
            try:
                self.sweep_expired()
            except Exception as exc:  # pragma: no cover - defensive
                print(f"upload sweeper error: {exc}")
            await asyncio.sleep(interval_seconds)

    # -------------------------------------------------------- adoption helper

    def adopt_into_session(self, group_id: str, session_dir: Path) -> Path:
        """Move ``group_id`` into ``session_dir/uploaded/`` and return the new dir."""
        gdir = self._group_dir(group_id)
        if not gdir.exists():
            raise UploadError(f"Unknown group_id: {group_id}", status=404)
        target_root = Path(session_dir) / "uploaded"
        target_root.mkdir(parents=True, exist_ok=True)
        target = target_root / group_id
        shutil.move(str(gdir), str(target))
        return target
