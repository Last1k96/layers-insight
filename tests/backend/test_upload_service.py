"""Unit tests for UploadService."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from backend.services.upload_service import UploadError, UploadService


def make_reader(payload: bytes):
    """Build a reader callable matching UploadFile.read(size) semantics."""
    pos = 0

    async def _read(size: int) -> bytes:
        nonlocal pos
        chunk = payload[pos : pos + size]
        pos += len(chunk)
        return chunk

    return _read


@pytest.fixture
def svc(tmp_path):
    return UploadService(sessions_dir=tmp_path / "sessions")


class TestSanitizeFilename:
    def test_basic(self, svc):
        assert svc.sanitize_filename("model.xml") == "model.xml"

    def test_strips_directory(self, svc):
        assert svc.sanitize_filename("dir/sub/model.xml") == "model.xml"

    def test_strips_backslash(self, svc):
        assert svc.sanitize_filename("dir\\sub\\model.xml") == "model.xml"

    def test_rejects_dotdot(self, svc):
        with pytest.raises(UploadError):
            svc.sanitize_filename("..")

    def test_rejects_empty(self, svc):
        with pytest.raises(UploadError):
            svc.sanitize_filename("")

    def test_rejects_hidden(self, svc):
        with pytest.raises(UploadError):
            svc.sanitize_filename(".hidden")

    def test_rejects_null_byte(self, svc):
        with pytest.raises(UploadError):
            svc.sanitize_filename("foo\x00bar")

    def test_rejects_control_char(self, svc):
        with pytest.raises(UploadError):
            svc.sanitize_filename("foo\nbar")

    def test_rejects_too_long(self, svc):
        with pytest.raises(UploadError):
            svc.sanitize_filename("x" * 250)


class TestGroupCreation:
    def test_create_group_returns_uuid(self, svc):
        gid = svc.create_group()
        assert (svc.uploads_dir / gid).exists()

    def test_get_or_create_creates_when_none(self, svc):
        gid = svc.get_or_create_group(None)
        assert (svc.uploads_dir / gid).exists()

    def test_get_or_create_rejects_unknown(self, svc):
        with pytest.raises(UploadError) as exc:
            svc.get_or_create_group("00000000-0000-0000-0000-000000000000")
        assert exc.value.status == 404

    def test_get_or_create_rejects_malformed(self, svc):
        with pytest.raises(UploadError):
            svc.get_or_create_group("not-a-uuid")


class TestStageUpload:
    @pytest.mark.asyncio
    async def test_single_file(self, svc):
        payload = b"hello world"
        staged = await svc.stage_upload(
            filename="hello.bin",
            reader=make_reader(payload),
            group_id=None,
            content_length=len(payload),
        )
        assert staged.size == len(payload)
        assert Path(staged.staged_path).read_bytes() == payload
        assert staged.original_filename == "hello.bin"

    @pytest.mark.asyncio
    async def test_two_files_same_group(self, svc):
        first = await svc.stage_upload(
            filename="model.xml",
            reader=make_reader(b"<model/>"),
            group_id=None,
        )
        second = await svc.stage_upload(
            filename="model.bin",
            reader=make_reader(b"\x00" * 32),
            group_id=first.group_id,
        )
        assert first.group_id == second.group_id
        gdir = svc.uploads_dir / first.group_id
        names = sorted(p.name for p in gdir.iterdir())
        assert names == ["model.bin", "model.xml"]

    @pytest.mark.asyncio
    async def test_traversal_filename_is_basenamed(self, svc):
        """`../escape.bin` is reduced to `escape.bin` — confined to the group dir."""
        staged = await svc.stage_upload(
            filename="../escape.bin",
            reader=make_reader(b"x"),
            group_id=None,
        )
        assert staged.original_filename == "escape.bin"
        # Confirm the file lives inside the group dir, not above it.
        target = Path(staged.staged_path).resolve()
        assert (svc.uploads_dir / staged.group_id) in target.parents

    def test_resolve_target_defense(self, svc):
        """Defense-in-depth: _resolve_target rejects unsanitized escapes."""
        gid = svc.create_group()
        with pytest.raises(UploadError):
            svc._resolve_target(gid, "../oops.bin")

    @pytest.mark.asyncio
    async def test_oversize_content_length_rejected(self, tmp_path):
        svc = UploadService(sessions_dir=tmp_path / "sessions", max_upload_bytes=10)
        with pytest.raises(UploadError) as exc:
            await svc.stage_upload(
                filename="big.bin",
                reader=make_reader(b"x" * 100),
                group_id=None,
                content_length=100,
            )
        assert exc.value.status == 413

    @pytest.mark.asyncio
    async def test_oversize_during_stream_rejected(self, tmp_path):
        svc = UploadService(sessions_dir=tmp_path / "sessions", max_upload_bytes=10)
        # Don't pass content_length so the limit is enforced mid-stream.
        with pytest.raises(UploadError) as exc:
            await svc.stage_upload(
                filename="big.bin",
                reader=make_reader(b"x" * 50),
                group_id=None,
                chunk_size=8,
            )
        assert exc.value.status == 413
        # Partial file must be cleaned up.
        groups = list(svc.uploads_dir.iterdir())
        if groups:
            assert all(not f.is_file() or f.stat().st_size == 0 for f in groups[0].iterdir())

    @pytest.mark.asyncio
    async def test_group_quota_rejected(self, tmp_path):
        svc = UploadService(
            sessions_dir=tmp_path / "sessions",
            max_upload_bytes=100,
            max_group_bytes=50,
        )
        gid = svc.create_group()
        await svc.stage_upload(
            filename="a.bin",
            reader=make_reader(b"x" * 30),
            group_id=gid,
        )
        with pytest.raises(UploadError) as exc:
            await svc.stage_upload(
                filename="b.bin",
                reader=make_reader(b"x" * 30),
                group_id=gid,
            )
        assert exc.value.status == 413


class TestListAndDelete:
    @pytest.mark.asyncio
    async def test_list_groups(self, svc):
        await svc.stage_upload(
            filename="a.bin", reader=make_reader(b"abc"), group_id=None
        )
        groups = svc.list_groups()
        assert len(groups) == 1
        assert groups[0]["total_size"] == 3

    @pytest.mark.asyncio
    async def test_delete_group(self, svc):
        staged = await svc.stage_upload(
            filename="a.bin", reader=make_reader(b"abc"), group_id=None
        )
        svc.delete_group(staged.group_id)
        assert svc.list_groups() == []

    def test_delete_unknown_group(self, svc):
        with pytest.raises(UploadError) as exc:
            svc.delete_group("00000000-0000-0000-0000-000000000000")
        assert exc.value.status == 404


class TestSweeper:
    @pytest.mark.asyncio
    async def test_sweep_removes_old(self, tmp_path):
        svc = UploadService(sessions_dir=tmp_path / "sessions", ttl_seconds=60)
        old = await svc.stage_upload(
            filename="old.bin", reader=make_reader(b"x"), group_id=None
        )
        fresh = await svc.stage_upload(
            filename="fresh.bin", reader=make_reader(b"x"), group_id=None
        )
        # Backdate the old group's mtime.
        old_dir = svc.uploads_dir / old.group_id
        backdate = time.time() - 3600
        os.utime(old_dir, (backdate, backdate))

        removed = svc.sweep_expired()
        assert removed == 1
        assert not old_dir.exists()
        assert (svc.uploads_dir / fresh.group_id).exists()


class TestAdoption:
    @pytest.mark.asyncio
    async def test_adopt_into_session(self, svc, tmp_path):
        staged = await svc.stage_upload(
            filename="model.xml", reader=make_reader(b"<model/>"), group_id=None
        )
        session_dir = tmp_path / "session_abc"
        session_dir.mkdir()
        target = svc.adopt_into_session(staged.group_id, session_dir)
        assert (target / "model.xml").read_bytes() == b"<model/>"
        # Original group dir should be gone.
        assert not (svc.uploads_dir / staged.group_id).exists()
