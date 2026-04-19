"""Write-only buffer for streaming ZIP generation."""
from __future__ import annotations


class ZipStreamBuffer:
    """Accumulates bytes written by zipfile and lets the caller drain them
    after each entry, so the response can be streamed incrementally.
    zipfile in write mode only needs write() and tell() — no seeking.
    """

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._pos = 0

    def write(self, data: bytes) -> int:
        self._chunks.append(data)
        self._pos += len(data)
        return len(data)

    def tell(self) -> int:
        return self._pos

    def flush(self) -> None:
        pass

    def drain(self) -> bytes:
        data = b"".join(self._chunks)
        self._chunks.clear()
        return data
