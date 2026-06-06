from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from rf_research.data.remote_zip import RemoteZip
from rf_research.data.subset_fetch import _safe_destination


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.status_code = 206

    def raise_for_status(self) -> None:
        return None


class _FakeRangeSession:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def get(self, url: str, headers: dict, timeout: tuple, stream: bool):
        requested = headers["Range"].removeprefix("bytes=")
        if requested.startswith("-"):
            length = abs(int(requested.rstrip("-")))
            result = self.content[-length:]
        else:
            start_text, end_text = requested.split("-", 1)
            start = int(start_text)
            end = int(end_text) if end_text else len(self.content) - 1
            result = self.content[start : end + 1]
        return _FakeResponse(result)


def _archive_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("signals/a.sc16", b"abcdefgh" * 100)
        bundle.writestr("signals/b.bin", bytes(range(256)) * 20)
    return output.getvalue()


def test_remote_zip_extracts_and_checks_complete_entry(tmp_path: Path) -> None:
    content = _archive_bytes()
    remote = RemoteZip("https://example.test/data.zip", _FakeRangeSession(content))
    entry = next(item for item in remote.entries if item.name.endswith("a.sc16"))
    destination = tmp_path / "a.sc16"
    record = remote.extract(entry, destination, chunk_size=11)
    assert destination.read_bytes() == b"abcdefgh" * 100
    assert record["saved_uncompressed_bytes"] == 800
    assert record["complete_entry"] == 1


def test_remote_zip_stops_after_requested_prefix(tmp_path: Path) -> None:
    content = _archive_bytes()
    remote = RemoteZip("https://example.test/data.zip", _FakeRangeSession(content))
    entry = next(item for item in remote.entries if item.name.endswith("b.bin"))
    destination = tmp_path / "b.bin"
    record = remote.extract_prefix(
        entry,
        destination,
        uncompressed_bytes=513,
        chunk_size=13,
    )
    expected = (bytes(range(256)) * 20)[:513]
    assert destination.read_bytes() == expected
    assert record["saved_uncompressed_bytes"] == 513
    assert record["complete_entry"] == 0


def test_safe_destination_rejects_windows_and_posix_traversal(
    tmp_path: Path,
) -> None:
    root = tmp_path / "extract"
    safe = _safe_destination(root, "signals/device/capture.bin")
    assert safe == root / "signals" / "device" / "capture.bin"

    for member in (
        "../escape.bin",
        "signals/../../escape.bin",
        r"signals\..\escape.bin",
        "C:/escape.bin",
    ):
        with pytest.raises(ValueError):
            _safe_destination(root, member)
