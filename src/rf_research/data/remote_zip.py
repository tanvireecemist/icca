from __future__ import annotations

import binascii
import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class RemoteZipEntry:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    crc32: int
    local_header_offset: int


class RemoteZip:
    """Read and selectively extract a non-ZIP64 remote ZIP using HTTP ranges."""

    def __init__(
        self,
        url: str,
        session: requests.Session | None = None,
        timeout: tuple[int, int] = (30, 300),
    ) -> None:
        self.url = url
        if session is None:
            session = requests.Session()
            retry = Retry(
                total=12,
                connect=6,
                read=6,
                status=12,
                backoff_factor=1.0,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET"}),
                respect_retry_after_header=True,
            )
            session.mount(
                "https://",
                HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4),
            )
        self.session = session
        self.timeout = timeout
        self.headers = {"User-Agent": "real-rf-dg/0.1 (academic reproducibility)"}
        self.entries = self._read_directory()

    def _range(self, start: int, end: int | None = None) -> bytes:
        headers = dict(self.headers)
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
        response = self.session.get(
            self.url, headers=headers, timeout=self.timeout, stream=False
        )
        response.raise_for_status()
        if response.status_code != 206:
            raise RuntimeError(
                f"Server did not honor HTTP Range for {self.url}: "
                f"status {response.status_code}"
            )
        return response.content

    def _read_directory(self) -> list[RemoteZipEntry]:
        tail = self._range(-4 * 1024 * 1024)
        eocd_offset = tail.rfind(b"PK\x05\x06")
        if eocd_offset < 0:
            raise ValueError(f"ZIP end-of-central-directory not found: {self.url}")
        eocd = struct.unpack_from("<4s4H2LH", tail, eocd_offset)
        entry_count = eocd[4]
        directory_size = eocd[5]
        directory_offset = eocd[6]
        if entry_count == 0xFFFF or directory_offset == 0xFFFFFFFF:
            raise ValueError("ZIP64 archives are not supported by selective fetch")

        directory = self._range(
            directory_offset, directory_offset + directory_size - 1
        )
        entries: list[RemoteZipEntry] = []
        cursor = 0
        while cursor + 46 <= len(directory):
            if directory[cursor : cursor + 4] != b"PK\x01\x02":
                break
            values = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
            name_length = values[10]
            extra_length = values[11]
            comment_length = values[12]
            name_bytes = directory[cursor + 46 : cursor + 46 + name_length]
            flags = values[3]
            encoding = "utf-8" if flags & 0x800 else "cp437"
            name = name_bytes.decode(encoding)
            entries.append(
                RemoteZipEntry(
                    name=name,
                    compression=values[4],
                    crc32=values[7],
                    compressed_size=values[8],
                    uncompressed_size=values[9],
                    local_header_offset=values[16],
                )
            )
            cursor += 46 + name_length + extra_length + comment_length
        if len(entries) != entry_count:
            raise ValueError(
                f"Parsed {len(entries)} ZIP entries but expected {entry_count}"
            )
        return entries

    def _data_offset(self, entry: RemoteZipEntry) -> int:
        header = self._range(entry.local_header_offset, entry.local_header_offset + 29)
        values = struct.unpack("<4s5H3L2H", header)
        if values[0] != b"PK\x03\x04":
            raise ValueError(f"Invalid local ZIP header for {entry.name}")
        return entry.local_header_offset + 30 + values[9] + values[10]

    @staticmethod
    def _decompressor(compression: int):
        if compression == 0:
            return None
        if compression == 8:
            return zlib.decompressobj(-zlib.MAX_WBITS)
        raise ValueError(f"Unsupported ZIP compression method: {compression}")

    def extract(
        self,
        entry: RemoteZipEntry,
        destination: Path,
        chunk_size: int = 8 * 1024 * 1024,
    ) -> dict[str, str | int]:
        return self.extract_prefix(
            entry,
            destination,
            uncompressed_bytes=entry.uncompressed_size,
            chunk_size=chunk_size,
            verify_complete=True,
        )

    def extract_prefix(
        self,
        entry: RemoteZipEntry,
        destination: Path,
        uncompressed_bytes: int,
        chunk_size: int = 8 * 1024 * 1024,
        first_chunk_size: int | None = None,
        verify_complete: bool = False,
    ) -> dict[str, str | int]:
        if not 0 < uncompressed_bytes <= entry.uncompressed_size:
            raise ValueError("Requested prefix length is outside the ZIP entry")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        partial.unlink(missing_ok=True)

        data_offset = self._data_offset(entry)
        decompressor = self._decompressor(entry.compression)
        sha256 = hashlib.sha256()
        crc = 0
        written = 0
        transferred = 0
        compressed_cursor = 0
        try:
            with partial.open("wb") as output:
                while (
                    written < uncompressed_bytes
                    and compressed_cursor < entry.compressed_size
                ):
                    requested_chunk = (
                        first_chunk_size
                        if compressed_cursor == 0 and first_chunk_size is not None
                        else chunk_size
                    )
                    length = min(
                        requested_chunk, entry.compressed_size - compressed_cursor
                    )
                    chunk = self._range(
                        data_offset + compressed_cursor,
                        data_offset + compressed_cursor + length - 1,
                    )
                    compressed_cursor += len(chunk)
                    transferred += len(chunk)
                    remaining = uncompressed_bytes - written
                    if decompressor is None:
                        decoded = chunk[:remaining]
                    else:
                        decoded = decompressor.decompress(chunk, remaining)
                    if decoded:
                        output.write(decoded)
                        sha256.update(decoded)
                        crc = binascii.crc32(decoded, crc)
                        written += len(decoded)
                if written != uncompressed_bytes:
                    raise EOFError(
                        f"Only decoded {written} of {uncompressed_bytes} bytes "
                        f"for {entry.name}"
                    )
            if verify_complete:
                if written != entry.uncompressed_size:
                    raise ValueError(f"Incomplete extraction of {entry.name}")
                if crc & 0xFFFFFFFF != entry.crc32:
                    raise ValueError(f"CRC32 mismatch for {entry.name}")
            partial.replace(destination)
        except Exception:
            partial.unlink(missing_ok=True)
            raise

        return {
            "entry": entry.name,
            "remote_compressed_size": entry.compressed_size,
            "remote_uncompressed_size": entry.uncompressed_size,
            "downloaded_compressed_bytes": transferred,
            "saved_uncompressed_bytes": written,
            "sha256": sha256.hexdigest(),
            "crc32": f"{crc & 0xFFFFFFFF:08x}",
            "complete_entry": int(verify_complete),
        }


def write_jsonl_record(handle: BinaryIO, record: dict) -> None:
    import json

    handle.write((json.dumps(record, sort_keys=True) + "\n").encode())
