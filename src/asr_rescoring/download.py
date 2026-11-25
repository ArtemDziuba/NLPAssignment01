"""Resume official corpus downloads and record observed sizes, hashes, and source headers."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import shutil
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import file_sha256, resolve_path, write_json

LOGGER = logging.getLogger(__name__)


def download_archive(source: dict[str, Any], directory: Path, retries: int = 5) -> dict[str, Any]:
    """Resume a partial file with HTTP ranges; completed files require a matching checksum."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source["filename"]
    metadata_path = destination.with_suffix(destination.suffix + ".json")
    if destination.exists():
        if not metadata_path.exists():
            raise ValueError(f"Existing archive has no provenance manifest: {destination}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["url"] != source["url"] or file_sha256(destination) != metadata["sha256"]:
            raise ValueError(f"Archive provenance/checksum mismatch: {destination}")
        if source.get("sha256") and metadata["sha256"] != source["sha256"]:
            raise ValueError(f"Archive differs from pinned source hash: {destination}")
        LOGGER.info("Reusing verified archive: %s", destination.name)
        return metadata

    partial = destination.with_suffix(destination.suffix + ".part")
    partial_meta = partial.with_suffix(partial.suffix + ".json")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "asr-rescoring-assignment/0.1", "Accept-Encoding": "identity"}
        if offset and partial_meta.exists():
            previous = json.loads(partial_meta.read_text(encoding="utf-8"))
            if previous.get("url") == source["url"]:
                headers["Range"] = f"bytes={offset}-"
                if previous.get("etag"):
                    headers["If-Range"] = previous["etag"]
            else:
                offset = 0
        try:
            request = urllib.request.Request(source["url"], headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError(f"Unexpected HTTP range: {content_range}")
                    total = int(content_range.split("/")[-1])
                    mode = "ab"
                else:
                    offset, mode = 0, "wb"
                    total = int(response.headers.get("Content-Length", 0))
                if shutil.disk_usage(directory).free < max(total - offset, 0) + 2 * 1024**3:
                    raise OSError("Insufficient disk space: preserve at least 2 GiB free")
                metadata = {
                    "domain": source["domain"],
                    "url": source["url"],
                    "filename": source["filename"],
                    "expected_bytes": total,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "checksum_origin": "local SHA256; no publisher checksum supplied",
                }
                write_json(partial_meta, metadata)
                written, last_log = offset, time.monotonic()
                with partial.open(mode) as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        written += len(chunk)
                        if time.monotonic() - last_log > 15:
                            LOGGER.info(
                                "%s: %.1f / %.1f MiB",
                                source["domain"],
                                written / 1024**2,
                                total / 1024**2,
                            )
                            last_log = time.monotonic()
                if total and written != total:
                    raise OSError(f"Incomplete download: {written} / {total} bytes")
                metadata["bytes"] = written
                metadata["sha256"] = file_sha256(partial)
                if source.get("sha256") and metadata["sha256"] != source["sha256"]:
                    raise ValueError(f"Downloaded archive differs from pinned hash: {destination}")
                metadata["downloaded_utc"] = datetime.now(UTC).isoformat()
                # Manifest is committed first; a crash leaves a resumable .part file.
                write_json(metadata_path, metadata)
                partial.replace(destination)
                LOGGER.info(
                    "Downloaded %s: %.1f MiB; SHA256 %s",
                    source["domain"],
                    written / 1024**2,
                    metadata["sha256"],
                )
                return metadata
        except (OSError, urllib.error.URLError) as error:
            if attempt + 1 == retries:
                raise
            LOGGER.warning(
                "%s download attempt %d failed: %s", source["domain"], attempt + 1, error
            )
            time.sleep(min(2**attempt, 15))
    raise RuntimeError("Download retries exhausted")


def download_corpus(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect all configured domains without extracting multi-gigabyte archives."""
    directory = resolve_path(config, "raw_corpus")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["corpus"].get("download_workers", 2)
    ) as pool:
        return list(
            pool.map(
                lambda source: download_archive(source, directory), config["corpus"]["sources"]
            )
        )
