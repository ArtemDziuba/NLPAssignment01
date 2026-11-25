"""Exercise downloads and resumption with in-memory HTTP responses, without network."""

import io
import json

import pytest

from asr_rescoring.download import download_archive


class Response(io.BytesIO):
    def __init__(self, payload, status=200, headers=None):
        super().__init__(payload)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(payload)), "ETag": '"v1"'}


def source():
    return {"domain": "fixture", "filename": "fixture.bz2", "url": "https://example.test/archive"}


def test_download_reuse_and_corruption(monkeypatch, tmp_path):
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        return Response(b"test-payload")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    metadata = download_archive(source(), tmp_path)
    assert metadata["bytes"] == 12
    assert (tmp_path / "fixture.bz2").read_bytes() == b"test-payload"
    assert download_archive(source(), tmp_path)["sha256"] == metadata["sha256"]
    assert len(calls) == 1
    (tmp_path / "fixture.bz2").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        download_archive(source(), tmp_path)


def test_resume_validates_range_and_etag(monkeypatch, tmp_path):
    (tmp_path / "fixture.bz2.part").write_bytes(b"abc")
    (tmp_path / "fixture.bz2.part.json").write_text(
        json.dumps({"url": source()["url"], "etag": '"v1"'}), encoding="utf-8"
    )

    def urlopen(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        assert request.get_header("If-range") == '"v1"'
        return Response(b"def", 206, {"Content-Range": "bytes 3-5/6", "ETag": '"v1"'})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    assert download_archive(source(), tmp_path)["bytes"] == 6
    assert (tmp_path / "fixture.bz2").read_bytes() == b"abcdef"


def test_server_ignoring_range_restarts_cleanly(monkeypatch, tmp_path):
    (tmp_path / "fixture.bz2.part").write_bytes(b"stale")
    (tmp_path / "fixture.bz2.part.json").write_text(
        json.dumps({"url": source()["url"], "etag": '"old"'}), encoding="utf-8"
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: Response(b"new"))
    download_archive(source(), tmp_path)
    assert (tmp_path / "fixture.bz2").read_bytes() == b"new"
