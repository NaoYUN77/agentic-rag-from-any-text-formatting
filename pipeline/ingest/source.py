r"""输入获取: URL / 本地文件 -> SourcePayload。"""

from __future__ import annotations

import hashlib
import ipaddress
import mimetypes
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

from .models import SourcePayload


USER_AGENT = "agentic-rag-format-router/0.1"


def _is_private_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return True
    return False


def detect_mime(data: bytes, declared: str = "", path: Optional[str] = None) -> str:
    declared = (declared or "").split(";", 1)[0].strip().lower()
    if declared and declared != "application/octet-stream":
        return declared
    head = data[:4096].lstrip()
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    low = head.lower()
    if low.startswith(b"<!doctype html") or b"<html" in low:
        return "text/html"
    if head.startswith(b"PK\x03\x04"):
        return "application/zip"
    if path:
        guessed, _ = mimetypes.guess_type(path)
        if guessed:
            return guessed
    return "application/octet-stream"


def load_local(path_value: str) -> SourcePayload:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    mime = detect_mime(data, path=str(path))
    return SourcePayload(
        source_type="local_file",
        uri=str(path),
        final_uri=str(path),
        mime_type=mime,
        content_type=mime,
        data=data,
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        local_path=str(path),
    )


def fetch_url(
    url: str,
    timeout: int = 30,
    max_bytes: int = 20 * 1024 * 1024,
    max_redirects: int = 5,
    allow_private: bool = False,
) -> SourcePayload:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("仅支持 http/https URL")
    if not parsed.hostname:
        raise ValueError("URL 缺少 hostname")
    if not allow_private and _is_private_host(parsed.hostname):
        raise ValueError("拒绝访问 localhost/内网地址")

    session = requests.Session()
    session.max_redirects = max_redirects
    response = session.get(
        url,
        timeout=timeout,
        allow_redirects=True,
        stream=True,
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    response.raise_for_status()
    chunks = []
    total = 0
    for chunk in response.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"响应超过 max_bytes={max_bytes}")
        chunks.append(chunk)
    data = b"".join(chunks)
    content_type = response.headers.get("Content-Type", "")
    mime = detect_mime(data, declared=content_type, path=response.url)
    return SourcePayload(
        source_type="url",
        uri=url,
        final_uri=response.url,
        mime_type=mime,
        content_type=content_type,
        data=data,
        encoding=response.encoding,
        headers={k: v for k, v in response.headers.items()},
        fetched_at=datetime.now(timezone.utc).isoformat(),
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def load_source(
    value: str,
    timeout: int = 30,
    max_bytes: int = 20 * 1024 * 1024,
    allow_private: bool = False,
) -> SourcePayload:
    if value.startswith("http://") or value.startswith("https://"):
        return fetch_url(
            value,
            timeout=timeout,
            max_bytes=max_bytes,
            allow_private=allow_private,
        )
    return load_local(value)
