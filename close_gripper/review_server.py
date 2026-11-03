"""Read-only review files and an explicit MP4 allowlist with HTTP byte ranges."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

MEDIA_SCHEMA = "close_gripper_video_media_v1"
MEDIA_URL = re.compile(r"/media/[0-9a-f]{64}\.mp4\Z")


def source_stat(path):
    value = Path(path).stat()
    return dict(size_bytes=value.st_size, mtime_ns=value.st_mtime_ns, device=value.st_dev, inode=value.st_ino)


def safe_child(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("review paths must be relative without ..")
    candidate = root / path
    if any(p.is_symlink() for p in (candidate, *candidate.parents)) or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("review paths cannot escape the root or traverse symlinks")
    return candidate


def byte_range(value, size):
    """Return an inclusive single byte range; reject malformed/unsatisfiable ones."""
    if value is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", value.strip())
    if not match or size <= 0 or not any(match.groups()):
        raise ValueError("unsupported byte range")
    first, last = match.groups()
    if not first:
        count = int(last)
        if count <= 0:
            raise ValueError("empty suffix byte range")
        return max(0, size - count), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, end


def load_media_manifest(root):
    manifest = json.loads(safe_child(root, "media_manifest.json").read_text())
    if manifest.get("schema") != MEDIA_SCHEMA or not isinstance(manifest.get("media"), dict):
        raise ValueError("invalid video media manifest")
    media = {}
    for url, entry in manifest["media"].items():
        if not MEDIA_URL.fullmatch(url) or not isinstance(entry, dict):
            raise ValueError("invalid allowlisted media URL")
        path = Path(entry.get("source_path", ""))
        if not path.is_absolute() or path.suffix.lower() != ".mp4" or not path.is_file():
            raise ValueError("allowlisted video must be an existing absolute MP4 path")
        if path != path.resolve() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("allowlisted video must use its real path without symlinks")
        if source_stat(path) != {key: entry.get(key) for key in ("size_bytes", "mtime_ns", "device", "inode")}:
            raise ValueError("allowlisted video changed since report preparation")
        media[url] = dict(entry)
    return media


def make_review_server(root, host="127.0.0.1", port=8766):
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("review root must not traverse symlinks")
    root = root.resolve(strict=True)
    media = load_media_manifest(root)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def do_GET(self):
            self.serve(False)

        def do_HEAD(self):
            self.serve(True)

        def empty(self, code, **headers):
            self.send_response(code)
            self.send_header("Content-Length", "0")
            for name, value in headers.items():
                self.send_header(name.replace("_", "-"), str(value))
            self.end_headers()

        def serve(self, head):
            request = urlsplit(self.path)
            if request.scheme or request.netloc:
                self.empty(400)
                return
            path = unquote(request.path)
            if "\x00" in path or "\\" in path:
                self.empty(404)
                return
            entry = None
            if path.startswith("/media/"):
                entry = media.get(path)
                if entry is None:
                    self.empty(404)
                    return
                file = Path(entry["source_path"])
                content_type = "video/mp4"
            else:
                relative = path.lstrip("/")
                if not relative or path.endswith("/"):
                    relative += "review.html"
                try:
                    file = safe_child(root, relative)
                except (ValueError, OSError):
                    self.empty(404)
                    return
                content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            try:
                if not file.is_file() or file.is_symlink():
                    self.empty(404)
                    return
                with file.open("rb") as stream:
                    stat = os.fstat(stream.fileno())
                    if entry is not None and (stat.st_size, stat.st_mtime_ns, stat.st_dev, stat.st_ino) != (
                            entry["size_bytes"], entry["mtime_ns"], entry["device"], entry["inode"]):
                        self.empty(409)
                        return
                    try:
                        selected = byte_range(self.headers.get("Range"), stat.st_size)
                    except ValueError:
                        self.empty(416, Content_Range=f"bytes */{stat.st_size}", Accept_Ranges="bytes")
                        return
                    start, end = selected if selected is not None else (0, stat.st_size - 1)
                    remaining = max(0, end - start + 1)
                    self.send_response(206 if selected is not None else 200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(remaining))
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("X-Content-Type-Options", "nosniff")
                    if selected is not None:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
                    self.end_headers()
                    if not head:
                        stream.seek(start)
                        while remaining:
                            block = stream.read(min(1024 * 1024, remaining))
                            if not block:
                                break
                            self.wfile.write(block)
                            remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except OSError:
                self.empty(404)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
