"""HTTP video seeking and the review/static file allowlist."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from close_gripper.review_server import MEDIA_SCHEMA, byte_range, make_review_server, source_stat


class ReviewServerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        self.root = base / "review"
        self.root.mkdir()
        (self.root / "review.html").write_text("<html>review</html>")
        self.video = base / "source.mp4"
        self.video.write_bytes(b"0123456789")
        self.url = "/media/" + "a" * 64 + ".mp4"
        manifest = dict(schema=MEDIA_SCHEMA, media={self.url: dict(source_path=str(self.video), **source_stat(self.video))})
        (self.root / "media_manifest.json").write_text(json.dumps(manifest))
        patcher = patch("close_gripper.review_server.ThreadingHTTPServer")
        server_factory = patcher.start()
        self.addCleanup(patcher.stop)
        make_review_server(self.root, port=0)
        self.handler = server_factory.call_args.args[1]

    def request(self, path, method="GET", headers=None):
        # Exercise the real HTTP parser and handler without opening a socket.
        request = self.handler.__new__(self.handler)
        extra = "".join(f"{name}: {value}\r\n" for name, value in (headers or {}).items())
        request.rfile = io.BytesIO(f"{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n{extra}\r\n".encode())
        request.wfile = io.BytesIO()
        request.client_address = ("127.0.0.1", 0)
        request.close_connection = True
        request.handle_one_request()
        head, body = request.wfile.getvalue().split(b"\r\n\r\n", 1)
        lines = head.decode().split("\r\n")
        return int(lines[0].split()[1]), dict(line.split(": ", 1) for line in lines[1:]), body

    def test_video_get_head_and_ranges_support_seek(self):
        status, headers, body = self.request(self.url)
        self.assertEqual((status, body), (200, b"0123456789"))
        self.assertEqual(headers["Content-Type"], "video/mp4")
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        cases = [("bytes=2-4", b"234", "bytes 2-4/10"), ("bytes=7-", b"789", "bytes 7-9/10"),
                 ("bytes=-3", b"789", "bytes 7-9/10"), ("bytes=7-100", b"789", "bytes 7-9/10")]
        for value, expected, content_range in cases:
            with self.subTest(value=value):
                status, headers, body = self.request(self.url, headers={"Range": value})
                self.assertEqual((status, body), (206, expected))
                self.assertEqual(headers["Content-Range"], content_range)
                self.assertEqual(int(headers["Content-Length"]), len(expected))
        status, headers, body = self.request(self.url, "HEAD", {"Range": "bytes=2-4"})
        self.assertEqual((status, body, headers["Content-Length"]), (206, b"", "3"))
        status, headers, body = self.request(self.url, "HEAD")
        self.assertEqual((status, body, headers["Content-Length"]), (200, b"", "10"))

    def test_bad_and_unsatisfiable_ranges_return_416(self):
        for value in ("bytes=10-", "bytes=5-3", "bytes=-0", "bytes=-", "bytes=0-1,3-4", "items=0-4"):
            with self.subTest(value=value):
                status, headers, body = self.request(self.url, headers={"Range": value})
                self.assertEqual((status, body), (416, b""))
                self.assertEqual(headers["Content-Range"], "bytes */10")
        with self.assertRaises(ValueError):
            byte_range("bytes=0-", 0)

    def test_static_root_and_media_are_not_arbitrary_file_access(self):
        self.assertEqual(self.request("/")[2], b"<html>review</html>")
        paths = ["/../source.mp4", "/%2e%2e/source.mp4", "/%252e%252e/source.mp4", "/source.mp4",
                 "/media/source.mp4", "/media/" + "b" * 64 + ".mp4", "/media/../source.mp4"]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], 404)
        (self.root / "leak.mp4").symlink_to(self.video)
        (self.root / "linked").symlink_to(self.video.parent, target_is_directory=True)
        self.assertEqual(self.request("/leak.mp4")[0], 404)
        self.assertEqual(self.request("/linked/source.mp4")[0], 404)
        self.assertEqual(self.request("/", "POST")[0], 501)

    def test_source_video_changes_are_not_silently_served(self):
        self.video.write_bytes(b"changed video")
        self.assertEqual(self.request(self.url)[0], 409)
        with self.assertRaisesRegex(ValueError, "changed"):
            make_review_server(self.root, port=0)


if __name__ == "__main__":
    unittest.main()
