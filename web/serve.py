"""
serve.py — server lokal untuk demo web, dengan dukungan HTTP Range.

`python -m http.server` tidak mendukung Range, sehingga <video> tidak bisa di-seek
(timeline macet di 0 s). Server ini menambahkan dukungan itu.

Pemakaian:  python web/serve.py [port]      lalu buka http://localhost:8000/standalone.html
"""
import http.server
import os
import re
import sys


class RangeHandler(http.server.SimpleHTTPRequestHandler):
    _remaining = None

    def send_head(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start, end, status = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range", "").strip())
        if m and (m.group(1) or m.group(2)):
            status = 206
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:                                   # "bytes=-N": N byte terakhir
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_error(416, "Range Not Satisfiable")
                return None
        f = open(path, "rb")
        f.seek(start)
        self.send_response(status)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self._remaining = end - start + 1
        return f

    def copyfile(self, source, outputfile):
        n = self._remaining
        while n and n > 0:
            chunk = source.read(min(1 << 16, n))
            if not chunk:
                break
            outputfile.write(chunk)
            n -= len(chunk)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Demo: http://localhost:{port}/standalone.html  (Ctrl+C untuk berhenti)")
    http.server.ThreadingHTTPServer(("", port), RangeHandler).serve_forever()
