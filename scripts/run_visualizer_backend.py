from __future__ import annotations

import http.server
from functools import partial
import socketserver
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISUALIZER_PUBLIC = ROOT / "visualizer" / "public"


def main() -> None:
    port = 8765
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(VISUALIZER_PUBLIC))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"serving {VISUALIZER_PUBLIC} at http://localhost:{port}/")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
