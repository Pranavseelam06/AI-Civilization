"""Serve only preview files and GLB assets on localhost (no paid services)."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / 'visualization' / 'model_preview'
MODELS = ROOT / 'assets' / 'models'


class PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        routes = {
            '/': (PREVIEW / 'index.html', 'text/html; charset=utf-8'),
            '/preview.js': (PREVIEW / 'preview.js', 'text/javascript; charset=utf-8'),
        }
        for name in ('man', 'woman', 'boy', 'girl', 'grass_tile', 'berry_bush'):
            routes[f'/models/{name}.glb'] = (MODELS / f'{name}.glb', 'model/gltf-binary')
        if path not in routes:
            self.send_error(404)
            return
        file, content_type = routes[path]
        try:
            data = file.read_bytes()
        except OSError:
            self.send_error(404, 'Preview asset missing')
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    with ThreadingHTTPServer(('127.0.0.1', args.port), PreviewHandler) as server:
        print(f'Open http://127.0.0.1:{args.port} — Ctrl+C to stop.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
