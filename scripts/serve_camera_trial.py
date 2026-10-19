"""Serve the bounded camera trial and its source video on localhost."""
from rinkspace.storage import resolve_asset as _resolve_asset
import argparse
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from rinkspace.web import byte_range


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--trial', default='outputs/camera_classification/trial_010800_013600')
    args = parser.parse_args()
    trial = (ROOT/args.trial).resolve()
    evidence = (_resolve_asset(ROOT, ROOT/'evidence/camera_teaching_20260922')).resolve()
    media = (_resolve_asset(ROOT, ROOT/'data/input/oxXu1avhrvs.mp4')).resolve()
    assert trial.is_relative_to(ROOT)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_HEAD(self): self.do_GET()
        def do_GET(self):
            parsed=urlsplit(self.path)
            if parsed.path=='/': path=trial/'index.html'
            elif parsed.path=='/asset':
                value=parse_qs(parsed.query).get('path',[''])[0]
                path=(ROOT/value).resolve()
                if not (path==media or path.is_relative_to(trial) or path.is_relative_to(evidence)):
                    self.send_error(403);return
            else: self.send_error(404);return
            if not path.is_file(): self.send_error(404);return
            size=path.stat().st_size
            try: start,end=byte_range(self.headers.get('Range'),size)
            except ValueError: self.send_error(416);return
            ranged=bool(self.headers.get('Range'))
            self.send_response(206 if ranged else 200)
            content_type=mimetypes.guess_type(path)[0] or 'application/octet-stream'
            if path.suffix in ('.html','.json','.txt','.md'):content_type+='; charset=utf-8'
            self.send_header('Content-Type',content_type)
            self.send_header('Content-Length',str(end-start+1))
            self.send_header('Accept-Ranges','bytes')
            self.send_header('Cache-Control','no-cache')
            if ranged:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if self.command=='HEAD':return
            try:
                with path.open('rb') as f:
                    f.seek(start);remaining=end-start+1
                    while remaining:
                        chunk=f.read(min(1024*1024,remaining))
                        if not chunk:break
                        self.wfile.write(chunk);remaining-=len(chunk)
            except (BrokenPipeError,ConnectionResetError,ConnectionAbortedError):pass

    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.daemon_threads=True
    print(f'http://127.0.0.1:{server.server_port}/',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
