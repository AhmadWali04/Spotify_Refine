"""python -m src.app [--library PATH] [--port PORT]  ->  http://127.0.0.1:8888

The default port is the one in SPOTIPY_REDIRECT_URI (8888), because Spotify sends the browser
back to that address after login and this app serves the callback.
"""
import argparse
from pathlib import Path

from src import paths
from src.app import auth
from src.app.server import create_app

ap = argparse.ArgumentParser()
ap.add_argument("--library", default=str(paths.LIBRARY))
ap.add_argument("--port", type=int, default=auth.redirect_port())
args = ap.parse_args()
app = create_app(Path(args.library))
if args.port != auth.redirect_port():
    print(f"Note: Spotify login redirects to {auth.redirect_uri()}, not port {args.port}; "
          "log in won't come back here unless you change SPOTIPY_REDIRECT_URI to match.")
print(f"Spotify Refine: http://127.0.0.1:{args.port}")
app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
