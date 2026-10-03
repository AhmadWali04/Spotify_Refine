"""python -m src.app [--library PATH] [--port 5000]  ->  http://127.0.0.1:5000"""
import argparse
from pathlib import Path

from src import paths
from src.app.server import create_app

ap = argparse.ArgumentParser()
ap.add_argument("--library", default=str(paths.LIBRARY))
ap.add_argument("--port", type=int, default=5000)
args = ap.parse_args()
app = create_app(Path(args.library))
print(f"Review app: http://127.0.0.1:{args.port}")
app.run(host="127.0.0.1", port=args.port, debug=False)
