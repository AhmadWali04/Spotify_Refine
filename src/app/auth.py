"""Browser login with Spotify (OAuth authorization-code flow) for the web app.

The token is cached in the same .spotify_cache the CLI uses, with the write scopes, so the pull and apply
steps (src.fetch.spotify.client) reuse it without opening a second consent window.
"""
from __future__ import annotations

import os
import secrets
from urllib.parse import urlparse

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

from src import paths
from src.fetch.spotify import WRITE_SCOPES

DEFAULT_REDIRECT = "http://127.0.0.1:8888/callback"


def redirect_uri() -> str:
    load_dotenv(paths.ROOT / ".env")
    return os.environ.get("SPOTIPY_REDIRECT_URI", DEFAULT_REDIRECT)


def redirect_port() -> int:
    return urlparse(redirect_uri()).port or 80


def configured() -> bool:
    load_dotenv(paths.ROOT / ".env")
    return bool(os.environ.get("SPOTIPY_CLIENT_ID") and os.environ.get("SPOTIPY_CLIENT_SECRET"))


def cache_path():
    return paths.ROOT / ".spotify_cache"


def oauth(state: str | None = None) -> SpotifyOAuth:
    if not configured():
        raise RuntimeError("Set SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in .env (see README, Setup).")
    uri = redirect_uri()
    if "localhost" in uri:
        raise ValueError("Spotify requires 127.0.0.1 in the redirect URI, not localhost.")
    return SpotifyOAuth(scope=WRITE_SCOPES, redirect_uri=uri, state=state, cache_path=str(cache_path()),
                        open_browser=False, show_dialog=True)


class Session:
    """One local user at a time: the app binds to 127.0.0.1 and keeps one library on disk."""

    def __init__(self):
        self.pending_state: str | None = None
        self.profile: dict | None = None

    def login_url(self) -> str:
        self.pending_state = secrets.token_urlsafe(16)
        return oauth(self.pending_state).get_authorize_url()

    def finish(self, code: str, state: str | None) -> dict:
        if not self.pending_state or state != self.pending_state:
            raise ValueError("Login state mismatch. Start the login again.")
        self.pending_state = None
        auth = oauth()
        auth.get_access_token(code, as_dict=False, check_cache=False)
        return self.refresh_profile(auth)

    def refresh_profile(self, auth: SpotifyOAuth | None = None) -> dict | None:
        """Profile of the cached token's user, or None if there is no usable token."""
        if not configured() or not cache_path().exists():
            self.profile = None
            return None
        auth = auth or oauth()
        if not auth.validate_token(auth.cache_handler.get_cached_token()):
            self.profile = None
            return None
        me = spotipy.Spotify(auth_manager=auth, requests_timeout=20).current_user()
        images = me.get("images") or []
        self.profile = {"id": me["id"], "name": me.get("display_name") or me["id"],
                        "image": images[0]["url"] if images else None,
                        "product": me.get("product")}
        return self.profile

    def user(self) -> dict | None:
        if self.profile is None and cache_path().exists():
            try:
                self.refresh_profile()
            except Exception:                       # network down or token revoked: treat as logged out
                self.profile = None
        return self.profile

    def logout(self) -> None:
        cache_path().unlink(missing_ok=True)
        self.profile = None
