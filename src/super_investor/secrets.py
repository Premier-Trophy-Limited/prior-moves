"""Centralized secret loader.

Order of precedence:
  1. macOS Keychain (``security find-generic-password -s <slot> -w``)
  2. ``os.environ`` (env-var fallback)

Howard centralizes credentials in macOS Keychain. Adapters MUST call
``get_secret("<slot>", env_var="<ENV_NAME>")`` rather than raw
``os.environ.get`` — see ``feedback_keychain_first_for_secrets.md``.

Slot naming convention (lowercase, hyphen-separated):
  ``nyt-api-key``  ``guardian-api-key``  ``alphavantage-api-key``  …
"""
from __future__ import annotations

import os
import shutil
import subprocess


def get_secret(slot: str, env_var: str | None = None) -> str | None:
    """Return the secret value, trying keychain first, then env.

    Args:
        slot: macOS Keychain service name (e.g. ``"nyt-api-key"``).
        env_var: Optional environment variable to fall back to.

    Returns:
        The secret string, or ``None`` if neither source has it.
    """
    if shutil.which("security"):
        try:
            r = subprocess.run(
                ["security", "find-generic-password", "-s", slot, "-w"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            val = r.stdout.strip()
            if val:
                return val
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        except FileNotFoundError:
            pass
    if env_var:
        v = os.environ.get(env_var)
        if v:
            return v
    return None
