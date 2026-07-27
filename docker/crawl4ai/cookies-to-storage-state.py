#!/usr/bin/env python3
"""Convert a Netscape ``cookies.txt`` export into a Playwright storage-state JSON.

Crawl4AI's Docker server builds its browser from ``config.yml`` ->
``crawler.browser.kwargs``, which accepts Playwright's ``storage_state`` (a path
to a JSON file holding cookies + localStorage). That is the only *server-side*
way to give the crawler a logged-in session: since 0.9 a request body may not
carry cookies, and the ``add_cookies`` hook is ignored by ``/md``.

Usage:
    python3 cookies-to-storage-state.py cookies.txt storage-state.json
    python3 cookies-to-storage-state.py cookies.txt storage-state.json --domain reddit.com

``--domain`` (repeatable) keeps only cookies for that domain and its subdomains.
ALWAYS use it: a raw browser export usually contains every site you are logged
into, and the crawl4ai server sends whatever it holds to whatever it is asked to
fetch. Narrowing the file to one site is what bounds the blast radius.

The output is written with mode 600 -- it contains live session credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Playwright rejects anything outside this set.
_VALID_SAME_SITE = {"Strict", "Lax", "None"}


def _parse_netscape(text: str) -> list[dict]:
    """Parse Netscape cookies.txt into Playwright cookie dicts."""
    cookies: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue

        # "#HttpOnly_" is a real record with a marker prefix; other "#" lines are comments.
        http_only = False
        if line.startswith("#HttpOnly_"):
            http_only = True
            line = line[len("#HttpOnly_") :]
        elif line.startswith("#"):
            continue

        parts = line.split("\t")
        if len(parts) < 7:
            print(f"warn: line {lineno}: expected 7 tab-separated fields, got {len(parts)} -- skipped", file=sys.stderr)
            continue

        domain, _include_subdomains, path, secure, expires, name, value = parts[:7]

        try:
            # Netscape stores 0 for "session cookie"; Playwright wants -1.
            expires_val = float(expires)
            expires_val = -1 if expires_val == 0 else expires_val
        except ValueError:
            expires_val = -1

        cookies.append(
            {
                "name": name,
                "value": value,
                "domain": domain,
                "path": path or "/",
                "expires": expires_val,
                "httpOnly": http_only,
                "secure": secure.upper() == "TRUE",
                # The export format carries no SameSite; Lax is the browser default
                # and is what a normal navigation would send.
                "sameSite": "Lax",
            }
        )
    return cookies


def _matches_domain(cookie_domain: str, wanted: str) -> bool:
    cd = cookie_domain.lstrip(".").lower()
    w = wanted.lstrip(".").lower()
    return cd == w or cd.endswith("." + w)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cookies_txt", type=Path, help="Netscape-format cookies.txt export")
    ap.add_argument("output_json", type=Path, help="storage-state JSON to write (mode 600)")
    ap.add_argument(
        "--domain",
        action="append",
        default=[],
        metavar="DOMAIN",
        help="keep only cookies for DOMAIN and its subdomains (repeatable; strongly recommended)",
    )
    ap.add_argument(
        "--grant-uid",
        type=int,
        default=None,
        metavar="UID",
        help=(
            "after writing, grant UID read access via a POSIX ACL (the crawl4ai container "
            "runs as uid 999, and the file is mode 600 owned by the host user). Re-run this "
            "on every rotation: a fresh file does not inherit the previous file's ACL."
        ),
    )
    args = ap.parse_args()

    cookies = _parse_netscape(args.cookies_txt.read_text(encoding="utf-8", errors="replace"))
    total = len(cookies)

    if args.domain:
        cookies = [c for c in cookies if any(_matches_domain(c["domain"], d) for d in args.domain)]
        print(f"kept {len(cookies)}/{total} cookies for {', '.join(args.domain)}", file=sys.stderr)
    else:
        print(
            f"warn: no --domain filter given; writing all {total} cookies. "
            "The crawl4ai server will offer these to every site it is asked to fetch.",
            file=sys.stderr,
        )

    if not cookies:
        print("error: no cookies matched -- refusing to write an empty storage state", file=sys.stderr)
        return 1

    for c in cookies:
        if c["sameSite"] not in _VALID_SAME_SITE:
            c["sameSite"] = "Lax"

    payload = {"cookies": cookies, "origins": []}

    # Create with 600 from the start so the credentials are never briefly world-readable.
    fd = os.open(args.output_json, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    if args.grant_uid is not None:
        import subprocess

        proc = subprocess.run(
            ["setfacl", "-m", f"u:{args.grant_uid}:r", str(args.output_json)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"error: setfacl failed: {proc.stderr.strip()}", file=sys.stderr)
            return 1
        print(f"granted uid {args.grant_uid} read access via ACL", file=sys.stderr)

    names = sorted({c["name"] for c in cookies})
    print(f"wrote {args.output_json} ({len(cookies)} cookies: {', '.join(names[:8])}{' ...' if len(names) > 8 else ''})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
