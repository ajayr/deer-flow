# Crawl4AI logged-in session (local deployment)

Some sites (Reddit, most notably) 403 anonymous traffic from datacenter IPs.
Giving the crawl4ai server a logged-in browser session fixes that for **every**
consumer at once — DeerFlow's `web_fetch` and the hermes skills both go through
this one server.

## Read this first — the blast radius

The session is **server-side and global**. Crawl4AI 0.9 rejects request-supplied
cookies (trust boundary) and `/md` ignores the `add_cookies` hook, so the only
way in is `config.yml -> crawler.browser.kwargs.storage_state`, which applies to
every crawl the server performs, for every caller.

Cookies are domain-scoped by the browser, so only the sites present in the state
file are affected — but for those sites, **anything readable while logged in is
readable by any caller**, including the DeerFlow agent following a URL it found
in injected page content. For Reddit that means DMs (`/message/inbox`), saved
items and private subreddits.

Mitigations, in order of effectiveness:

1. Use a **burner account** for the session rather than your main one.
2. Always narrow the state file with `--domain` (below). A raw browser export
   contains every site you are logged into; without the filter you would hand
   the crawler all of them.
3. Rotate/revoke by logging that session out in the browser.

## Files

| Path | What |
|---|---|
| `~/.crawl4ai/` | Secrets dir, mode 700. **Outside the repo** — the repo root is bind-mounted into the gateway, and a stray `git add` of session cookies would be bad. |
| `~/.crawl4ai/session-state.json` | Playwright storage state (the cookies). Mode 600 + an ACL granting uid 999. |
| `~/.crawl4ai/config.yml` | The server config, copied out of the image and pointed at the state file. Mounted over `/app/config.yml`. |
| `docker/crawl4ai/cookies-to-storage-state.py` | Converter, `cookies.txt` → storage state. No secrets in it, so it lives in the repo. |

## Creating / rotating the session

1. Log into the site in a browser (burner account preferred).
2. Export cookies in **Netscape `cookies.txt`** format (e.g. the "Get
   cookies.txt LOCALLY" extension). Save it somewhere temporary.
3. Convert, filtering to just that site, and grant the container read access:

   ```bash
   python3 ~/code/deer-flow/docker/crawl4ai/cookies-to-storage-state.py \
     ~/Downloads/cookies.txt ~/.crawl4ai/session-state.json \
     --domain reddit.com --grant-uid 999
   rm ~/Downloads/cookies.txt          # don't leave the export lying around
   docker restart crawl4ai             # see "restart" below
   ```

4. Verify (any cookie-reflecting endpoint proves delivery without exposing the
   real cookie value):

   ```bash
   docker exec crawl4ai sh -c 'curl -s -X POST http://localhost:11235/md \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer $CRAWL4AI_API_TOKEN" \
     -d "{\"url\": \"https://www.reddit.com/r/programming/.json\", \"f\": \"raw\"}"' | head -c 300
   ```

   A `{"error": "Internal server error", "correlation_id": ...}` means the crawl
   itself failed — match the id in `docker logs crawl4ai` for the reason
   ("Blocked by anti-bot protection" = the site still refused; a 401 instead
   would mean the *bearer token* is wrong, an unrelated problem).

## Gotchas that cost time

- **`--grant-uid 999` on every rotation.** The container runs as uid 999
  (`appuser`); the state file is mode 600 owned by you (uid 1000). A newly
  written file does not inherit the old file's ACL, so skipping this makes the
  server silently start with no session. (Host-side note: uid 999 happens to be
  the `dnsmasq` account on this host, so a host process running as dnsmasq could
  read the file too. Acceptable here; worth knowing.)
- **Mount `/app/config.yml`, not the `/tmp/project/...` copy.** The image ships a
  full source tree at `/tmp/project/deploy/docker/config.yml` that *looks* like
  the live config but is never read — the server runs from `/app`. Mounting over
  the wrong one fails silently (verified the hard way).
- **Restart after rotating.** The browser pool builds its `BrowserConfig` from
  the config at startup. The state file is dir-mounted (not file-mounted) so an
  atomic rewrite is visible without recreating the container, but a restart is
  what guarantees the running pool picks it up.
- **Sessions expire.** Logging out in the browser, or a password change, kills
  it. Symptom is a silent return to anonymous behaviour (i.e. Reddit 403s
  again), not an obvious error.
