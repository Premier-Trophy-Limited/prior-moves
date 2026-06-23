"""FT Alphaville scraper via Playwright (with stealth) + ticker-quarter aggregator.

FT Cloudflare bot management blocks curl + naive requests-style scrapes.
Playwright drives a real Chromium with playwright-stealth patches injected
before navigation, which gets past the bot challenge with the right cookies.

Setup flow:
  1. User runs scripts/ft_login.py once. That opens a NON-headless Chromium
     with persistent profile at data/playwright_profile/. User logs in to
     ft.com in that window. Cookies persist on disk in the profile.
  2. Subsequent scrapes use that same persistent profile in headless mode.

Scraping flow:
  - Navigate to https://www.ft.com/alphaville
  - For each visible article card, extract href + title + timestamp
  - Per article, fetch full HTML and extract <article> body text
  - Cache per-article to data/ft_alphaville_cache/<article_id>.json
  - Output: per-day parquet of (date, title, body_text, url)

Caveats:
  - FT may rotate session cookies; if scrapes start returning login pages,
    re-run scripts/ft_login.py to re-authenticate
  - Cloudflare bot challenge sometimes pops a JS challenge even with stealth;
    we wait for the 'article' selector with 30s timeout, retry once
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class AlphavillePost:
    article_id: str
    url: str
    title: str
    published_at: pd.Timestamp
    body_text: str


def _slugify(url: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", url.lower())[:120]


def _extract_post(html: str) -> dict:
    """Parse a single FT article HTML into title + body text + published_at."""
    from bs4 import BeautifulSoup
    s = BeautifulSoup(html, "html.parser")

    # Title
    title_el = s.select_one("h1") or s.select_one('[data-trackable="header"]')
    title = title_el.get_text(strip=True) if title_el else ""

    # Published timestamp
    time_el = s.select_one("time[datetime]")
    pub = pd.Timestamp(time_el["datetime"]) if time_el and time_el.has_attr("datetime") else pd.NaT

    # Article body: FT uses <article> wrapping; body paragraphs are <p>
    article_el = s.select_one("article") or s
    paragraphs = [p.get_text(strip=True) for p in article_el.select("p")
                  if len(p.get_text(strip=True)) > 30]
    body = "\n\n".join(paragraphs)
    return {"title": title, "published_at": pub, "body_text": body}


class FTAlphavilleScraper:
    """Persistent-profile Playwright scraper for FT Alphaville."""

    def __init__(self, profile_dir: Path, cache_dir: Path | None = None,
                 headless: bool = True):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless

    def _launch(self):
        """Open persistent context. Caller must close it."""
        from playwright.sync_api import sync_playwright
        from playwright_stealth import Stealth  # noqa
        p = sync_playwright().start()
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/131.0.0.0 Safari/537.36",
            locale="en-US",
            timezone_id="Europe/London",
        )
        # Re-inject .env cookies on every launch in case the persistent
        # profile lost / didn't store them. Cheap + idempotent.
        self._inject_env_cookies(ctx)
        return p, ctx

    def login_interactive(self) -> None:
        """Open a non-headless Chromium so the user can log in once.

        Cookies persist to the profile dir; subsequent scrapes use them headless.
        """
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=False,
                viewport={"width": 1280, "height": 800},
            )
            self._inject_env_cookies(ctx)
            page = ctx.new_page()
            page.goto("https://www.ft.com/login")
            print("Browser open. Log in to FT. Close the window when done.")
            try:
                page.wait_for_event("close", timeout=600_000)  # 10 min
            except Exception:
                pass
            ctx.close()

    def warmup_interactive(self, section: str) -> None:
        """Open a non-headless Chromium pointed at an FT section so the user
        can click through the Cloudflare challenge once for that section.

        Subsequent headless scrapes of the same section will reuse the now-
        warm cookies in the persistent profile.
        """
        from playwright.sync_api import sync_playwright
        section_clean = section.lstrip("/")
        url = f"https://www.ft.com/{section_clean}"
        with sync_playwright() as p:
            ctx = p.chromium.launch_persistent_context(
                user_data_dir=str(self.profile_dir),
                headless=False,
                viewport={"width": 1280, "height": 800},
            )
            self._inject_env_cookies(ctx)
            page = ctx.new_page()
            page.goto(url)
            print(
                f"Browser open at {url}.\n"
                f"Click through any Cloudflare challenge if shown; once you see\n"
                f"the section page (article links visible), close the window."
            )
            try:
                page.wait_for_event("close", timeout=600_000)
            except Exception:
                pass
            ctx.close()

    @staticmethod
    def _inject_env_cookies(ctx) -> None:
        """Read FT_* cookies from .env and pre-seed the browser context.

        Reuses credentials Howard already provided so headless cold-starts
        don't re-trip Cloudflare. The .env values are checked once;
        anything missing is silently skipped (Playwright tolerates partial
        cookie state — the persistent profile fills the rest).
        """
        import os
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv()
        except Exception:
            pass
        spec = [
            ("FTSession", "FT_SESSION_COOKIE"),
            ("FTCSRF", "FT_CSRF"),
            ("FTConsent", "FT_CONSENT_UUID"),
            ("spoor-id", "FT_SPOOR_ID"),
        ]
        to_set = []
        for cookie_name, env_var in spec:
            val = os.environ.get(env_var)
            if not val:
                continue
            to_set.append({
                "name": cookie_name,
                "value": val,
                "domain": ".ft.com",
                "path": "/",
                "secure": True,
                "httpOnly": False,
                "sameSite": "Lax",
            })
        if to_set:
            try:
                ctx.add_cookies(to_set)
                print(f"  injected {len(to_set)} FT cookies from .env")
            except Exception as e:
                print(f"  WARN: cookie inject failed: {e}")

    def _read_cache(self, url: str) -> Optional[AlphavillePost]:
        """Return a cached post if it exists, else None — never opens a browser."""
        if not self.cache_dir:
            return None
        cache_path = self.cache_dir / f"{_slugify(url)}.json"
        if not cache_path.exists():
            return None
        d = json.loads(cache_path.read_text())
        return AlphavillePost(
            article_id=_slugify(url), url=d["url"], title=d["title"],
            published_at=pd.Timestamp(d["published_at"]) if d.get("published_at") else pd.NaT,
            body_text=d["body_text"],
        )

    def _write_cache(self, url: str, parsed: dict) -> None:
        if not self.cache_dir:
            return
        cache_path = self.cache_dir / f"{_slugify(url)}.json"
        cache_path.write_text(json.dumps({
            "url": url,
            "title": parsed["title"],
            "published_at": parsed["published_at"].isoformat() if pd.notna(parsed["published_at"]) else None,
            "body_text": parsed["body_text"],
        }))

    def list_alphaville_urls(self, n_pages: int = 5, start_page: int = 1,
                              section: str = "alphaville") -> list[str]:
        """Walk an FT section index, harvest post URLs.

        Pages walked: [start_page, start_page + n_pages). Use ``start_page`` to
        resume deep historical walks without re-listing the front of the index.

        Sections (use ``--section`` from the pull script):
          - ``alphaville``       FT Alphaville blog (default)
          - ``on-wall-street``   FT On Wall Street column / archive
          - ``markets``          broad markets coverage
          - ``opinion``          op-ed pieces
        """
        from playwright_stealth import Stealth
        p, ctx = self._launch()
        try:
            page = ctx.new_page()
            Stealth().apply_stealth_sync(page)
            urls: list[str] = []
            for i in range(start_page, start_page + n_pages):
                target = f"https://www.ft.com/{section}?page={i}"
                try:
                    page.goto(target, wait_until="domcontentloaded", timeout=45_000)
                    page.wait_for_timeout(2500)
                    hrefs = page.eval_on_selector_all(
                        'a[href*="/content/"]',
                        "els => Array.from(new Set(els.map(e => e.href)))",
                    )
                    urls.extend(hrefs)
                except Exception as e:
                    print(f"  list page {i} failed: {e}")
                time.sleep(0.75)
            return sorted(set(urls))
        finally:
            ctx.close()
            p.stop()

    def fetch_post(self, url: str) -> Optional[AlphavillePost]:
        """Fetch one post (cache-first). For batches, prefer fetch_posts_batch."""
        cached = self._read_cache(url)
        if cached is not None:
            return cached
        from playwright_stealth import Stealth
        p, ctx = self._launch()
        try:
            page = ctx.new_page()
            Stealth().apply_stealth_sync(page)
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            try:
                page.wait_for_selector("article", timeout=20_000)
            except Exception:
                page.wait_for_timeout(2500)
            html = page.content()
        finally:
            ctx.close()
            p.stop()
        parsed = _extract_post(html)
        if not parsed["body_text"]:
            return None
        self._write_cache(url, parsed)
        return AlphavillePost(
            article_id=_slugify(url), url=url, title=parsed["title"],
            published_at=parsed["published_at"], body_text=parsed["body_text"],
        )

    def fetch_posts_batch(
        self,
        urls: list[str],
        *,
        delay_sec: float = 0.5,
        progress_every: int = 25,
    ):
        """Yield ``AlphavillePost`` for each URL, reusing one Playwright context.

        Cached URLs are returned without opening a browser. For uncached URLs we
        open ONE persistent context for the whole batch — eliminating the
        ~2-3s relaunch cost that dominated the single-fetch path.

        Yields tuples ``(index_1_based, url, post_or_None)`` so callers can
        checkpoint-write incrementally. ``post`` is ``None`` on failure.
        """
        from playwright_stealth import Stealth

        # Separate cached vs needs-fetch. Cached first → caller can checkpoint
        # immediately without waiting for Playwright spin-up.
        total = len(urls)
        to_fetch: list[str] = []
        for i, url in enumerate(urls, 1):
            cached = self._read_cache(url)
            if cached is not None:
                yield i, url, cached
            else:
                to_fetch.append(url)

        if not to_fetch:
            return

        p, ctx = self._launch()
        try:
            page = ctx.new_page()
            Stealth().apply_stealth_sync(page)
            n_done = total - len(to_fetch)
            for url in to_fetch:
                n_done += 1
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                    try:
                        page.wait_for_selector("article", timeout=20_000)
                    except Exception:
                        page.wait_for_timeout(2500)
                    html = page.content()
                except Exception as e:
                    print(f"  [{n_done}/{total}] navigate fail: {url} :: {e}")
                    yield n_done, url, None
                    time.sleep(delay_sec)
                    continue
                parsed = _extract_post(html)
                if not parsed["body_text"]:
                    yield n_done, url, None
                    time.sleep(delay_sec)
                    continue
                self._write_cache(url, parsed)
                post = AlphavillePost(
                    article_id=_slugify(url), url=url, title=parsed["title"],
                    published_at=parsed["published_at"], body_text=parsed["body_text"],
                )
                yield n_done, url, post
                if progress_every and n_done % progress_every == 0:
                    print(f"  [{n_done}/{total}] {parsed['title'][:60]}")
                time.sleep(delay_sec)
        finally:
            ctx.close()
            p.stop()


# ---------------------------------------------------------------------------
# Ticker extraction + per-(ticker, quarter) aggregation
# ---------------------------------------------------------------------------

# FT prose uses two reliable ticker patterns; we accept both and intersect with
# the validated universe to drop false positives like "(USD)" or "$VIX".
# However, the bulk of ticker exposure in FT copy comes via company *name*
# mentions ("Nvidia", "Apollo Global Management"), not cashtags — so
# build_company_name_matcher() builds an alternation regex over a name→ticker
# dict for the third extraction path.
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
PAREN_TICKER_RE = re.compile(r"\(([A-Z]{2,5})\)")

# Common corp-form suffixes stripped before regex matching. Order matters: we
# strip multi-word suffixes first so single-word residues like "CL A" don't
# trim the wrong token.
_SUFFIX_PATTERNS = [
    re.compile(r"\s+CLASS\s+[AB]$", re.IGNORECASE),
    re.compile(r"\s+CL\s+[AB]$", re.IGNORECASE),
    re.compile(r"\s+COMMON\s+STOCK$", re.IGNORECASE),
    re.compile(
        r"\s+(INC|CORP|CORPORATION|CO|LTD|LIMITED|PLC|LLC|LP|HOLDINGS?|GROUP|GRP|"
        r"COMPANIES?|COMPANY|SHS|N\.?V\.?|S\.?A\.?|AB|AG|SE|TRUST|REIT)\.?$",
        re.IGNORECASE,
    ),
]

# Names too short or ambiguous for high-precision matching in FT prose.
# Hand-curated stoplist; expand as false positives surface. Two categories:
#  1) Common-word company *names* (STRATEGY = MSTR, POPULAR = BPOP, etc.)
#     that match plain English in finance prose
#  2) Short single-word brand names (APPLE, FORD, etc.) that also appear as
#     ordinary nouns
_NAME_STOPLIST = {
    # Common-word company names
    "STRATEGY", "POPULAR", "GLOBAL", "GROWTH", "INCOME", "EQUITY", "CAPITAL",
    "ENERGY", "SELECT", "MOTION", "SCIENCE", "BENCHMARK", "ALPHA", "BETA",
    "PROGRESSIVE", "MEDICAL", "FINANCIAL", "INDUSTRIES", "INTERNATIONAL",
    "TECHNOLOGIES", "HOLDINGS", "SYSTEMS", "RESOURCES", "PARTNERS", "TRUST",
    "ENTERPRISE", "ENTERPRISES", "FRONTIER", "PIONEER", "FIRST", "AMERICAN",
    "NATIONAL", "UNIVERSAL", "PRIME", "SUMMIT", "SIGNATURE", "LEGACY",
    "HARBOUR", "ATLANTIC", "PACIFIC", "ARROW", "EAGLE", "TOWER",
    "BANDWIDTH", "BROADCAST", "MOMENTUM", "PROVIDENCE",
    # Short single-word names
    "APPLE", "FORD", "TARGET", "VISA", "GAP", "GAS", "OIL", "BOX",
    "SHELL", "BANK", "HOPE", "REGAL", "ALPHABET", "META", "AT&T",
}


def _clean_company_name(raw: str) -> str:
    """Strip incorporation suffixes and surrounding whitespace from a company name."""
    s = raw.strip()
    for pat in _SUFFIX_PATTERNS:
        s = pat.sub("", s)
    return s.strip()


def build_company_name_matcher(
    name_to_ticker: dict[str, str],
    *,
    min_name_len: int = 5,
    max_names: int = 1500,
) -> tuple[re.Pattern[str], dict[str, str]]:
    """Compile a single alternation regex over cleaned company names.

    Returns:
      (compiled regex with case-insensitive word-boundary alternation,
       lowercase name → ticker dict for resolution)

    The caller supplies a name→ticker dict ordered by priority (the typical
    pattern is to feed it sorted by descending 13F holding popularity). Only
    the first ``max_names`` survive — this keeps the alternation small enough
    to compile and match quickly.
    """
    seen_lower: dict[str, str] = {}
    for raw_name, ticker in name_to_ticker.items():
        if not raw_name or not ticker:
            continue
        cleaned = _clean_company_name(raw_name)
        if len(cleaned) < min_name_len:
            continue
        key = cleaned.lower()
        if key.upper() in _NAME_STOPLIST:
            continue
        if key in seen_lower:
            continue
        seen_lower[key] = ticker.upper()
        if len(seen_lower) >= max_names:
            break
    # Sort by length descending so longer names win over substrings
    # ("BERKSHIRE HATHAWAY" before "BERKSHIRE")
    ordered = sorted(seen_lower.keys(), key=len, reverse=True)
    pattern = r"\b(" + "|".join(re.escape(n) for n in ordered) + r")\b"
    return re.compile(pattern, re.IGNORECASE), seen_lower

# Tone lexicon shared shape with the Reddit adapter so downstream models see
# consistent feature semantics across text sources.
BULLISH_TERMS = re.compile(
    r"\b(rally|surge|beat|outperform|upgrade|breakout|catalyst|moat|buy)\b",
    re.IGNORECASE,
)
BEARISH_TERMS = re.compile(
    r"\b(plunge|miss|downgrade|short|fraud|sell|warning|guidance cut|bubble)\b",
    re.IGNORECASE,
)


def extract_tickers(
    text: str,
    valid_tickers: set[str],
    *,
    name_matcher: tuple[re.Pattern[str], dict[str, str]] | None = None,
) -> list[str]:
    """Return validated tickers found in ``text``.

    Sources, in order:
      1. ``$TICKER`` cashtags
      2. ``(TICKER)`` paren-tickers
      3. company-name mentions resolved via ``name_matcher`` (optional;
         the bulk of FT exposure surfaces this way)
    """
    if not isinstance(text, str) or not text:
        return []
    found: set[str] = set()
    for raw in CASHTAG_RE.findall(text):
        t = raw.upper()
        if t in valid_tickers:
            found.add(t)
    for raw in PAREN_TICKER_RE.findall(text):
        t = raw.upper()
        if t in valid_tickers:
            found.add(t)
    if name_matcher is not None:
        pattern, name_to_ticker = name_matcher
        for match in pattern.findall(text):
            t = name_to_ticker.get(match.lower())
            if t and t in valid_tickers:
                found.add(t)
    return sorted(found)


def aggregate_posts_to_ticker_quarter(
    posts: list[AlphavillePost],
    valid_tickers: set[str],
    *,
    name_matcher: tuple[re.Pattern[str], dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Roll up Alphaville posts to one row per (ticker, quarter_end).

    Columns:
      ticker, quarter_end, n_mentions, mean_body_len, bullish_count,
      bearish_count, joined_text
    The joined_text blob is intended for downstream Gemma embedding.
    """
    if not posts:
        return pd.DataFrame()
    rows: list[dict] = []
    for p in posts:
        text = f"{p.title}\n{(p.body_text or '')[:4000]}"
        tickers = extract_tickers(text, valid_tickers, name_matcher=name_matcher)
        if not tickers:
            continue
        bull = len(BULLISH_TERMS.findall(text))
        bear = len(BEARISH_TERMS.findall(text))
        body_len = len(p.body_text or "")
        for t in tickers:
            rows.append({
                "ticker": t,
                "published_at": p.published_at,
                "body_len": body_len,
                "bull": bull,
                "bear": bear,
                # Keep the title for the joined blob — full bodies blow up
                # the embed cost and most signal is in the lede + headline.
                "text": p.title,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    if df.empty:
        return pd.DataFrame()
    df["quarter_end"] = (
        df["published_at"].dt.tz_convert(None).dt.to_period("Q").dt.end_time.dt.normalize()
    )
    agg = (
        df.groupby(["ticker", "quarter_end"])
        .agg(
            n_mentions=("ticker", "size"),
            mean_body_len=("body_len", "mean"),
            bullish_count=("bull", "sum"),
            bearish_count=("bear", "sum"),
            joined_text=("text", lambda s: " || ".join(s)[:4000]),
        )
        .reset_index()
    )
    return agg
