"""Registry of tracked super-investors and their primary 13F-filing CIKs.

CIK = SEC Central Index Key, 10-digit zero-padded. Every entity that files with
the SEC has exactly one CIK. For 13F-HR filings, the CIK identifies the
INSTITUTIONAL MANAGER, not the fund or LP. Some managers file multiple 13Fs
(e.g. Soros Fund Management LLC vs Soros Capital Management) — we list the
primary holdings filer per name.

These CIKs are verified at first scrape via
`https://data.sec.gov/submissions/CIK{cik}.json`. If a CIK changes (rare), update
this file.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Investor:
    name: str
    slug: str          # short id used in filenames / CLI / config
    cik: str           # 10-digit zero-padded
    style: str         # short tag for UI / filtering
    notes: str = ""


# Listed in approximate AUM / fame order. CIKs verified Aug 2024; re-verify
# annually with `scripts/verify_ciks.py`.
INVESTORS: tuple[Investor, ...] = (
    Investor("Berkshire Hathaway", "buffett", "0001067983", "value-long-horizon",
             "Includes Todd Combs + Ted Weschler picks; primary filer for the entire BRK portfolio"),
    Investor("Pershing Square Capital", "ackman", "0001336528", "concentrated-activist"),
    Investor("Appaloosa LP", "tepper", "0001656456", "distressed-contrarian"),
    Investor("Baupost Group", "klarman", "0001061768", "deep-value",
             "Seth Klarman; CIK 1061165 was mislabeled (that is Lone Pine) — corrected 2026-06"),
    Investor("Greenlight Capital", "einhorn", "0001079114", "long-short-value"),
    Investor("Scion Asset Management", "burry", "0001649339", "tail-risk-contrarian",
             "Famously volatile portfolio; ~10 positions per filing"),
    Investor("Third Point", "loeb", "0001040273", "event-driven-activist"),
    Investor("Dalal Street LLC", "pabrai", "0001549575", "concentrated-value",
             "Mohnish Pabrai; current 13F filer (Dalal Street). Old CIK 1173334 was dormant — updated 2026-06"),
    Investor("Gotham Asset Management", "greenblatt", "0001510387", "magic-formula-systematic",
             "Joel Greenblatt; broad portfolio, formulaic value+quality"),
    Investor("Duquesne Family Office", "druckenmiller", "0001536411", "macro"),
    Investor("Soros Fund Management", "soros", "0001029160", "macro-family-office",
             "Soros LLC files; the legacy hedge fund wound down in 2011"),
    Investor("Renaissance Technologies", "renaissance", "0001037389", "quant-high-turnover",
             "RIEF / Renaissance Institutional Equities Fund; Medallion is not 13F-disclosed"),
    Investor("Two Sigma Investments", "two_sigma", "0001179392", "quant-high-turnover"),
    # --- expansion 2026-06: 15 verified 13F-HR filers (CIKs checked vs SEC submissions) ---
    Investor("Tiger Global Management", "tiger_global", "0001167483", "growth-tiger-cub",
             "Chase Coleman; concentrated tech/growth"),
    Investor("Coatue Management", "coatue", "0001135730", "growth-tiger-cub", "Philippe Laffont"),
    Investor("Viking Global Investors", "viking", "0001103804", "long-short-tiger-cub", "Andreas Halvorsen"),
    Investor("TCI Fund Management", "tci", "0001647251", "concentrated-activist", "Chris Hohn"),
    Investor("Himalaya Capital", "himalaya", "0001709323", "concentrated-value", "Li Lu"),
    Investor("Akre Capital Management", "akre", "0001112520", "quality-compounder", "Chuck Akre"),
    Investor("Harris Associates (Oakmark)", "oakmark", "0000813917", "value", "Bill Nygren"),
    Investor("Citadel Advisors", "citadel", "0001423053", "multi-strat-quant", "Ken Griffin"),
    Investor("Millennium Management", "millennium", "0001273087", "multi-strat-quant", "Izzy Englander"),
    Investor("Point72 Asset Management", "point72", "0001603466", "multi-strat", "Steve Cohen"),
    Investor("Bridgewater Associates", "bridgewater", "0001350694", "macro", "Ray Dalio"),
    Investor("D. E. Shaw", "de_shaw", "0001009207", "multi-strat-quant"),
    Investor("Fairholme Capital", "fairholme", "0001056831", "concentrated-value", "Bruce Berkowitz"),
    Investor("Icahn Capital", "icahn", "0000921669", "activist",
             "Carl Icahn; CIK 1412093 (Icahn Capital LP) files 13F-NT notice-only — 921669 (Carl C Icahn) is the 13F-HR holdings filer. Corrected 2026-06"),
    Investor("Altimeter Capital", "altimeter", "0001541617", "growth", "Brad Gerstner"),
    # --- expansion 2026-06 (megasweep): 16 net-new verified 13F-HR filers ---
    Investor("Oaktree Capital Management", "marks", "0000949509", "distressed-credit", "Howard Marks"),
    Investor("Aquamarine Capital", "spier", "0001953324", "concentrated-value", "Guy Spier"),
    Investor("Trian Fund Management", "peltz", "0001345471", "activist", "Nelson Peltz"),
    Investor("Inclusive Capital Partners", "ubben", "0001418814", "esg-activist", "Jeff Ubben (ex-ValueAct)"),
    Investor("Elliott Investment Management", "singer", "0001791786", "activist-distressed", "Paul Singer"),
    Investor("Glenview Capital", "robbins", "0001138995", "long-short-healthcare", "Larry Robbins"),
    Investor("Lone Pine Capital", "mandel", "0001061165", "long-short-tiger-cub", "Stephen Mandel"),
    Investor("Maverick Capital", "ainslie", "0000934639", "long-short-tiger-cub", "Lee Ainslie"),
    Investor("D1 Capital Partners", "sundheim", "0001747057", "growth-crossover", "Dan Sundheim"),
    Investor("Abrams Capital Management", "abrams", "0001358706", "concentrated-value", "David Abrams"),
    Investor("Punch Card Management", "punchcard", "0001631664", "concentrated-value", "Norbert Lou"),
    Investor("Gardner Russo & Quinn", "russo", "0000860643", "global-value-compounder", "Tom Russo"),
    Investor("Giverny Capital", "rochon", "0001641864", "quality-compounder", "Francois Rochon"),
    Investor("Valley Forge Capital", "kantesaria", "0001697868", "concentrated-quality", "Dev Kantesaria"),
    Investor("Semper Augustus", "semper", "0001115373", "concentrated-value", "Chris Bloomstran"),
    Investor("Fairfax Financial", "watsa", "0000915191", "value-insurance-float", "Prem Watsa"),
)


_BY_SLUG: dict[str, Investor] = {i.slug: i for i in INVESTORS}
_BY_CIK: dict[str, Investor] = {i.cik: i for i in INVESTORS}


def by_slug(slug: str) -> Investor:
    if slug not in _BY_SLUG:
        raise KeyError(f"unknown investor slug {slug!r}; known: {sorted(_BY_SLUG)}")
    return _BY_SLUG[slug]


def by_cik(cik: str) -> Investor:
    cik = cik.zfill(10)
    if cik not in _BY_CIK:
        raise KeyError(f"unknown investor CIK {cik}; known: {sorted(_BY_CIK)}")
    return _BY_CIK[cik]
