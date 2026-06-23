"""End-to-end smoke: registry resolves, 13F XML parses, one filing round-trips."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from super_investor.investors import INVESTORS, by_slug, by_cik


def test_registry_has_core_investors():
    # Registry only grows as investors are added; assert the floor, not an exact
    # count, so expanding INVESTORS no longer breaks this smoke test.
    assert len(INVESTORS) >= 13


def test_registry_slug_lookups():
    assert by_slug("buffett").name == "Berkshire Hathaway"
    assert by_slug("burry").style == "tail-risk-contrarian"
    with pytest.raises(KeyError):
        by_slug("nonexistent")


def test_registry_cik_lookups():
    assert by_cik("1067983").slug == "buffett"        # zero-pad lenient
    assert by_cik("0001067983").slug == "buffett"


def test_parse_minimal_information_table_xml(tmp_path):
    """Synthetic 13F XML with one position parses to one row with correct fields."""
    from super_investor.adapters.sec_13f import _parse_info_table_xml, FilingRef
    import pandas as pd

    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>178391000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>905560000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
    <investmentDiscretion>DFND</investmentDiscretion>
    <votingAuthority>
      <Sole>905560000</Sole><Shared>0</Shared><None>0</None>
    </votingAuthority>
  </infoTable>
</informationTable>
"""
    ref = FilingRef(
        cik="0001067983",
        accession="0001067983-24-000005",
        accession_nodash="000106798324000005",
        form="13F-HR",
        filed_at=pd.Timestamp("2024-05-15"),
        period_of_report=pd.Timestamp("2024-03-31"),
    )
    df = _parse_info_table_xml(xml, ref)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["name_of_issuer"] == "APPLE INC"
    assert row["cusip"] == "037833100"
    assert row["shares"] == 905_560_000
    assert row["value_usd"] == 178_391_000 * 1000  # filing reports thousands -> USD


@pytest.mark.skipif(not os.environ.get("SEC_USER_AGENT"),
                    reason="needs SEC_USER_AGENT env (live SEC call)")
def test_live_buffett_filings_list(tmp_path):
    """Touches the live SEC EDGAR API; opt-in only via env."""
    from super_investor.adapters.sec_13f import Edgar13FClient
    client = Edgar13FClient(cache_dir=tmp_path)
    filings = client.list_filings("0001067983")
    assert len(filings) > 50, "Berkshire should have 70+ 13Fs since 1994"
    assert any(f.form == "13F-HR" for f in filings)
