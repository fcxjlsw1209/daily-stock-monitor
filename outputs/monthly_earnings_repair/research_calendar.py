"""Research-only XNYS calendar with checksum-verified official closure evidence.

Do not register this class globally or modify the pinned dependency/account code.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pandas as pd
from exchange_calendars.exchange_calendar_xnys import XNYSExchangeCalendar

P = Path(__file__).resolve().parent
EVIDENCE_ROOT = P.parent / 'index_union'
MANIFEST = EVIDENCE_ROOT / 'CALENDAR_EXCEPTIONS.json'


def closure_evidence():
    manifest = json.loads(MANIFEST.read_text())
    result = []
    for row in manifest['closures']:
        if 'XNYS' not in row['markets']:
            continue
        source = EVIDENCE_ROOT / row['source_file']
        if hashlib.sha256(source.read_bytes()).hexdigest() != row['source_sha256']:
            raise ValueError('Calendar source checksum mismatch')
        result.append(dict(row, source_file=str(source)))
    return result


class EarningsResearchXNYS(XNYSExchangeCalendar):
    @property
    def adhoc_holidays(self):
        return sorted(set(super().adhoc_holidays) |
                      {pd.Timestamp(row['date']) for row in closure_evidence()})


@lru_cache(maxsize=4)
def market_calendar(start='2000-01-01', end='2035-12-31'):
    return EarningsResearchXNYS(start=start, end=end)


def provenance():
    return dict(implementation='research_only_XNYS_official_closure_overlay_v1',
                manifest=str(MANIFEST),
                manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                closures=closure_evidence(), global_registry_modified=False)
