"""Offline, metadata-only inventory of original monthly stocks and SEC snapshots.

Never imports strategy producers or reads prices, financial facts, ranks, or returns.
Current ticker links and historical identity admission are deliberately separate.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sys

from after156_event_scope_local_review import Sources, require

ROOT = Path(__file__).resolve().parents[1]
META = 'outputs/monthly_local_linear70/META.json'
PRIOR = 'work/monthly_profitability141_identity_review.json'
OUT = 'work/after156_filing_universe_inventory.json'
BEGIN, END = '2015-01-01', '2026-10-06'
MANIFEST = 'migration/top100_local_completion_20261019/FULL_LOCAL_MANIFEST_20261019T202326644648Z.json'
HINTS = 'outputs/dynamic_top100/registry_identity_hints.json'
FIELDS = ('accessionNumber', 'form', 'filingDate', 'reportDate', 'acceptanceDateTime', 'primaryDocument')
PAGE_BASE = 'outputs/dynamic_top100/registry_sec/filing_indexes/'


def symbol_key(s):
    # Explicit aliases already reviewed in PRIOR; never use a reused FB quote symbol.
    return {'research_equity_FB': 'META', 'BRK.B': 'BRK-B'}.get(s, s)


def relevant(entry):
    # Retain an adjacent-day boundary guard because prior review found descriptor
    # dates differing from actual filingDate by one day.
    low = (date.fromisoformat(BEGIN) - timedelta(days=1)).isoformat()
    high = (date.fromisoformat(END) + timedelta(days=1)).isoformat()
    return entry.get('filingFrom', '') <= high and entry.get('filingTo', '9999') >= low


def rows(raw):
    require(isinstance(raw, dict), 'SEC table not an object')
    require(all(isinstance(raw.get(k), list) for k in FIELDS), 'Missing SEC metadata column')
    n = len(raw['accessionNumber'])
    require(all(len(v) == n for v in raw.values() if isinstance(v, list)), 'Ragged SEC columns')
    return [tuple(raw[k][i] for k in FIELDS) for i in range(n)]


def self_test():
    require(symbol_key('research_equity_FB') == 'META', 'Internal alias')
    require(symbol_key('FB') == 'FB', 'No implicit reused ticker join')
    require(relevant({'filingFrom': '2010-01-01', 'filingTo': '2014-12-31'}), 'Boundary guard')
    require(not relevant({'filingFrom': '2010-01-01', 'filingTo': '2014-12-30'}), 'Before range')
    require(relevant({'filingFrom': END, 'filingTo': END}), 'Inclusive end')
    require(rows({k: ['x'] for k in FIELDS}) == [('x',) * 6], 'Metadata table')
    try:
        rows(dict({k: ['x'] for k in FIELDS}, form=[]))
    except ValueError:
        pass
    else:
        raise ValueError('Ragged table accepted')
    return 7


class Inventory:
    def __init__(self, src):
        self.src = src
        self.manifest = {x['path']: x for x in src.read(MANIFEST)}
        self.data = {}
        self.proofs = {}

    def read(self, path, cik=None, page=False, header_only=False):
        key = (path, header_only)
        if key in self.data:
            return self.data[key], self.proofs[path]
        pin = self.manifest.get(path)
        blob = self.src.read_bytes(path, pin['sha256'] if pin else None)
        if header_only:
            # The top-level SEC identity precedes filings; parse that valid JSON
            # prefix only during broad discovery, then full relevant snapshots.
            marker = re.search(rb',\s*"filings"\s*:', blob)
            require(marker is not None, 'No SEC filings boundary')
            raw = json.loads(blob[:marker.start()] + b'}')
        else:
            raw = json.loads(blob)
        proof = dict(path=path, sha256=self.src.hashes[path], migration_hash_verified=bool(pin))
        receipt = str(Path(path).with_suffix('.provenance.json'))
        expected_url = ('https://data.sec.gov/submissions/' + Path(path).name if page else
                        'https://data.sec.gov/submissions/CIK' + str(cik or raw.get('cik')).zfill(10) + '.json')
        proof['expected_source_url'] = expected_url
        proof['source_receipt_verified'] = False
        if self.src.path(receipt).is_file():
            rec = self.src.read(receipt)
            actual_url = rec.get('source', rec.get('url'))
            proof.update(receipt=receipt, source_url=actual_url, retrieved_at=rec.get('retrieved_at'))
            require(rec.get('sha256') == proof['sha256'], 'Receipt hash mismatch: ' + path)
            require(actual_url == expected_url, 'Receipt source mismatch: ' + path)
            proof['source_receipt_verified'] = True
        else:
            proof['missing_source_receipt'] = receipt
        self.data[key], self.proofs[path] = raw, proof
        return raw, proof

    def snapshot(self, path):
        raw, proof = self.read(path)
        cik = str(raw['cik']).zfill(10)
        recent = rows(raw['filings']['recent'])
        allrows = list(recent)
        pages, warnings, missing = [], [], []
        for desc in raw['filings'].get('files', []):
            name = desc['name']
            require(re.fullmatch('CIK' + cik + r'-submissions-\d+\.json', name) is not None,
                    'Unexpected advertised page identity')
            target = PAGE_BASE + name
            item = dict(advertised=desc, required_by_date_or_boundary_guard=relevant(desc), path=target)
            if not self.src.path(target).is_file():
                item['status'] = 'MISSING_LOCAL_PAGE'
                if relevant(desc):
                    missing.append(target)
            elif not relevant(desc):
                item['status'] = 'PRESENT_OUTSIDE_SCOPE_NOT_PARSED'
            else:
                chunk, page_proof = self.read(target, cik=cik, page=True)
                rr = rows(chunk)
                item.update(source=page_proof, rows=len(rr), count_matches=len(rr) == desc['filingCount'],
                            actual_filing_date_bounds=[min(r[2] for r in rr), max(r[2] for r in rr)] if rr else None)
                item['status'] = 'PRESENT_COUNT_MATCH' if item['count_matches'] else 'PRESENT_COUNT_MISMATCH'
                outside = [r for r in rr if not desc['filingFrom'] <= r[2] <= desc['filingTo']]
                if outside:
                    warnings.append(dict(path=target, outside_descriptor_rows=len(outside),
                        actual_bounds=item['actual_filing_date_bounds'],
                        affected_original_forms=sum(r[1] in ('10-Q', '10-K') and BEGIN <= r[2] <= END for r in outside)))
                allrows.extend(rr)
            pages.append(item)
        distinct = defaultdict(set)
        for row in allrows:
            distinct[row[0]].add(row)
        conflicts = sorted(k for k, rr in distinct.items() if len(rr) > 1)
        selected = set(r for r in allrows if BEGIN <= r[2] <= END)
        original = [r for r in selected if r[1] in ('10-Q', '10-K')]
        invalid = []
        for row in original:
            try:
                require(re.fullmatch(r'\d{10}-\d{2}-\d{6}', row[0]) is not None, 'Accession')
                parsed = datetime.fromisoformat(row[4].replace('Z', '+00:00'))
                require(parsed.tzinfo is not None, 'Naive acceptance timestamp')
                require(date.fromisoformat(row[3]) <= date.fromisoformat(row[2]), 'Report after filing')
                require(bool(row[5]), 'Missing primaryDocument')
            except (ValueError, TypeError, AttributeError) as exc:
                invalid.append(dict(accession=row[0], reason=str(exc)))
        result = dict(source=proof, cik=cik, sec_name=raw['name'], current_tickers=raw.get('tickers', []),
            former_names=raw.get('formerNames', []), recent_rows=len(recent), advertised_pages=pages,
            missing_required_pages=missing, descriptor_warnings=warnings,
            conflicting_accessions=conflicts, invalid_original_rows=invalid,
            observed_original_forms=dict(Counter(r[1] for r in original)),
            observed_original_events=len({r[0] for r in original}),
            observed_original_filing_date_bounds=[min(r[2] for r in original), max(r[2] for r in original)] if original else None,
            observed_original_by_year={str(y):sum(r[2][:4] == str(y) for r in original) for y in range(2015, 2027)},
            amendments_excluded=dict(Counter(r[1] for r in selected if r[1] in ('10-Q/A', '10-K/A'))),
            foreign_forms_excluded=dict(Counter(r[1] for r in selected if r[1] in ('20-F', '40-F', '6-K', '20-F/A', '40-F/A', '6-K/A'))),
            snapshot_pagination_present_and_counts_match=not missing and all(x.get('count_matches', True) for x in pages),
            every_consumed_source_has_original_receipt=proof['source_receipt_verified'] and all(
                x.get('source', {}).get('source_receipt_verified', False) for x in pages if 'source' in x),
            full_SEC_history_certified=False, clock_admitted=False)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    src = Sources()
    meta = src.read(META, '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
    prior = src.read(PRIOR)
    stocks = sorted(set(s for row in meta['symbols'] for s in row) - set(meta['etfs']))
    require(len(stocks) == 73 and len(meta['etfs']) == 17, 'Original universe changed')
    if args.inspect:
        print(json.dumps(dict(stocks=stocks, rows=prior['rows']), indent=2))
        return
    checks = self_test()
    require(len(meta['dates']) == 122 and all(len(x) == 49 for x in meta['symbols']), 'Original grid')
    inv = Inventory(src)
    old = {r['symbol']: r for r in prior['rows']}
    require(set(old) == set(stocks), 'Prior identity universe differs')
    hint_rows = src.read(HINTS)
    hints = defaultdict(list)
    for row in hint_rows:
        for symbol in stocks:
            if row['symbol'] in (symbol, symbol_key(symbol)):
                hints[symbol].append({k: row[k] for k in ('evidence_id', 'symbol', 'names', 'candidate_ciks',
                    'previous_possible_ciks', 'name_evidence', 'identity_verified') if k in row})
    # Inspect only submission metadata, never facts/shares/market caches.
    paths = sorted(set(str(p.relative_to(ROOT)) for base in
        ('outputs/dynamic_top100', 'outputs/index_union/sources/sec_identity')
        for p in src.path(base).rglob('*_submissions.json')))
    by_cik = defaultdict(list)
    by_symbol = defaultdict(set)
    filename_leads = defaultdict(set)
    parse_failures = []
    for path in paths:
        try:
            raw, _ = inv.read(path, header_only=True)
            cik = str(raw['cik']).zfill(10)
            require(re.fullmatch(r'\d{10}', cik) is not None, 'Invalid CIK')
            require('name' in raw and 'tickers' in raw, 'Missing identity fields')
        except (ValueError, KeyError) as exc:
            parse_failures.append(dict(path=path, error=str(exc)))
            continue
        by_cik[cik].append(path)
        for symbol in stocks:
            if symbol_key(symbol) in raw.get('tickers', []):
                by_symbol[symbol].add(cik)
            if Path(path).name == symbol_key(symbol) + '_submissions.json':
                filename_leads[symbol].add(cik)
    # Reverify the one research alias against its explicitly reviewed source bytes.
    alias = src.read('outputs/monthly_earnings_repair/meta/IDENTITY.json')
    require(alias['research_ticker'] == 'research_equity_FB', 'Research alias changed')
    src.read_bytes(alias['evidence_path'], alias['evidence_sha256'])
    stock_rows, needed_ciks = [], set()
    for symbol in stocks:
        candidates = set(old[symbol].get('additional_locating_ciks', [])) | filename_leads[symbol]
        for hint in hints[symbol]:
            candidates.update(hint.get('candidate_ciks', []))
            candidates.update(hint.get('previous_possible_ciks', []))
        candidates |= by_symbol[symbol]
        candidates |= {x['cik'] for x in old[symbol].get('current_registrants', [])}
        needed_ciks |= candidates
        matches = sorted(by_symbol[symbol])
        status = ('EXACT_CURRENT_METADATA_LINK' if len(matches) == 1 else
                  'AMBIGUOUS_MULTIPLE_CURRENT_CIKS' if matches else
                  'LOCATING_CANDIDATES_ONLY' if candidates else 'MISSING_CIK_LINK')
        indices = [i for i, row in enumerate(meta['symbols']) if symbol in row]
        stock_rows.append(dict(symbol=symbol, current_metadata_link_status=status,
            exact_current_ciks=matches, non_admitted_locating_ciks=sorted(candidates - set(matches)),
            local_snapshot_paths={c: by_cik[c] for c in sorted(candidates)},
            existing_identity_review_status=old[symbol]['status'],
            existing_identity_notes={k: old[symbol][k] for k in ('note', 'foreign_reporting_note', 'additional_evidence') if k in old[symbol]},
            registry_hints=hints[symbol],
            historical_security_identity_admitted=False,
            candidate_slot_count=len(indices), first_candidate_date=meta['dates'][indices[0]],
            last_candidate_date=meta['dates'][indices[-1]],
            slot_counts=dict(early53=sum(i < 53 for i in indices), remaining_first88=sum(53 <= i < 88 for i in indices),
                            later34=sum(i >= 88 for i in indices))))
    snapshots = []
    print('Scanning complete: %d submission files, %d candidate CIKs; checking required pages.' %
          (len(paths), len(needed_ciks)), file=sys.stderr, flush=True)
    for cik in sorted(needed_ciks):
        for path in by_cik[cik]:
            snapshots.append(inv.snapshot(path))
            if len(snapshots) % 25 == 0:
                print('Inventoried %d relevant snapshots.' % len(snapshots), file=sys.stderr, flush=True)
    snap_by_path = {s['source']['path']:s for s in snapshots}
    pilot = src.read('work/after156_filing_event_feasibility.json')
    for row in pilot['issuers']:
        snap = snap_by_path['outputs/dynamic_top100/registry_sec/' + row['cik'] + '_submissions.json']
        require(snap['observed_original_events'] == row['original_filing_events'] and
                snap['observed_original_forms'] == row['original_forms'], 'Prior pilot metadata reconciliation')
    for row in stock_rows:
        current = [snap_by_path[p] for c in row['exact_current_ciks'] for p in by_cik[c]]
        row['any_current_snapshot_has_original_10Q_10K'] = any(s['observed_original_events'] for s in current)
        row['any_current_snapshot_has_closed_advertised_pagination'] = any(s['snapshot_pagination_present_and_counts_match'] for s in current)
        row['any_current_snapshot_has_closed_pagination_and_source_receipts'] = any(
            s['snapshot_pagination_present_and_counts_match'] and s['every_consumed_source_has_original_receipt'] for s in current)
    missing = sorted({p for s in snapshots for p in s['missing_required_pages']})
    summary = dict(original_stocks=len(stocks), excluded_etfs=len(meta['etfs']), original_months=len(meta['dates']),
        stock_candidate_slots=sum(r['candidate_slot_count'] for r in stock_rows),
        scanned_local_submission_files=len(paths), relevant_cik_candidates=len(needed_ciks), relevant_snapshots=len(snapshots),
        current_link_status_counts=dict(Counter(r['current_metadata_link_status'] for r in stock_rows)),
        missing_or_locating_only_symbols=[r['symbol'] for r in stock_rows if not r['exact_current_ciks']],
        ambiguous_current_link_symbols=[r['symbol'] for r in stock_rows if len(r['exact_current_ciks']) > 1],
        stocks_with_observed_original_forms=sum(r['any_current_snapshot_has_original_10Q_10K'] for r in stock_rows),
        current_link_symbols_without_observed_original_forms=[r['symbol'] for r in stock_rows if r['exact_current_ciks'] and not r['any_current_snapshot_has_original_10Q_10K']],
        stocks_with_any_closed_current_snapshot=sum(r['any_current_snapshot_has_closed_advertised_pagination'] for r in stock_rows),
        stocks_with_any_closed_current_snapshot_and_receipts=sum(r['any_current_snapshot_has_closed_pagination_and_source_receipts'] for r in stock_rows),
        unique_missing_required_history_pages=len(missing), snapshot_count_mismatches=sum(
            x.get('count_matches') is False for s in snapshots for x in s['advertised_pages']),
        snapshot_invalid_original_rows=sum(len(s['invalid_original_rows']) for s in snapshots),
        fully_admitted_historical_security_mappings=0, clocks_admitted=0)
    report = dict(schema_version=1, status='FULL_ORIGINAL_STOCK_UNIVERSE_METADATA_INVENTORY_NOT_ADMISSION',
        created_utc=datetime.now(timezone.utc).isoformat(), runtime_python=sys.version,
        scope=dict(begin=BEGIN, end=END, stocks=stocks, excluded_etfs=meta['etfs'],
            forms=['10-Q', '10-K'], all_73_original_equity_identities_preserved=True,
            date_filter='Actual filingDate, inclusive. Every advertised page is listed; pages overlapping the interval or adjacent-day guard are parsed if present and otherwise remain explicit gaps. Present pages strictly outside the guard are not parsed.',
            source_search_roots=['outputs/dynamic_top100/**/*_submissions.json', 'outputs/index_union/sources/sec_identity/**/*_submissions.json']),
        summary=summary, stocks=stock_rows, submission_snapshots=snapshots,
        missing_required_history_pages=missing, scan_failures=parse_failures,
        candidate_ciks_without_local_submission={c:[r['symbol'] for r in stock_rows if c in r['local_snapshot_paths']]
            for c in sorted(needed_ciks) if not by_cik[c]},
        remaining_gates=[
            'Exact current-metadata CIK links do not establish historical security/class continuity. Resolve each relevant dated identity, successor, ticker-reuse and ADR/class bridge before using auxiliary events.',
            'Registry name hints and previous_possible_ciks are unverified locating candidates; never concatenate their filing histories by ticker or company-name similarity.',
            'Resolve every missing required advertised page, count mismatch, conflicting accession and missing original receipt. Migration hashes establish preserved local bytes, not original SEC provenance or complete history.',
            'Foreign 20-F/40-F/6-K records require a separately specified event taxonomy; no foreign form is silently recast as an original 10-Q/10-K event.',
            'Authenticate original acceptance-time semantics independently against original headers/indexes, then define conservative market-session/close mapping. Parseable timestamps and pagination closure confer no clock admission.',
            'Before any price access, freeze event taxonomy, amendment/duplicate policy, common coverage mask, same-mask controls and prospective protocol. Keep dominant skip42 long-term momentum, three fixed intramonth holdings and original universe.'],
        activity_counts=dict(price_values_read=0, financial_values_extracted=0, factors=0, ranks=0, returns=0,
            orders=0, trials=0, network_requests=0, frozen_files_modified=0),
        limitations=['This is a bounded inventory of specified local source trees, not a claim that missing CIKs or filings do not exist.',
            'Repeated source snapshots are listed separately; event counts must not be summed across copies or unadmitted candidate CIKs.',
            'Every current ticker match is exact after only the explicit BRK.B spelling and research_equity_FB alias. Historical identity is never inferred from that match.',
            'No permanent universe reduction to the prior three-issuer pilot; all 73 identities remain visible, including unresolved identities and non-10-Q/10-K issuers.'],
        synthetic_checks=dict(passed=True, count=checks),
        prior_pilot_reconciliation=dict(passed=True, issuers=3, original_events=141))
    for path in ('work/after156_filing_universe_inventory.py', 'work/after156_event_scope_local_review.py',
                 'AGENTS.md', 'LOCAL_SETUP.md', 'run_local.sh', 'local_run.py', 'local.sandbox.sb'):
        src.read_bytes(path)
    print('All relevant pages inspected; rechecking input hashes.', file=sys.stderr, flush=True)
    src.recheck()
    report['input_hashes'] = dict(sorted(src.hashes.items()))
    if args.write:
        with src.path(OUT).open('x') as f:
            json.dump(report, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(summary=summary, scan_failures=parse_failures,
                         candidate_ciks_without_local_submission=report['candidate_ciks_without_local_submission']), indent=2))


if __name__ == '__main__':
    main()
