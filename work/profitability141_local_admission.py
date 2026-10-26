"""Offline annual source-admission candidates; never factors, portfolios or returns."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import monthly_profitability141_sources as source

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = 'outputs/monthly_profitability141_local_admission'
ORIGINAL_PLAN = 'outputs/monthly_profitability141_sources/DATA_PLAN.json'
ORIGINAL_SHA = '214fb91e749d6d6dc54ff4b479e6230a1b4cecac48bcca69a15b017e4110d112'
INVENTORY = 'work/profitability141_migrated_cache_inventory_complete_20261019T202336581108Z.json'
INVENTORY_SHA = '22a14e629a95558a1b1c3943312c6e6b7fffdf6ac1af33d8087374b7f4da30fa'
CALENDAR_PLAN = 'outputs/monthly_forward_guidance_preflight_v1/CALENDAR_PLAN.json'
CALENDAR_SHA = '4b8f32ad8a555d6537d7cf3bd0f831e64fbb2de26d033c8daaf68590fcd4a6ea'
CALENDAR_SESSIONS = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SELF = 'work/profitability141_local_admission.py'
TEST = 'work/test_profitability141_local_admission.py'
TAGS = ('GrossProfit', 'Assets')


def now():
    return datetime.now(timezone.utc).isoformat()


def path(relative, root=ROOT):
    p = Path(relative)
    p = p if p.is_absolute() else root / p
    p.relative_to(root)
    if p.resolve() != p:
        raise ValueError('Symlink or noncanonical path')
    return p


def read(relative, root=ROOT):
    return source.strict_json(path(relative, root).read_bytes())


def pin(relative, root=ROOT):
    p = path(relative, root)
    return {'path': str(p.relative_to(root)), 'sha256': source.sha(p.read_bytes())}


def verify(binding, root=ROOT):
    p = path(binding['path'], root)
    raw = p.read_bytes()
    if source.sha(raw) != binding['sha256']:
        raise ValueError('Source digest changed: ' + binding['path'])
    return raw


def write(relative, obj, root=ROOT):
    source.exclusive_bytes(path(relative, root), source.json_bytes(obj))


def receipt_check(raw, cik, receipt):
    reasons = []
    if receipt.get('sha256') != source.sha(raw):
        reasons.append('origin_receipt_sha_mismatch')
    if receipt.get('source') != source.endpoint(cik):
        reasons.append('origin_receipt_url_cik_mismatch')
    try:
        stamp = datetime.fromisoformat(receipt['retrieved_at'].replace('Z', '+00:00'))
        if stamp.utcoffset() is None:
            raise ValueError('Naive origin time')
    except (ValueError, TypeError, KeyError, AttributeError):
        reasons.append('invalid_origin_retrieved_at')
    return reasons


def first_session_after(filed, sessions, coverage_start=None):
    filed = source.iso_day(filed).isoformat()
    if not sessions or sessions != sorted(set(sessions)):
        raise ValueError('Sorted unique session list required')
    if filed < (coverage_start or sessions[0]):
        return None
    pos = bisect_right(sessions, filed)
    return sessions[pos] if pos < len(sessions) else None


def temporally_available(candidate, as_of):
    """Timing predicate only, never overrides admission/context blockers."""
    day = source.iso_day(as_of).isoformat()
    first = candidate.get('first_usable_session')
    return first is not None and first <= day


def exact_amount(row):
    value = row.get('val')
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError('invalid_amount')
    value = Decimal(value)
    if not value.is_finite():
        raise ValueError('nonfinite_amount')
    return value


def row_evidence(row, tag, unit, index, binding):
    evidence = {'source_path': binding['path'], 'source_sha256': binding['sha256'],
                'json_pointer': '/facts/us-gaap/' + tag + '/units/' + unit + '/' + str(index),
                'metadata': {k: row[k] for k in ('accn', 'filed', 'form', 'start', 'end', 'fy', 'fp', 'frame') if k in row}}
    try:
        evidence['amount_decimal'] = format(exact_amount(row), 'f')
    except ValueError:
        evidence['amount_decimal'] = None
        evidence['invalid_amount_type'] = type(row.get('val')).__name__
    return evidence


def extract_company(cik, documents, sessions, catalog=None, calendar_start=None):
    """Pair strict annual source rows, retaining all filing vintages and blockers.

    documents: [(raw_bytes, {path,sha256}), ...], all independently provenance checked.
    catalog maps accession to original submission catalog metadata, not XBRL context.
    """
    source.check_cik(cik)
    groups = defaultdict(lambda: {'GrossProfit': [], 'Assets': []})
    rejected = Counter()
    for raw, binding in documents:
        obj = source.validate_facts(raw, cik)
        gaap = obj['facts'].get('us-gaap', {})
        for tag in TAGS:
            concept = gaap.get(tag)
            if concept is None:
                rejected['missing_' + tag + '_concept'] += 1
                continue
            if not isinstance(concept, dict) or not isinstance(concept.get('units'), dict):
                raise ValueError('Invalid concept units')
            for unit, rows in sorted(concept['units'].items()):
                if not isinstance(rows, list):
                    raise ValueError('Invalid concept rows')
                for index, row in enumerate(rows):
                    if not isinstance(row, dict):
                        rejected['invalid_row_object'] += 1
                        continue
                    if row.get('form') != '10-K':
                        rejected['not_10K_' + tag] += 1
                        continue
                    if not re.fullmatch(r'[A-Z]{3}', unit):
                        rejected['not_monetary_unit_' + tag] += 1
                        continue
                    try:
                        filed, end = source.iso_day(row.get('filed')), source.iso_day(row.get('end'))
                        if end > filed:
                            raise ValueError('Future period')
                    except (ValueError, TypeError):
                        rejected['invalid_dates_' + tag] += 1
                        continue
                    accn = row.get('accn')
                    if not isinstance(accn, str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', accn):
                        rejected['invalid_accession_' + tag] += 1
                        continue
                    key = (cik, accn, filed.isoformat(), '10-K', unit, end.isoformat())
                    if tag == 'GrossProfit' and source.annual_period(row) is None:
                        rejected['not_annual_duration_GrossProfit'] += 1
                        continue
                    if tag == 'Assets' and 'start' in row:
                        rejected['Assets_not_instant'] += 1
                        continue
                    groups[key][tag].append(row_evidence(row, tag, unit, index, binding))
    candidates = []
    for key, values in sorted(groups.items()):
        gp, assets = values['GrossProfit'], values['Assets']
        # Asset-only keys remain counted as explicit source gaps.
        if not gp:
            rejected['annual_assets_without_matching_GP_key'] += 1
            continue
        _, accn, filed, form, unit, end = key
        reasons = []
        starts = sorted({r['metadata']['start'] for r in gp})
        if len(starts) != 1:
            reasons.append('ambiguous_annual_duration')
        if not assets:
            reasons.append('missing_same_filing_instant_assets')
        fiscal_years = set()
        for row in gp + assets:
            meta = row['metadata']
            if 'fp' in meta and meta['fp'] != 'FY':
                reasons.append('nonannual_fiscal_period_metadata')
            if 'fy' in meta:
                if type(meta['fy']) is not int or not 1900 <= meta['fy'] <= 2200:
                    reasons.append('invalid_fiscal_year_metadata')
                else:
                    fiscal_years.add(meta['fy'])
        if len(fiscal_years) > 1:
            reasons.append('conflicting_filing_fiscal_years')
        amounts = {}
        for tag, evidence in values.items():
            vs = {Decimal(r['amount_decimal']) for r in evidence if r['amount_decimal'] is not None}
            if any(r['amount_decimal'] is None for r in evidence):
                reasons.append('invalid_' + tag + '_amount')
            if len(vs) > 1:
                reasons.append('conflicting_' + tag + '_values')
            amounts[tag] = format(next(iter(vs)), 'f') if len(vs) == 1 else None
            if tag == 'Assets' and any(v <= 0 for v in vs):
                reasons.append('nonpositive_assets')
        first = first_session_after(filed, sessions, calendar_start)
        if first is None:
            reasons.append('outside_verified_calendar_range')
        entries = (catalog or {}).get(accn, [])
        matching = [r for r in entries if r.get('form') == form and r.get('filingDate') == filed]
        if any(r.get('validation_blockers') for r in matching):
            reasons.append('invalid_filing_catalog_row')
        if entries and len(matching) != len(entries):
            reasons.append('filing_catalog_conflict')
        report_dates = {r.get('reportDate') for r in matching if r.get('reportDate')}
        if len(report_dates) > 1:
            reasons.append('filing_catalog_report_date_conflict')
        reported_end = next(iter(report_dates)) if len(report_dates) == 1 else None
        if reported_end:
            try:
                source.iso_day(reported_end)
                if end > reported_end:
                    reasons.append('period_after_catalog_report_date')
            except (ValueError, TypeError):
                reasons.append('invalid_catalog_report_date')
        period_role = ('current_filing_period' if end == reported_end else 'comparative_period') if reported_end else 'catalog_report_period_unverified'
        candidates.append({'cik': cik, 'accession': accn, 'filed': filed, 'form': form, 'unit': unit, 'end': end,
                           'gross_profit_starts': starts, 'duration_days_inclusive': (source.iso_day(end) - source.iso_day(starts[0])).days + 1 if len(starts) == 1 else None,
                           'gross_profit_decimal': amounts['GrossProfit'], 'assets_decimal': amounts['Assets'],
                           'evidence': values, 'structural_blockers': sorted(set(reasons)),
                           'structurally_complete': not reasons, 'first_usable_session': first,
                           'availability_rule': 'first pinned XNYS+closure-overlay session strictly after filed date; timing only',
                           'filing_catalog_matches': matching, 'filing_catalog_verified': bool(matching) and not any('catalog' in r for r in reasons),
                           'filing_report_date': reported_end, 'period_role': period_role,
                           'original_filing_context_verified': False, 'historical_security_identity_verified': False,
                           'admitted': False, 'admission_blockers': ['original_10K_context_not_verified', 'historical_security_identity_not_verified'],
                           'status': 'blocked_source_structure' if reasons else 'source_candidate_not_admitted'})
    return candidates, dict(sorted(rejected.items()))


def annual_filing_events(ciks, catalogs, candidates, sessions, calendar_start=None):
    """Preserve unavailable/missing annual events; never filter to valid values first."""
    groups = defaultdict(list)
    for cik in ciks:
        for accn, entries in catalogs.get(cik, {}).items():
            for entry in entries:
                if entry.get('form') not in ('10-K', '10-K/A', '20-F', '20-F/A', '40-F', '40-F/A'):
                    continue
                try:
                    filed = source.iso_day(entry.get('filingDate')).isoformat()
                except (ValueError, TypeError):
                    filed = str(entry.get('filingDate'))
                groups[cik, str(accn), filed].append(entry)
    for c in candidates:
        groups.setdefault((c['cik'], c['accession'], c['filed']), [])
    result = []
    for (cik, accn, filed), catalog in sorted(groups.items()):
        items = [c for c in candidates if (c['cik'], c['accession'], c['filed']) == (cik, accn, filed)]
        report_dates = {r.get('reportDate') for r in catalog if r.get('reportDate')}
        report_date = next(iter(report_dates)) if len(report_dates) == 1 else None
        current = [c for c in items if c['end'] == report_date] if report_date else []
        forms = {r.get('form') for r in catalog if isinstance(r.get('form'), str)}
        form = next(iter(forms)) if len(forms) == 1 else ('10-K' if not forms else 'conflicting_annual_forms')
        blockers = ['original_10K_context_not_verified', 'historical_security_identity_not_verified']
        if form != '10-K':
            blockers.append('annual_form_outside_10K_contract')
        blockers.extend(sorted({reason for row in catalog for reason in row.get('validation_blockers', [])}))
        try:
            first = first_session_after(filed, sessions, calendar_start)
        except (ValueError, TypeError):
            first = None
            blockers.append('invalid_filing_date')
        if not catalog:
            blockers.append('annual_filing_catalog_unverified')
        if not report_date:
            blockers.append('current_annual_report_date_unverified_or_conflicting')
        if not current:
            blockers.append('no_current_annual_GP_Assets_pair')
        elif not any(c['structurally_complete'] for c in current):
            blockers.append('current_annual_pairs_all_structurally_blocked')
        result.append({'cik': cik, 'accession': accn, 'filed': filed, 'form': form,
                       'report_date': report_date, 'first_usable_session': first,
                       'catalog_evidence': catalog, 'candidate_keys': [{'end': c['end'], 'unit': c['unit']} for c in items],
                       'current_period_candidate_count': len(current), 'current_period_structurally_complete_count': sum(c['structurally_complete'] for c in current),
                       'admission_blockers': blockers, 'admitted': False})
    return result


def catalog_receipt_check(raw, binding, cik, origin):
    reasons = []
    if origin.get('sha256') != source.sha(raw):
        reasons.append('catalog_origin_sha_mismatch')
    expected = 'https://data.sec.gov/submissions/' + Path(binding['path']).name
    primary = 'https://data.sec.gov/submissions/CIK' + cik + '.json'
    if origin.get('source') not in (expected, primary):
        reasons.append('catalog_origin_url_mismatch')
    try:
        stamp = datetime.fromisoformat(origin['retrieved_at'].replace('Z', '+00:00'))
        if stamp.utcoffset() is None:
            raise ValueError('Naive catalog origin time')
    except (ValueError, TypeError, KeyError, AttributeError):
        reasons.append('catalog_origin_retrieved_at_invalid')
    return reasons


def catalog_rows(raw, binding, cik):
    obj = source.strict_json(raw)
    if not isinstance(obj, dict):
        raise ValueError('Catalog object required')
    if 'cik' in obj and str(obj['cik']).zfill(10) != cik:
        raise ValueError('Catalog CIK mismatch')
    filings = obj.get('filings', {})
    if not isinstance(filings, dict):
        raise ValueError('Catalog filings object required')
    table = filings.get('recent', obj)
    if not isinstance(table, dict):
        raise ValueError('Catalog table object required')
    required = ('accessionNumber', 'filingDate', 'reportDate', 'form')
    fields = required + ('primaryDocument',)
    if not all(isinstance(table.get(k), list) for k in required):
        raise ValueError('Catalog required columns missing')
    size = len(table['accessionNumber'])
    if any(k in table and (not isinstance(table[k], list) or len(table[k]) != size) for k in fields):
        raise ValueError('Catalog column lengths')
    result = []
    for i in range(size):
        row = dict({k: table[k][i] for k in fields if k in table}, source_path=binding['path'], source_sha256=binding['sha256'], row_index=i)
        problems = []
        if not isinstance(row['accessionNumber'], str) or not re.fullmatch(r'\d{10}-\d{2}-\d{6}', row['accessionNumber']):
            problems.append('invalid_catalog_accession')
        if not isinstance(row['form'], str) or not row['form']:
            problems.append('invalid_catalog_form')
        try:
            filed = source.iso_day(row['filingDate'])
        except (ValueError, TypeError):
            filed = None;problems.append('invalid_catalog_filing_date')
        annual = row['form'] in ('10-K', '10-K/A', '20-F', '20-F/A', '40-F', '40-F/A')
        if annual or row['reportDate']:
            try:
                end = source.iso_day(row['reportDate'])
                if filed is not None and end > filed:
                    problems.append('catalog_report_date_after_filed')
            except (ValueError, TypeError):
                problems.append('invalid_catalog_report_date')
        row['validation_blockers'] = problems
        result.append(row)
    return result



def referenced_archives(raw, cik):
    """Metadata only: primary catalog references are not a complete event guarantee."""
    obj = source.strict_json(raw)
    if not isinstance(obj, dict) or not isinstance(obj.get('filings', {}), dict):
        return [], ['invalid_catalog_archive_container']
    files = obj.get('filings', {}).get('files', [])
    if not isinstance(files, list):
        return [], ['invalid_catalog_archive_list']
    names, reasons = [], []
    for row in files:
        name = row.get('name') if isinstance(row, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r'CIK' + cik + r'-submissions-[0-9]+\.json', name):
            reasons.append('invalid_catalog_archive_reference')
        else:
            names.append(name)
    return sorted(set(names)), sorted(set(reasons))


def archive_coverage(checks):
    result = []
    for row in checks:
        others = [c for c in checks if c['cik'] == row['cik']]
        names = row['referenced_archive_files']
        result.append({'cik': row['cik'], 'catalog_path': row['raw']['path'],
                       'referenced_archive_files_missing': [n for n in names if not any(Path(c['raw']['path']).name == n for c in others)],
                       'referenced_archive_files_unverified': [n for n in names if not any(Path(c['raw']['path']).name == n and c['accepted'] for c in others)],
                       'archive_reference_blockers': row['archive_reference_blockers']})
    return result


def freeze_plan(root=ROOT):
    original_raw = verify({'path': ORIGINAL_PLAN, 'sha256': ORIGINAL_SHA}, root)
    original = source.strict_json(original_raw)
    inv = source.strict_json(verify({'path': INVENTORY, 'sha256': INVENTORY_SHA}, root))
    calplan = source.strict_json(verify({'path': CALENDAR_PLAN, 'sha256': CALENDAR_SHA}, root))
    allow = original['allowed_ciks']
    if len(allow) != 43:
        raise ValueError('Original 43-CIK boundary changed')
    bindings = {}
    def bind(p):
        b = pin(p, root);bindings[b['path']] = b['sha256'];return b
    for p in (ORIGINAL_PLAN, INVENTORY, SELF, TEST, 'work/profitability141_admission_design_review.json', 'work/monthly_profitability141_sources.py', CALENDAR_PLAN, CALENDAR_SESSIONS):
        bind(p)
    for rel, digest in calplan['bindings'].items():
        b = bind(rel)
        if b['sha256'] != digest:
            raise ValueError('Original calendar binding changed')
    chosen, excluded = [], []
    for entry in inv['candidates']:
        if entry['cik'] not in allow:
            continue
        origin = entry['origin_sidecar']
        if not origin.get('verified_url_hash_time'):
            excluded.append({'cik': entry['cik'], 'path': entry['relative_path'], 'reason': 'origin_url_hash_time_sidecar_not_verified', 'sha256': entry['sha256']})
            continue
        b = bind(entry['relative_path'])
        if b['sha256'] != entry['sha256']:
            raise ValueError('Inventory/raw digest mismatch')
        receipt = bind(origin['sidecar_path'])
        if receipt['sha256'] != origin['sidecar_sha256']:
            raise ValueError('Origin sidecar changed')
        chosen.append({'cik': entry['cik'], 'entity_name': entry['entity_name'], 'raw': b, 'origin': receipt})
    metadata = read('migration/top100_local_completion_20261019/INVENTORY.json', root)['trees']['outputs/dynamic_top100']['source']
    catalogs = []
    for cik in allow:
        related = {str(Path(row['raw']['path']).with_name(Path(row['raw']['path']).name.replace('_facts.json', '_submissions.json'))) for row in chosen if row['cik'] == cik}
        related.update('outputs/dynamic_top100/' + rel for rel in metadata if cik in rel and 'submissions' in rel and rel.endswith('.json') and '.provenance.' not in rel)
        for rel in sorted(related):
            p = path(rel, root)
            if not p.is_file():
                continue
            pp = p.with_suffix('.provenance.json')
            catalogs.append({'cik': cik, 'raw': bind(p), 'origin': bind(pp) if pp.is_file() else None})
    plan = {'schema_version': 1, 'purpose': 'offline_source_admission_candidates_only', 'created_utc': now(),
            'original_source_plan': {'path': ORIGINAL_PLAN, 'sha256': ORIGINAL_SHA}, 'inventory': {'path': INVENTORY, 'sha256': INVENTORY_SHA},
            'allowed_ciks': allow, 'original_symbols': original['candidate_universe'], 'symbol_cik_locating_evidence': original['candidate_cik_locating_evidence'],
            'sources': chosen, 'excluded_inventory_sources': excluded, 'filing_catalogs': catalogs, 'bindings': bindings,
            'output_dir': OUTPUT, 'calendar': {'basis': 'pinned XNYS plus locally hash-verified official closure overlay; not daily exchange certification', 'start': '2000-01-01', 'end': '2027-12-31', 'comparison_start': '2015-01-01', 'comparison_end': '2027-12-31', 'original_plan_sha256': CALENDAR_SHA},
            'contract': {'taxonomy': 'us-gaap', 'tags': list(TAGS), 'form': '10-K', 'gross_profit_inclusive_days': [330, 400], 'assets': 'instant without start', 'join': ['cik', 'accn', 'filed', 'form', 'unit', 'end'], 'duplicates': 'same numeric Decimal values may duplicate; conflicting values, start dates or supplied FY/FP block entire key', 'nonpositive_gross_profit': 'preserved', 'nonpositive_assets': 'preserved but blocked', 'time': 'retain each accession and filed vintage; first pinned session strictly after filed; never backfill comparative periods', 'fallbacks': 'none: no quarterly aggregation, derived GrossProfit, IFRS, amendments or currency conversion'},
            'source_policy': 'Require exact local raw and original URL/hash/time sidecar. Retain all valid provenance-backed sources per CIK; compare duplicates across sources. No requests.',
            'event_history_policy': 'Preserve missing and invalid annual events; select latest known event before validity checks. Local catalog history completeness is not certified, and absence cannot establish no newer annual report.',
            'context_policy': 'Catalog identity/form/filing/reportDate may be checked. No qualifying original 10-K context document has been registered in this plan; original_filing_context_verified is always false and admitted is always false.',
            'artifacts': ['CALENDAR.json', 'SOURCE_CHECKS.json', 'CATALOG_CHECKS.json', 'ANNUAL_CANDIDATES.json', 'ANNUAL_FILING_EVENTS.json', 'COVERAGE.json', 'SUMMARY.json'],
            'prohibited': ['ratio', 'factor', 'rank', 'portfolio', 'returns', 'new requests', 'frozen strategy or plan changes', 'observer or ledger access'],
            'missing_policy': 'Keep every original 43 CIK and 73 symbol; report missing sources/fields/contexts without deleting companies.'}
    write(OUTPUT + '/EXTRACTION_PLAN.json', plan, root)
    return pin(OUTPUT + '/EXTRACTION_PLAN.json', root)


def calendar_from_plan(plan, root=ROOT):
    spec = importlib.util.spec_from_file_location('profitability141_pinned_calendar', path('outputs/monthly_earnings_repair/research_calendar.py', root))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cal = module.market_calendar(start=plan['calendar']['start'], end=plan['calendar']['end'])
    sessions = [str(d.date()) for d in cal.sessions]
    prior = read(CALENDAR_SESSIONS, root)
    if prior['plan_sha256'] != CALENDAR_SHA or prior['sessions'] != [s for s in sessions if s >= plan['calendar']['comparison_start']]:
        raise ValueError('Calendar does not reproduce registered sessions')
    return sessions


def execute(plan_path, expected_sha, root=ROOT):
    plan_raw = verify({'path': plan_path, 'sha256': expected_sha}, root)
    plan = source.strict_json(plan_raw)
    if plan.get('purpose') != 'offline_source_admission_candidates_only' or plan['output_dir'] != OUTPUT:
        raise ValueError('Wrong extraction plan')
    original = source.strict_json(verify({'path': ORIGINAL_PLAN, 'sha256': ORIGINAL_SHA}, root))
    if plan['allowed_ciks'] != original['allowed_ciks'] or plan['original_symbols'] != original['candidate_universe']:
        raise ValueError('Original CIK/symbol boundary changed')
    for rel, digest in plan['bindings'].items():
        verify({'path': rel, 'sha256': digest}, root)
    sessions = calendar_from_plan(plan, root)
    documents = defaultdict(list);checks = []
    for row in plan['sources']:
        raw = verify(row['raw'], root);receipt = source.strict_json(verify(row['origin'], root))
        reasons = receipt_check(raw, row['cik'], receipt)
        if len(raw) > source.MAX_BYTES:
            reasons.append('companyfacts_exceeds_40MiB')
        try:
            source.validate_facts(raw, row['cik'])
        except (ValueError, TypeError) as exc:
            reasons.append('invalid_companyfacts_' + type(exc).__name__)
        if not reasons:
            documents[row['cik']].append((raw, row['raw']))
        checks.append(dict(row, status='verified_original_url_hash_time' if not reasons else 'blocked', reasons=reasons, original_http_receipt_complete=False))
    catalogs = defaultdict(lambda: defaultdict(list));catchecks = []
    for row in plan['filing_catalogs']:
        reasons = [];raw = verify(row['raw'], root)
        if row['origin'] is None:
            reasons.append('catalog_origin_sidecar_missing')
        else:
            origin = source.strict_json(verify(row['origin'], root))
            reasons.extend(catalog_receipt_check(raw, row['raw'], row['cik'], origin))
        try:
            entries = catalog_rows(raw, row['raw'], row['cik'])
        except (ValueError, TypeError):
            entries = [];reasons.append('invalid_catalog')
        if not reasons:
            for entry in entries:
                catalogs[row['cik']][str(entry['accessionNumber'])].append(entry)
        try:
            archive_names, archive_reasons = referenced_archives(raw, row['cik'])
        except (ValueError, TypeError):
            archive_names, archive_reasons = [], ['invalid_catalog_archive_json']
        catchecks.append(dict(row, referenced_archive_files=archive_names, archive_reference_blockers=archive_reasons, reasons=reasons, rows=len(entries), blocked_rows=sum(bool(e['validation_blockers']) for e in entries), rejected_row_reasons=dict(Counter(r for e in entries for r in e['validation_blockers'])), accepted=not reasons, original_http_receipt_complete=False, proves_original_XBRL_context=False))
    all_candidates = [];coverage = []
    for cik in plan['allowed_ciks']:
        candidates, rejected = extract_company(cik, documents[cik], sessions, catalogs[cik], plan['calendar']['start'])
        all_candidates.extend(candidates)
        reasons = []
        if not documents[cik]:reasons.append('no_verified_local_companyfacts')
        if not candidates:reasons.append('no_annual_GrossProfit_Assets_candidate_keys')
        if not any(c['structurally_complete'] for c in candidates):reasons.append('no_structurally_complete_annual_pairs')
        reasons.extend(['original_10K_context_not_verified', 'historical_security_identity_not_verified'])
        coverage.append({'cik': cik, 'symbols': [r['symbol'] for r in plan['symbol_cik_locating_evidence'] if cik in r['candidate_ciks']], 'verified_source_files': len(documents[cik]), 'annual_candidate_keys': len(candidates), 'structurally_complete_keys': sum(c['structurally_complete'] for c in candidates), 'catalog_verified_keys': sum(c['filing_catalog_verified'] for c in candidates), 'context_verified_keys': 0, 'admitted_keys': 0, 'rejected_row_counts': rejected, 'gap_reasons': reasons, 'year_coverage': sorted({c['end'][:4] for c in candidates if c['structurally_complete']})})
    artifacts = {'CALENDAR.json': {'plan_sha256': expected_sha, 'calendar': plan['calendar'], 'sessions': sessions, 'prior_sessions_reproduced': True},
                 'SOURCE_CHECKS.json': {'plan_sha256': expected_sha, 'checks': checks},
                 'CATALOG_CHECKS.json': {'plan_sha256': expected_sha, 'checks': catchecks, 'event_history_completeness': 'not_certified', 'archive_coverage': archive_coverage(catchecks)},
                 'ANNUAL_CANDIDATES.json': {'plan_sha256': expected_sha, 'candidates': all_candidates},
                 'ANNUAL_FILING_EVENTS.json': {'plan_sha256': expected_sha, 'event_history_completeness': 'not_certified', 'events': annual_filing_events(plan['allowed_ciks'], catalogs, all_candidates, sessions, plan['calendar']['start']), 'selection_rule': 'A downstream strategy must choose the latest then-known annual filing event before evaluating data validity; a missing, conflicting or unverified current event must not silently reuse an older valid value.'},
                 'COVERAGE.json': {'plan_sha256': expected_sha, 'all_43_ciks_preserved': len(coverage) == 43, 'all_73_symbols_preserved': plan['original_symbols'], 'coverage': coverage}}
    for name, artifact in artifacts.items():
        write(OUTPUT + '/' + name, artifact, root)
    summary = {'schema_version': 1, 'created_utc': now(), 'status': 'source_candidates_extracted_not_admitted', 'plan_sha256': expected_sha,
               'original_source_plan_sha256': ORIGINAL_SHA, 'allowed_ciks': len(coverage), 'original_symbols': len(plan['original_symbols']), 'ciks_with_verified_source': sum(bool(r['verified_source_files']) for r in coverage), 'ciks_with_structurally_complete_pairs': sum(bool(r['structurally_complete_keys']) for r in coverage), 'annual_candidate_keys': len(all_candidates), 'structurally_complete_keys': sum(c['structurally_complete'] for c in all_candidates), 'catalog_verified_keys': sum(c['filing_catalog_verified'] for c in all_candidates), 'original_context_verified_keys': 0, 'admitted_keys': 0, 'structural_blocker_counts': dict(Counter(r for c in all_candidates for r in c['structural_blockers'])), 'artifacts': {name: pin(OUTPUT + '/' + name, root) for name in artifacts},
               'event_history_completeness': 'not_certified', 'referenced_archive_files_missing': sum(len(r['referenced_archive_files_missing']) for r in archive_coverage(catchecks)), 'referenced_archive_files_unverified': sum(len(r['referenced_archive_files_unverified']) for r in archive_coverage(catchecks)), 'source_requests': 0, 'financial_ratios_computed': 0, 'rankings_computed': 0, 'return_computations': 0, 'strategy_state_changed': False, 'remaining': ['Independent review of annual source values and matching context', 'Register and verify original filing contexts for exact accession/period/unit/consolidation', 'Historical identity admission remains separate'], 'limitations': ['Companyfacts is a present snapshot of historic filings, not a vintage archive.', 'No original filing document has been registered; catalog matching is not context certification.', 'Pinned XNYS with verified official closure overlay is a calendar model, not daily exchange certification.', 'Catalog event history is not certified complete; absence does not prove no newer annual filing.']}
    write(OUTPUT + '/SUMMARY.json', summary, root)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__);sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('plan');run = sub.add_parser('extract');run.add_argument('--plan', default=OUTPUT + '/EXTRACTION_PLAN.json');run.add_argument('--plan-sha256', required=True)
    args = parser.parse_args(argv)
    result = freeze_plan() if args.command == 'plan' else execute(args.plan, args.plan_sha256)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
