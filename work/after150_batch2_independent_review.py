"""Independent batch2 source audit; no batch2 producer/helper imports.

Only the old independent auditor's generic grid/date/decimal/Reader primitives
are reused. Date columns are rediscovered from HTML headers, before comparing
published locators. Real I/O requires explicit frozen PLAN/EXTRACTIONS hashes.
"""
import argparse
from bisect import bisect_right
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlsplit
from bs4 import BeautifulSoup, NavigableString
from after149_batch1_independent_review import (
    Reader, grid_from_html, parsed_dates, signed_text, norm, sha, require)

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'outputs/monthly_financial_coverage_batch2_v1/PLAN.json'
EXTRACTIONS = 'outputs/monthly_financial_coverage_batch2_v1/EXTRACTIONS.json'
CONTEXT = 'work/after150_next_source_context.json'
POLICY = 'work/after150_batch2_admission_policy.json'
SESSIONS = 'outputs/monthly_forward_guidance_preflight_v1/SESSIONS.json'
SELF = 'work/after150_batch2_independent_review.py'
TEST = 'work/test_after150_batch2_independent_review.py'
OUTPUT = 'work/after150_batch2_independent_review.json'
FIELDS = ('annual_net_income', 'current_total_assets', 'prior_total_assets')
PRE_HOLD = 'KLAC_FY2011_preformatted_column_geometry_unverified'
ASC_HOLD = 'KLAC_FY2019_ASC606_current_comparison_not_restated__average_assets_use_hold'
INTC_HOLD = 'annual_start_not_directly_quoted_requires_source_use_assumption'
ASSUMPTION = 'INTC_consecutive_annual_boundaries_v1'
ANNUAL = re.compile(r'\b(?:twelve|12)\s+months?\s+ended\b', re.I)
SHORT = re.compile(r'\b(?:(?:three|six|nine|3|6|9)\s+months?|quarters?)\s+ended\b', re.I)
NI = re.compile(r'net\s+(?:income|loss)(?:\s*\((?:income|loss)\))?', re.I)
BODY = re.compile(r'(?:assets|current assets|(?:net |total )?revenues?|(?:net )?sales|cash(?: and cash equivalents)?|net (?:income|loss)(?:\s*\((?:income|loss)\))?)\s*:?', re.I)
PREFIX = r'(?:condensed\s+)?consolidated\s+(?:(?:unaudited|condensed|summary)\s+)*'
INCOME = re.compile(PREFIX + r'statements?\s+of\s+(?:operations|income|earnings)', re.I)
BALANCE = re.compile(PREFIX + r'balance\s+sheets?', re.I)
BAD = re.compile(r'non[\s\-\u2010-\u2015]*gaap|pro\s+forma|reconciliation|as\s+adjusted', re.I)
OTHER_CURRENCY = re.compile(r'[€£¥]|\b(?:EUR|GBP|JPY|CAD|AUD|HKD)\b', re.I)
PUNCT = re.compile(r'[\s$()\-\u2212]+')


def source_files():
    return [SELF, TEST, 'work/after149_batch1_independent_review.py']


def relative(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    require(not p.is_symlink(), 'Symlink path')
    return str(p.resolve().relative_to(ROOT))


def external_title(table, pattern):
    parts = []
    for node in table.previous_elements:
        if getattr(node, 'name', None) == 'table':
            break
        if not isinstance(node, NavigableString):
            continue
        if node.find_parent('table') is not None:
            break
        text = norm(str(node))
        if text:
            parts.insert(0, text)
        joined = ' '.join(parts)
        if pattern.search(joined):
            return joined
        if len(parts) > 25 or len(joined) > 1400:
            break
    return None


def headers_for_statement(table, context, table_index, row, annual):
    require(table.find('table') is None, 'Nested statement')
    cells, grid = grid_from_html(table)
    frozen = [x for x in context['context']['statement_tables'] if x['table'] == table_index]
    require(len(frozen) == 1, 'Statement context missing or duplicate')
    stop = frozen[0]['header_stop_row']
    require(type(stop) is int and 0 < stop <= row, 'Invalid header boundary')
    headers = [c for (r, p), c in cells.items() if r < stop]
    require(not any(BODY.fullmatch(c['text']) for c in headers if c['physical_cell'] == 0), 'Financial row inside headers')
    for h in frozen[0]['headers']:
        actual = cells.get((h['row'], h['physical_cell']))
        require(actual is not None and actual['row'] < stop and all(actual[k] == h[k] for k in ('columns', 'rowspan', 'text')), 'Frozen structural header changed')
    pattern = INCOME if annual else BALANCE
    titles = [c['text'] for c in headers if pattern.search(c['text'])]
    if not titles:
        outside = external_title(table, pattern)
        titles = [outside] if outside else []
    require(len(titles) == 1 and not BAD.search(titles[0]), 'Formal consolidated title missing or ambiguous')
    require(not any(BAD.search(c['text']) for c in headers), 'Non-GAAP statement header')
    units = [c for c in headers if re.search(r'\bin\s+(?:thousands|millions|billions)\b', c['text'], re.I)]
    names = {x.lower() for c in units for x in re.findall(r'\bin\s+(thousands|millions|billions)\b', c['text'], re.I)}
    wanted = 'millions' if context['symbol'] == 'INTC' else 'thousands'
    require(names == {wanted}, 'Missing or conflicting source scale')
    text = table.get_text(' ', strip=True)
    require('$' in text and not OTHER_CURRENCY.search(text), 'Currency evidence conflict')
    return cells, grid, headers, units, titles[0], stop


def derive_date_groups(headers, target_row, annual):
    """Map all date-bearing columns from raw stacked headers, not locator columns."""
    possible = sorted({x for c in headers for x in c['columns']})
    groups = {}
    for col in possible:
        hs = [c for c in headers if col in c['columns']]
        text = ' '.join(c['text'] for c in hs)
        if annual and (not ANNUAL.search(text) or SHORT.search(text)):
            continue
        ds = parsed_dates(text)
        if not ds:
            continue
        require(len(ds) == 1, 'Conflicting date headers')
        day = next(iter(ds))
        year_cells = [c for c in hs if re.search(r"\b(?:19|20)\d{2}\b", c["text"])]
        require(len(year_cells) == 1, "Duplicated or stacked year anchors")
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
        require(years == {day[:4]}, 'Conflicting header years')
        groups.setdefault(day, []).append(col)
    require(len(groups) == 2, 'Need exactly two distinct annual date groups')
    for columns in groups.values():
        require(columns == list(range(min(columns), max(columns) + 1)), 'Disjoint date group')
    if annual:
        parents = [c for c in headers if ANNUAL.search(c['text']) and not SHORT.search(c['text'])]
        require(len(parents) == 1 and all(set(cols) <= set(parents[0]['columns']) for cols in groups.values()), 'Ambiguous annual parent')
        domain = set(parents[0]['columns'])
    else:
        domain = set(range(1, max(max(c['columns']) for c in target_row) + 1))
    # Columns not bearing a date may only be punctuation/separators; assign
    # ownership by nearest date interval, rejecting equidistant ambiguity.
    ownership = {}
    for col in domain:
        distances = {day: min(abs(col-x) for x in cols) for day, cols in groups.items()}
        closest = [day for day, dist in distances.items() if dist == min(distances.values())]
        ownership[col] = closest[0] if len(closest) == 1 else None
    return groups, ownership


def source_field(soup, context, field):
    annual = field == 'annual_net_income'
    spec = context['context_summary']['net_income_locator' if annual else 'assets_locator']
    ti, row = spec['table'], spec['row']
    tables = soup.find_all('table')
    require(type(ti) is int and 0 <= ti < len(tables), 'Invalid source table')
    cells, grid, headers, units, title, stop = headers_for_statement(tables[ti], context, ti, row, annual)
    labels = [c for (r,p),c in cells.items() if p == 0 and
              (NI.fullmatch(c['text'].rstrip(':')) if annual else c['text'].rstrip(':').lower() == 'total assets')]
    require(len(labels) == 1 and labels[0]['row'] == row and labels[0]['rowspan'] == 1, 'Exact total row missing or ambiguous')
    require(spec['label_cell'] == 0 and norm(spec['label']).rstrip(':') == labels[0]['text'].rstrip(':'), 'Context label changed')
    ordered = sorted([c for (r,p),c in cells.items() if r == row], key=lambda c:c['physical_cell'])
    groups, ownership = derive_date_groups(headers, ordered, annual)
    current = context['official_actual_period_end']
    prior = context['context_summary']['comparison_assets_date']
    require(set(groups) == {current, prior}, 'Raw fiscal endpoints differ from catalog/context')
    require(groups[current] == spec['current_header_columns'] and groups[prior] == spec['comparison_header_columns'], 'Independent header columns differ')
    ending = prior if field == 'prior_total_assets' else current
    columns = groups[ending]
    require(all((row,col) in grid and grid[row,col]['row'] == row for col in columns), 'Cross-row or absent value')
    chosen = [c for c in ordered if set(c['columns']) & set(columns)]
    require(chosen, 'No physical amount cells')
    # Reconstruct a complete syntactic amount only within the independently
    # assigned date domain. KLAC confines signs to its explicit date span.
    lo, hi = ordered.index(chosen[0]), ordered.index(chosen[-1])
    for idx in (lo-1, hi+1):
        if 0 <= idx < len(ordered):
            c = ordered[idx]
            if c['text'] and PUNCT.fullmatch(c['text']):
                require(context['symbol'] == 'INTC' and all(ownership.get(x) == ending for x in c['columns']), 'Sign outside period ownership')
                chosen.append(c)
    chosen.sort(key=lambda c:c['physical_cell'])
    for c in chosen:
        require(c['rowspan'] == 1 and all(ownership.get(x) == ending for x in c['columns']), 'Cross-period or cross-row amount')
        if context['symbol'] == 'KLAC':
            require(set(c['columns']) <= set(columns), 'KLAC amount outside date span')
    display = ' '.join(c['text'] for c in chosen if c['text'])
    value = signed_text(display)
    require(Decimal(value).is_finite(), 'Nonfinite amount')
    if not annual:
        require(Decimal(value) > 0, 'Nonpositive total assets')
    dollars = [c for (r,p),c in cells.items() if r >= stop and '$' in c['text'] and set(c['columns']) & set(columns)]
    require(dollars or context['symbol'] == 'INTC', 'Selected date currency evidence missing')
    selected_headers = [c for c in headers if set(c['columns']) & set(columns)]
    return dict(table=ti,row=row,value_decimal=value,displayed_amount=display,
                scale='1000000' if context['symbol'] == 'INTC' else '1000',as_of=ending,
                columns=columns,physical_cells=chosen,header_cells=selected_headers,
                scale_cells=units,statement_title=title,all_date_groups=groups)


def verify_published_cell(cell, reading, field, event, expected_period):
    require(cell is not None, 'Unexpected null outside declared PRE hold')
    amount = Decimal(cell['value_decimal'])
    require(amount.is_finite() and amount == Decimal(reading['value_decimal']), 'Numeric amount mismatch')
    require(Decimal(signed_text(cell['displayed_amount'])) == amount, 'Displayed amount/sign mismatch')
    require(cell['scale'] == reading['scale'] and cell['currency_literal'] == '$' and cell.get('currency_iso') is None, 'Published units mismatch')
    require(cell['source'] == event['source'] and cell['available_from'] == event['available_from'], 'Field source/availability mismatch')
    require(cell['as_of'] == reading['as_of'] and cell['fact_admitted'] is False, 'Field date/certification mismatch')
    require(cell['period'] == (expected_period if field == 'annual_net_income' else None), 'Field period mismatch')
    loc = cell['locator']
    require(loc.get('table',loc.get('table_index_zero_based')) == reading['table'] and loc.get('row',loc.get('row_index_zero_based')) == reading['row'], 'Published table/row mismatch')
    require(loc.get('header_columns',loc.get('logical_columns')) == reading['columns'], 'Published date columns mismatch')
    published = loc['physical_cells']
    require(len(published) == len(reading['physical_cells']), 'Physical-cell count mismatch')
    for p,a in zip(published,reading['physical_cells']):
        require(p.get('physical_cell',p.get('physical_cell_index_zero_based')) == a['physical_cell'], 'Physical-cell identity mismatch')
        require(p.get('row',p.get('row_index_zero_based')) == a['row'] and p.get('columns',p.get('logical_columns')) == a['columns'] and p['rowspan'] == 1, 'Physical-cell geometry mismatch')
        require(p.get('markup_sha256',p.get('parsed_cell_markup_sha256')) == a['markup_sha256'], 'Physical-cell markup mismatch')
        require(p['text'] == a['text'], 'Physical-cell source text mismatch')
    return dict(is_null=False,verified=True,source_value_present=True,value_and_sign_verified=True,
                units_verified=True,period_verified=True,source_locator_verified=True,independent_source_reading=reading)


def receipt_evidence(reader, context, raw):
    ref = context['context']['receipt']
    doc = reader.load(ref['path'], ref['sha256'])
    candidates = doc if isinstance(doc,list) else ([doc] if "sha256" in doc else list(doc.values()))
    matches = [x for x in candidates if isinstance(x,dict) and x.get('sha256') == sha(raw)]
    require(len(matches) == 1, 'Raw receipt missing or ambiguous')
    item = matches[0]
    require(item.get('status',item.get('http_status')) == 200 and item['url'] == context['source_url'], 'Receipt status/URL mismatch')
    if 'bytes' in item:
        require(item['bytes'] == len(raw), 'Receipt length mismatch')
    stamp = datetime.fromisoformat(item['retrieved_at'].replace('Z','+00:00'))
    require(stamp.tzinfo is not None and item['retrieved_at'] == ref['retrieved_at'], 'Receipt retrieval timestamp')
    return dict(path=relative(ref['path']),sha256=ref['sha256'],url=item['url'],http_status=200,retrieved_at=item['retrieved_at'])


def periods(context, event, policy):
    symbol, year = context['symbol'],context['fiscal_year']
    end = context['official_actual_period_end']
    prior = context['context_summary']['comparison_assets_date']
    start = (date.fromisoformat(prior)+timedelta(days=1)).isoformat()
    days = (date.fromisoformat(end)-date.fromisoformat(start)).days+1
    p = dict(period_start=start,period_end=end,days_inclusive=days)
    require(event['periods']['current'] == p, 'Current annual period mismatch')
    require(event['periods']['comparison'] == dict(period_start=None,period_end=prior,days_inclusive=None), 'Comparison endpoint metadata mismatch')
    meta = context['context_summary']['current_annual_period']
    require(meta['end'] == end, 'Context period end mismatch')
    if symbol == 'INTC':
        setting = policy['periods']['INTC_52_53_week_assumption']
        require(setting['enabled'] is True and setting['not_a_quoted_fact'] is True and setting['assumption_id'] == ASSUMPTION, 'INTC assumption policy missing')
        require(days in (364,371) and meta['start'] is None and meta['candidate_start_from_prior_end_plus_one'] == start, 'INTC derived boundary invalid')
        require(event['periods']['start_basis'] == 'derived_same_report_prior_annual_end_plus_one_not_directly_quoted' and event['periods']['start_assumption_certified'] is False, 'Derived boundary presented as certified')
        require(INTC_HOLD in event['context_holds'], 'INTC contextual hold lost')
    else:
        require(end == '%d-06-30'%year and prior == '%d-06-30'%(year-1) and start == '%d-07-01'%(year-1) and days in (365,366), 'KLAC calendar boundary invalid')
        require(meta['start'] == start, 'KLAC interpreted start mismatch')
    return p


def review_pre(soup, context, event):
    require(context['symbol'] == 'KLAC' and context['fiscal_year'] == 2011 and PRE_HOLD in event['context_holds'], 'Unexpected PRE hold')
    require(context['context_summary']['status'] == 'preformatted_original_requires_separate_column_geometry_review', 'PRE status changed')
    blocks = soup.find_all('pre')
    evidence = []
    for key,pattern,label in [('net_income_locator',INCOME,NI),('assets_locator',BALANCE,re.compile(r'total assets',re.I))]:
        spec = context['context_summary'][key]
        require(0 <= spec['pre_block'] < len(blocks), 'PRE block absent')
        text = blocks[spec['pre_block']].get_text()
        require(pattern.search(norm(text)), 'PRE formal statement absent')
        require(any(label.match(norm(line)) for line in text.splitlines()), 'PRE label absent')
        evidence.append(dict(pre_block=spec['pre_block'],formal_statement_present=True,target_label_present=True,monetary_columns_parsed=False))
    require(event['periods'] == dict(current=dict(period_start=None,period_end=context['official_actual_period_end'],days_inclusive=None),comparison=dict(period_start=None,period_end=None,days_inclusive=None)), 'PRE period fabricated')
    out = {}
    for field in FIELDS:
        require(event['fields'][field] is None, 'Frozen PRE null was filled')
        gaps = [g for g in event['gaps'] if g['field'] == field]
        require(len(gaps) == 1 and gaps[0]['reason'] == PRE_HOLD, 'PRE gap reason changed')
        out[field] = dict(is_null=True,null_reason_verified=True,source_value_present=None,frozen_null_preserved=True,
                          null_classification='original_report_present_unparsed_PRE_column_geometry',frozen_gap=gaps[0],source_metadata=evidence)
    return out


def review_event(reader, row, event, context, sessions, policy):
    require(all(event[k] == context[k] == row[k] for k in ('event_id','symbol','fiscal_year')), 'Event identity mismatch')
    require(event['event_id'] == event['symbol']+'_'+context['publication_date'], 'Event ID/date mismatch')
    raw = reader.read(row['source_path'],row['source_sha256'])
    require(relative(context['evidence_path']) == relative(row['source_path']) and context['evidence_sha256'] == sha(raw), 'Context source mismatch')
    require(relative(event['source']['path']) == relative(row['source_path']) and event['source']['sha256'] == sha(raw) and event['source']['url'] == context['source_url'], 'Published source mismatch')
    receipt = receipt_evidence(reader,context,raw)
    if 'receipt' in event['source']:
        require(relative(event['source']['receipt']) == receipt['path'], 'Published receipt path mismatch')
    soup = BeautifulSoup(raw,'html.parser')
    for node in soup(['script','style']):
        node.decompose()
    text = norm(soup.get_text(' ',strip=True))
    symbol = event['symbol']
    issuer = r'Intel Corporation' if symbol == 'INTC' else r'KLA(?:[\s-]+Tencor)? Corporation'
    require(re.search(issuer,text,re.I) and re.search(r'NASDAQ\s*[:(]\s*'+symbol+r'\b',text,re.I), 'Raw identity anchor missing')
    url = urlsplit(context['source_url'])
    require(url.scheme == 'https' and not url.username and not url.password and url.hostname in ({'www.intc.com','intc.com'} if symbol == 'INTC' else {'ir.kla.com'}), 'Issuer URL mismatch')
    release, available = context['publication_date'],context['available_session']
    require(release in parsed_dates(text), 'Release date absent from source')
    require(event['publication_date'] == release and event.get('release_date',release) == release and event['available_from'] == available and context['official_actual_period_end'] < release < available, 'Release/availability mismatch')
    require(sessions == sorted(set(sessions)) and sessions, 'Calendar invalid')
    clock = sessions[0] <= release < sessions[-1]
    if clock:
        require(sessions[bisect_right(sessions,release)] == available, 'Not first next pinned session')
    require(set(event['fields']) == set(FIELDS) and all(event[k] is False for k in ('fact_admitted','model_admitted','strategy_admitted')), 'Fields or admission flags changed')
    require(event['accounting_review_flags'] == context['context_summary'].get('accounting_attention',[]), 'Accounting context lost')
    if symbol == 'KLAC':
        require(event['original_context'] == context and event['accounting_context'] == context['context'].get('accounting_contexts_numbers_redacted',[]), 'Original context lost')
        require(event['receipt'] == context['context']['receipt'], 'Receipt metadata changed')
        if context['fiscal_year'] == 2019:
            require(ASC_HOLD in event['context_holds'], 'ASC606 hold lost')
    is_pre = symbol == 'KLAC' and context['fiscal_year'] == 2011
    if is_pre:
        reviewed = review_pre(soup,context,event)
        p = None
    else:
        p = periods(context,event,policy)
        readings = {f:source_field(soup,context,f) for f in FIELDS}
        reviewed = {f:verify_published_cell(event['fields'][f],readings[f],f,event,p) for f in FIELDS}
        require(not event['gaps'], 'Nonnull event has unexpected gaps')
        if symbol == 'INTC':
            # Check only the selected historical statements, not future outlook
            # (which may legitimately describe a different 53-week year).
            header_text = ' '.join(c['text'] for r in readings.values() for c in r['header_cells'])
            require(not re.search(r'transition period|stub period|change.*fiscal year|from .+? to .+?',header_text,re.I), 'Contradictory fiscal boundary evidence')
    result = dict(event_id=event['event_id'],symbol=symbol,fiscal_year=event['fiscal_year'],passed=True,
                  metadata_verified=True,identity_verified=True,periods_verified=not is_pre,currency_verified=not is_pre,
                  metadata_scope='null classification only; period and currency columns unverified' if is_pre else 'report-local identity, annual dates, dollar units, source locators',
                  clock_verified=clock,clock_scope='first_pinned_next_session' if clock else 'outside_pinned_calendar_not_certified',
                  release_date=release,available_from=available,receipt=receipt,source=event['source'],fields=reviewed,
                  retained_context_holds=event['context_holds'],first_public_version_verified=False,certified_daily_security_master=False,source_admitted=False)
    if symbol == 'INTC':
        result.update(boundary_assumption_verified=True,assumption_id=ASSUMPTION,derived_start=p['period_start'],start_is_source_quoted=False,
                      boundary_evidence={f:dict(table=readings[f]['table'],date=readings[f]['as_of'],headers=readings[f]['header_cells']) for f in FIELDS},
                      boundary_scope='Same-report explicitly annual column and two printed dates; 52/53-week consecutive-boundary policy, not a first-vintage or exhaustive correction-history certificate.')
    elif not is_pre:
        result.update(annual_start_interpretation='July1 from explicit Twelve Months Ended June30',start_is_source_quoted=False)
    return result


def verify_population(plan, data, context, policy):
    docs = context['documents']
    require(len(docs) == 32 and len({x['event_id'] for x in docs}) == 32 and Counter(x['symbol'] for x in docs) == {'INTC':16,'KLAC':16}, 'Full catalog population')
    chosen = [x for x in docs if x.get('html_context_reviewed') is True]
    require(len(chosen) == 19 and Counter(x['symbol'] for x in chosen) == {'INTC':3,'KLAC':16}, 'Extraction scope changed')
    require([x['event_id'] for x in plan['records']] == [x['event_id'] for x in chosen] == [x['event_id'] for x in data['events']], 'Event order/population differs')
    require(plan['catalog_ids'] == data['catalog_ids'] == [x['event_id'] for x in docs] and plan['catalog_events'] == 32, 'Catalog dropped/reordered')
    projection = [dict(event_id=x['event_id'],symbol=x['symbol'],fiscal_year=x['fiscal_year'],publication_date=x['publication_date'],available_session=x['available_session'],official_actual_period_end=x['official_actual_period_end'],source_path=relative(x.get('source',{}).get('path',x['evidence_path'])),source_sha256=x.get('source',{}).get('sha256',x['evidence_sha256']),html_context_reviewed=x.get('html_context_reviewed') is True,retained_context_gap=x.get('gap')) for x in docs]
    require(policy['fixed_catalog_metadata'] == projection and policy['required_fields'] == list(FIELDS), 'Policy catalog differs')
    require(data['counts'] == dict(events=19,by_symbol={'INTC':3,'KLAC':16},nonnull_fields=sum(e['fields'][f] is not None for e in data['events'] for f in FIELDS)), 'Counts differ')
    return {x['event_id']:x for x in chosen}


def audit(plan_sha, extractions_sha, publish=False):
    require(all(re.fullmatch(r'[0-9a-f]{64}',x or '') for x in (plan_sha,extractions_sha)), 'Explicit frozen hashes required')
    reader = Reader()
    plan,data = reader.load(PLAN,plan_sha),reader.load(EXTRACTIONS,extractions_sha)
    require(data['plan_sha256'] == plan_sha and plan['context'] == CONTEXT and plan['policy'] == POLICY, 'Frozen plan linkage')
    for path,digest in {**plan['inputs'],**plan['code']}.items():
        reader.read(path,digest)
    for path in source_files():
        reader.read(path)
    context,policy = reader.load(CONTEXT),reader.load(POLICY)
    require(policy['source_scope']['source_context'] == plan['catalog_ref'] == data['catalog_ref'] == dict(path=CONTEXT,sha256=reader.hashes[CONTEXT]), 'Context reference mismatch')
    require(policy['policy_id'] == 'intc_klac_annual_ni_same_report_assets_batch2_v1', 'Wrong policy')
    for path,digest in policy.get('input_hashes',{}).items():
        reader.read(path,digest)
    sessions = reader.load(SESSIONS)['sessions']
    contexts = verify_population(plan,data,context,policy)
    events,failures = [],[]
    for row,event in zip(plan['records'],data['events']):
        try:
            events.append(review_event(reader,row,event,contexts[row['event_id']],sessions,policy))
        except Exception as exc:
            failures.append(dict(event_id=row['event_id'],error_type=type(exc).__name__,error=str(exc)))
    fields = [f for e in events for f in e['fields'].values()]
    nonnull,nulls = sum(not f['is_null'] for f in fields),sum(f['is_null'] for f in fields)
    passed = not failures and len(events) == 19 and nonnull == 54 and nulls == 3
    for path,digest in list(reader.hashes.items()):
        reader.read(path,digest)
    result = dict(schema_version=1,stage='independent_source_value_and_null_classification_review',created_utc=datetime.now(timezone.utc).isoformat(),passed=passed,
                  plan_sha256=plan_sha,extractions_sha256=extractions_sha,event_reviews=events,failures=failures,
                  scope='19 fixed reports, 57 source positions; preserve three KLAC2011 PRE nulls and the full32 catalog. No financial-use admission.',
                  counters=dict(events=len(events),published_nonnull_values_verified=nonnull,null_fields_reviewed=nulls,by_symbol=dict(Counter(e['symbol'] for e in events)),input_hashes_verified=len(reader.hashes),boundary_assumptions_verified=sum(e.get('boundary_assumption_verified') is True for e in events)),
                  input_hashes=reader.hashes,source_hashes={p:reader.hashes[p] for p in source_files()},runtime=sys.version,
                  independence='Generic primitives from prior independent auditor; own raw-header column derivation and physical-cell verification; no batch2 producer/helper imports.',
                  limitations=['KLAC2011 original PRE statements exist; monetary columns, dates and units remain unparsed, not absent.',
                               'INTC starts use the explicit consecutive-annual-boundary assumption; they are not quoted source facts.',
                               'KLAC2019 accounting hold and all earlier source decisions remain unchanged.',
                               'Outside-calendar availability, first-public versions, full correction history, ISO currency and continuous security identity are not newly certified.'],
                  financial_calculations=0,factor_calculations=0,returns_computed=0,new_admissions=0)
    if publish:
        require(passed, 'Review failed; no success receipt published')
        with (ROOT/OUTPUT).open('x') as stream:
            json.dump(result,stream,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(passed=passed,counters=result['counters'],failures=failures,receipt_sha256=sha((ROOT/OUTPUT).read_bytes()) if publish else None),ensure_ascii=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--extractions-sha',required=True)
    parser.add_argument('--publish',action='store_true')
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.plan_sha,args.extractions_sha,args.publish)['passed'] else 1)
