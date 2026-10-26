"""Offline integration and readiness audit. No model fitting or account writes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from research_calendar import market_calendar, provenance as calendar_provenance
import pandas as pd

from event_schema import normalize_event, event_available, reaction_features
from cached_prices import load_supplemental

P = Path(__file__).resolve().parent
O = P.parent
SOURCES = ['apple', 'nvidia', 'microsoft', 'amazon', 'amazon/supplement', 'tsmc',
           'alphabet', 'meta', 'jpmorgan', 'berkshire', 'walmart', 'visa', 'jnj', 'pg',
           'warmup/verified', 'warmup/cvx', 'warmup/bud', 'warmup/vz', 'warmup/pep_supplement',
           'warmup/dis_supplement', 'warmup/predecessors', 'warmup/financial_supplement',
           'integrated_batches/financial_consumer_687', 'integrated_batches/financial_candidate_34',
           'integrated_batches/financial_candidate_66', 'integrated_batches/xom_supplement_21',
           'integrated_batches/technology_industrial_759', 'integrated_batches/cost_supplement_19',
           'integrated_batches/health_energy_foreign_900', 'warmup/health_early',
           'integrated_batches/training_health_35', 'integrated_batches/financial_candidate_5',
           'integrated_batches/pfe_supplement_20', 'integrated_batches/tm_predecessor_1',
           'integrated_batches/ge_nvo_supplement_48', 'integrated_batches/sap_predecessor_1',
           'integrated_batches/abbv_supplement_1', 'integrated_batches/nvs_supplement_2']
CORRECTIONS = ['integrated_batches/cost_supplement_19/CORRECTIONS.json']
# Explicit legacy issuer ID, not today's recycled FB security.
PRICE_SYMBOLS = {'research_equity_FB': 'META'}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def dump(name, obj):
    (P / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def resolve_evidence(value, folder):
    p = Path(value)
    if not p.is_absolute():
        p = p if p.exists() else folder / p
    if not p.is_file():
        raise ValueError('Missing evidence: ' + str(p))
    return p


def adapt(row, folder, calendar, provider_by_date):
    file = resolve_evidence(row['evidence_path'], folder)
    expected = row.get('evidence_sha256') or row.get('sha256')
    if not expected or sha(file) != expected:
        raise ValueError('Evidence checksum mismatch: ' + str(file))
    additional_evidence = row.get('additional_evidence', [])
    for evidence in additional_evidence:
        extra = resolve_evidence(evidence['evidence_path'], folder)
        if sha(extra) != evidence['sha256']:
            raise ValueError('Additional publication evidence checksum mismatch')
    timestamp_evidence = row.get('timestamp_evidence')
    if row.get('publication_timestamp'):
        if not timestamp_evidence:
            raise ValueError('Exact publication needs independently referenced timestamp evidence')
        tf = resolve_evidence(timestamp_evidence['evidence_path'], folder)
        if sha(tf) != timestamp_evidence['sha256']:
            raise ValueError('Publication timestamp evidence checksum mismatch')
    symbol = row['ticker']; vendor = provider_by_date.get((symbol, row['release_date']), {})
    # Match only verified release date+issuer, never calendar-quarter labels.
    event = normalize_event(dict(symbol=symbol, asset_type='equity', vendor=vendor,
        official=dict(actual_period_end=row.get('fiscal_period_end_actual'),
                      publication_date=row['release_date'],
                      publication_at=row.get('publication_timestamp'),
                      publication_timezone=row.get('publication_timezone'),
                      publication_kind='release', source_url=row['source_url'], verified=True)), calendar)
    event.update(fiscal_year=row.get('fiscal_year'), fiscal_quarter=row.get('fiscal_quarter'),
        evidence_path=str(file), evidence_sha256=expected, source_precision=row['precision'],
        source_kind=row.get('source_kind') or row.get('source_class') or row.get('evidence_kind'),
        publication_timezone_basis=row.get('publication_timezone_basis') or row.get('publication_timezone_evidence'),
        event_id=symbol + '_' + row['release_date'],
        provider_join='issuer_and_verified_release_date' if vendor else None,
        additional_evidence=additional_evidence,
        historical_archive_limitation='Current issuer archive evidence; not a contemporaneously captured immutable news feed.')
    if timestamp_evidence:
        event['timestamp_evidence'] = timestamp_evidence
    for key in ('issuer_name_at_release', 'historical_ticker', 'issuer_identity',
                'issuer_identity_note', 'issuer_identity_evidence', 'identity_source_url',
                'evidence_limitation', 'publication_version', 'period_basis', 'first_public_time_verified',
                'training_predecessor_only', 'visual_first_page_evidence', 'original_response_available',
                'limitation', 'publication_evidence', 'period_evidence', 'source_correction'):
        if key in row:
            event[key] = row[key]
    return event


def apply_corrections(content, source_file, manifests):
    """Explicit, hash-bound overlay; immutable acquisition sources stay intact."""
    rows=[dict(r) for r in content]
    for manifest_file,manifest in manifests:
        original=Path(manifest['original_input'])
        if original.resolve()!=source_file.resolve():continue
        if sha(source_file)!=manifest['original_input_sha256']:
            raise ValueError('Correction original source checksum mismatch')
        for correction in manifest['corrections']:
            matches=[i for i,r in enumerate(rows) if all(r.get(k)==correction[k] for k in ('ticker','release_date','fiscal_year','fiscal_quarter'))]
            if len(matches)!=1:raise ValueError('Correction must target exactly one event')
            i=matches[0];old=rows[i]
            if old['fiscal_period_end_actual']!=correction['old_end']:raise ValueError('Correction prior value mismatch')
            evidence=correction['independent_evidence'];file=resolve_evidence(evidence['evidence_path'],manifest_file.parent)
            if sha(file)!=evidence['sha256']:raise ValueError('Correction evidence checksum mismatch')
            new=dict(correction['corrected_event'])
            if new['fiscal_period_end_actual']!=correction['verified_end']:raise ValueError('Correction new value mismatch')
            if any(new[k]!=old[k] for k in ('ticker','release_date','fiscal_year','fiscal_quarter')):raise ValueError('Correction identity changed')
            new['source_correction']=dict(manifest=str(manifest_file),manifest_sha256=sha(manifest_file),
                original_input=str(source_file),original_input_sha256=sha(source_file),
                old_fiscal_period_end=correction['old_end'],new_fiscal_period_end=correction['verified_end'],reason=correction['reason'])
            rows[i]=new
    return rows


def quarter_gaps(events):
    """Internal fiscal-quarter sequence audit, not proof no extra release existed."""
    ids = [int(r['fiscal_year']) * 4 + int(r['fiscal_quarter']) - 1 for r in events
           if r.get('fiscal_year') is not None and r.get('fiscal_quarter') is not None]
    missing = sorted(set(range(min(ids), max(ids) + 1)) - set(ids)) if ids else []
    return dict(identified_quarters=len(ids), duplicates=len(ids) - len(set(ids)),
                missing_between_observed_endpoints=['%dQ%d' % (x // 4, x % 4 + 1) for x in missing])


def monthly_reaction_panel(availability, events, prices, calendar, horizons=(3, 10, 21)):
    """Latest known event only; unavailable windows never become zero or old events."""
    lookup = {event['event_id']: event for event in events}
    panel = []
    for membership in availability:
        row = dict(membership, reactions={})
        asof = calendar.session_close(pd.Timestamp(row['signal_date']))
        row['signal_asof'] = asof.isoformat()
        if row['status'] != 'verified_event_available':
            panel.append(row)
            continue
        event = lookup[row['event_id']]
        if not event_available(event, asof) or event['symbol'] != row['symbol']:
            raise ValueError('Monthly event identity/availability mismatch')
        if row['symbol'] not in prices:
            row['status'] = 'missing_cached_price_file'
            panel.append(row)
            continue
        for horizon in horizons:
            result = reaction_features(event, prices[row['symbol']], asof,
                horizon=horizon, calendar=calendar, benchmark_bars=prices.get('SPY'))
            if result['status'] == 'ready' and pd.Timestamp(result['end_close_at']) > asof:
                raise ValueError('Future reaction reached monthly feature panel')
            row['reactions'][str(horizon)] = result
        panel.append(row)
    return panel


def asof_availability(meta, groups, calendar, etfs):
    """Same event-availability policy for evaluation and training inputs."""
    rows=[]
    for day,symbols in zip(meta['dates'],meta['symbols']):
        asof=calendar.session_close(pd.Timestamp(day))
        for symbol in symbols:
            if symbol in etfs:
                rows.append(dict(signal_date=day,symbol=symbol,status='not_applicable_etf',event_id=None));continue
            known=[r for r in groups.get(symbol,[]) if event_available(r,asof)]
            latest=max(known,key=lambda r:r['known_by_at']) if known else None
            age=(asof-pd.Timestamp(latest['known_by_at'])).days if latest else None
            rows.append(dict(signal_date=day,symbol=symbol,
                status=('verified_event_available' if age<=185 else 'stale_event') if latest else 'missing_event_history',
                event_id=latest['event_id'] if latest else None,known_age_calendar_days=age,
                history_coverage_complete=False,explanation='Event availability is not completeness proof; never fill unknown histories with zero.'))
    return rows


def main():
    calendar = market_calendar(start='2010-01-01', end='2030-12-31')
    provider_file = O / 'monthly_event_confirmation/raw/AAPL.json'
    vendor = json.loads(provider_file.read_text())
    by_date = {('AAPL', r['reportedDate']): r for r in vendor['quarterlyEarnings']}
    if len(by_date) != len(vendor['quarterlyEarnings']):
        raise ValueError('Duplicate provider release dates require explicit reconciliation')
    rows, raw_records, inputs = [], [], {str(provider_file): sha(provider_file)}
    calendar_meta=calendar_provenance()
    inputs[calendar_meta['manifest']]=calendar_meta['manifest_sha256']
    for evidence in calendar_meta['closures']:
        inputs[evidence['source_file']]=evidence['source_sha256']
    dump('CALENDAR_PROVENANCE.json',calendar_meta)
    corrections=[]
    for name in CORRECTIONS:
        file=P/name
        if file.exists():
            inputs[str(file)]=sha(file);corrections.append((file,json.loads(file.read_text())))
    for name in SOURCES:
        file = P / name / 'events.json'
        if not file.exists(): continue
        inputs[str(file)] = sha(file)
        identity_file = file.parent / 'IDENTITY.json'
        if identity_file.exists():
            identity = json.loads(identity_file.read_text())
            identity_evidence = resolve_evidence(identity['evidence_path'], file.parent)
            if sha(identity_evidence) != identity['evidence_sha256']:
                raise ValueError('Issuer identity evidence checksum mismatch')
            research_key = identity['research_ticker']
            if PRICE_SYMBOLS.get(research_key) not in identity['historical_tickers']:
                raise ValueError('Price alias not supported by issuer identity evidence')
            inputs[str(identity_file)] = sha(identity_file)
            inputs[str(identity_evidence)] = sha(identity_evidence)
        content = json.loads(file.read_text())
        data = content['events'] if isinstance(content, dict) else content
        data = apply_corrections(data,file,corrections)
        for row in data:
            raw_records.append(row)
            rows.append(adapt(row, file.parent, calendar, by_date))
    keys = [r['event_id'] for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Conflicting duplicate issuer/release dates')
    rows.sort(key=lambda r: (r['symbol'], r['publication_date'] or ''))
    dump('EVENTS.json', rows)
    official_aapl = {r['publication_date']: r for r in rows if r['symbol'] == 'AAPL'}
    reconcile = []
    for (_, day), hint in sorted(by_date.items()):
        if not '2015-01-01' <= day <= '2026-09-24': continue
        actual = official_aapl.get(day)
        reconcile.append(dict(provider_reported_date=day, provider_period_label=hint['fiscalDateEnding'],
            official_date_matched=actual is not None,
            official_period_end=actual['official_actual_period_end'] if actual else None,
            fiscal_labels_differ=actual['fiscal_labels_differ'] if actual else None,
            match_key='AAPL plus verified release date; period identity checked in official fiscal-year/quarter text'))
    dump('PROVIDER_RECONCILIATION.json', reconcile)
    groups = defaultdict(list)
    for row in rows:
        groups[row['symbol']].append(row)
    meta_file = O / 'monthly_architecture_restart/META.json'
    inputs[str(meta_file)] = sha(meta_file)
    all_meta = json.loads(meta_file.read_text())
    meta = all_meta['monthly']
    training_meta = all_meta.get('training',dict(dates=[],symbols=[]))
    equities = sorted(set(sum(meta['symbols'], [])) - set(meta['etfs']))
    training_equities = sorted(set(sum(training_meta['symbols'], [])) - set(meta['etfs']))
    coverage = []
    for symbol in sorted(set(equities)|set(training_equities)):
        g = groups[symbol]
        membership = [d for d, ss in zip(meta['dates'], meta['symbols']) if symbol in ss]
        training_membership = [d for d,ss in zip(training_meta['dates'],training_meta['symbols']) if symbol in ss]
        ready = [r for r in g if r['status'] == 'ready']
        dates = [r['publication_date'] for r in ready]
        sequence = quarter_gaps(ready)
        coverage.append(dict(symbol=symbol, member_months=len(membership),
            first_member_signal=min(membership,default=None), last_member_signal=max(membership,default=None),
            training_member_months=len(training_membership),
            first_training_signal=min(training_membership,default=None),last_training_signal=max(training_membership,default=None),
            scope='monthly_and_training' if membership and training_membership else ('monthly_only' if membership else 'training_only'),
            collected_records=len(g), verified_events=len(ready), pending_review=len(g)-len(ready),
            earliest_release=min(dates, default=None),
            latest_release=max(dates, default=None),
            quarter_sequence=sequence,
            status=('verified_history_collected' if len(ready) >= 40 and not sequence['duplicates'] and not sequence['missing_between_observed_endpoints'] else
                    ('partial_verified_history' if ready else ('pending_review' if g else 'not_collected'))),
            full_universe_factor_ready=False))
    # Reusable missing mask for every actual monthly candidate, not a new pool.
    monthly = asof_availability(meta,groups,calendar,set(meta['etfs']))
    training = asof_availability(training_meta,groups,calendar,set(meta['etfs']))
    dump('MONTHLY_AVAILABILITY.json', monthly)
    dump('TRAINING_AVAILABILITY.json',training)
    # Smoke exercise with real cached adjusted prices; not a strategy backtest.
    prices, price_inputs = {}, {}
    for symbol in sorted(set(groups) | {'SPY'}):
        price_symbol = PRICE_SYMBOLS.get(symbol, symbol)
        f = O / 'index_union/normalized_prices' / (price_symbol + '_yahoo.csv.gz')
        if not f.exists():
            supplemental, provenance = load_supplemental(symbol, O)
            if supplemental is not None:
                prices[symbol] = supplemental
                price_inputs.update(provenance)
            continue
        df = pd.read_csv(f, index_col='date', parse_dates=['date'])
        if 'symbol' not in df or set(df['symbol'].dropna().unique()) != {price_symbol}:
            raise ValueError('Cached price issuer identity mismatch: ' + symbol)
        # Respect cached bar validity; do not forward-fill holes or price units.
        df = df.loc[df['bar_valid'] == True].copy()
        prices[symbol] = pd.DataFrame({'close': df['adj_close'], 'volume': df['volume']})
        price_inputs[str(f)] = sha(f)
    features = []
    endpoint = calendar.session_close(pd.Timestamp('2026-09-24'))
    for event in rows:
        if event['symbol'] not in prices:
            features.append(dict(event_id=event['event_id'], status='missing_cached_price_file',
                post_known_close_to_close_return=None, market_adjusted_return=None))
            continue
        result = reaction_features(event, prices[event['symbol']], endpoint, horizon=3,
                                   calendar=calendar, benchmark_bars=prices.get('SPY'))
        features.append(dict(event_id=event['event_id'], **result))
    dump('REACTION_PREVIEW.json', dict(status='technical_data_preview_not_backtest',
        semantics='Three completed XNYS sessions after conservative known-by anchor close; excludes original announcement gap.',
        price_limit='Uses existing cached adjusted close and split-adjusted volume with inherited vendor/identity limitations; not a fresh price validation.',
        price_inputs=price_inputs, features=features))
    monthly_features = monthly_reaction_panel(monthly, rows, prices, calendar)
    training_features = monthly_reaction_panel(training, rows, prices, calendar)
    dump('TRAINING_FEATURES.json',dict(status='asof_training_data_preview_not_model_fit',
        semantics='Same post-known 3/10/21 completed-session drift rules as the monthly panel.',
        missing_policy='Missing/immature/N/A remain null. No eligible-pool change.',
        price_inputs=price_inputs,rows=training_features))
    dump('MONTHLY_FEATURES.json', dict(status='asof_monthly_candidate_feature_inputs_not_backtest',
        semantics='Latest verified known event at each actual monthly signal; three/ten/twenty-one completed XNYS sessions after the conservative known-by close. No initial earnings gap.',
        missing_policy='Unknown, stale and ETF events are not scored. Immature or missing-price reactions remain null; never substitute an older completed event for the latest event.',
        coverage_limit='Availability alone does not prove completeness. Candidate-scoped readiness is separately audited in SCOPED_READINESS.json; no claim of every preliminary release or historical surprise.',
        price_inputs=price_inputs, rows=monthly_features))
    protected_file = P/'PROTECTED_CONTINUATION.json'
    if not protected_file.exists(): protected_file = P/'PROTECTED.json'
    protected = json.loads(protected_file.read_text())
    changed = [f for f, h in protected.items() if not Path(f).is_file() or sha(Path(f)) != h]
    if changed:
        raise ValueError('Protected inputs changed: ' + repr(changed[:5]))
    queue = json.loads((P / 'SEC_REVIEW_QUEUE.json').read_text())
    stops = dict(json.loads((P/'PLAN.json').read_text())['source_stops'])
    for name in ['nvidia', 'amazon']:
        validation = P/name/'VALIDATION.json'
        if validation.exists():
            stops.update(json.loads(validation.read_text()).get('blocked_sources', {}))
    stop_manifests = {}
    for file in [P/name/'SOURCE_STOPS.json' for name in SOURCES] + [P/'ROUND3_SOURCE_STOPS.json', P/'warmup/SOURCE_STOPS.json',
            P/'warmup/BUDVZ_SOURCE_STOPS.json', P/'batch_financial_consumer/SOURCE_STOPS.json',
            P/'batch_health_energy_foreign/SOURCE_STOPS.json', P/'batch_technology_industrial/SOURCE_STOPS.json']:
        if file.exists():
            stop_manifests[str(file)] = json.loads(file.read_text())
            inputs[str(file)] = sha(file)
    status = dict(generated_at=datetime.now(timezone.utc).isoformat(),
        status='regular_release_data_built_pending_independent_scoped_audit',
        full_universe_event_factor_ready=False, factor_backtested=False,
        scoped_readiness_artifact='SCOPED_READINESS.json',
        legacy_full_universe_flag_meaning='Not a globally exhaustive historical announcements/first-public-clock/EPS-surprise archive. Actual frozen candidate post-known drift readiness is reported separately.',
        universe_equities=len(equities), etfs_not_applicable=len(meta['etfs']),
        training_equities=len(training_equities),all_scope_equities=len(set(equities)|set(training_equities)),
        training_only_equities=sorted(set(training_equities)-set(equities)),
        collected_event_records=len(rows), verified_events=sum(r['status'] == 'ready' for r in rows),
        ready_event_rows=sum(r['status'] == 'ready' for r in rows),
        events_by_symbol=dict(Counter(r['symbol'] for r in rows)),
        precision_counts=dict(Counter(r['precision'] for r in rows)),
        fiscal_label_differences=sum(r['fiscal_labels_differ'] is True for r in rows),
        matched_aapl_provider_dates=sum(r['provider_join'] is not None for r in rows),
        preview_status_counts=dict(Counter(f['status'] for f in features)),
        monthly_availability_counts=dict(Counter(r['status'] for r in monthly)),
        training_availability_counts=dict(Counter(r['status'] for r in training)),
        training_feature_status_counts={str(h):dict(Counter(r['reactions'][str(h)]['status'] for r in training_features if str(h) in r['reactions'])) for h in (3,10,21)},
        monthly_signal_dates=len(meta['dates']),
        monthly_feature_status_counts={str(h): dict(Counter(r['reactions'][str(h)]['status']
            for r in monthly_features if str(h) in r['reactions'])) for h in (3, 10, 21)},
        sec_offline_review_candidates=queue['candidate_filings'],
        sec_issuers_with_candidates=queue['issuers_with_candidates'],
        source_stops=stops,
        source_stop_manifests=stop_manifests,
        remaining=['Run independent candidate-scoped audit after every rebuild; SCOPED_READINESS.json lists actual unresolved gaps and source-verified structural exceptions.',
                   'Noncandidate archive gaps in COVERAGE.json are not implicitly filled. Preliminary/amendment releases and first-public clocks are outside the regular-release drift data scope.',
                   'Date-only policy supports later post-known drift, not exact announcement-gap or historical EPS-surprise replication.',
                   'New model/event-factor comparisons require a separately fixed research plan; none was run by this data build.'],
        protected_files_checked=len(protected), protected_files_changed=0,
        protected_manifest=str(protected_file),
        price_issuer_mapping=PRICE_SYMBOLS,
        calendar=calendar_meta,
        inputs=inputs)
    dump('COVERAGE.json', coverage); dump('STATUS.json', status)
    print(json.dumps({k: status[k] for k in ['verified_events', 'ready_event_rows', 'events_by_symbol', 'fiscal_label_differences', 'preview_status_counts', 'protected_files_changed']}))


if __name__ == '__main__':
    main()
