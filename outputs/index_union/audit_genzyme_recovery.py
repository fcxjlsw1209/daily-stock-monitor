"""Audit a retired issuer separately from a modern reused ticker.

This writes a candidate and evidence audit only. It never edits a manifest,
membership panel, strategy, plan or ledger, and cannot grant admission.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
from normalize_new_prices import validate_frame, session_mask
from research_calendar import sessions

P = Path(__file__).resolve().parent


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reviewed_frame(path):
    return pd.read_csv(path).rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Adj Close': 'adj_close', 'Volume': 'volume'})


def compare_quarters(frame, facts, tolerance=0.005):
    dates = pd.to_datetime(frame.date, errors='raise')
    result = []
    for row in facts:
        period = pd.Period(row['quarter'], freq='Q')
        part = frame.loc[dates.dt.to_period('Q').eq(period)]
        high, low = float(part.high.max()), float(part.low.min())
        match = bool(len(part) and abs(high-row['high']) <= tolerance
                     and abs(low-row['low']) <= tolerance)
        result.append(dict(row, observed_high=high, observed_low=low,
                           observations=len(part), matched=match))
    return result


def audit_window(frame, start='2008-01-02', end='2011-04-08'):
    d = frame.loc[frame.date.between(start, end)].sort_values('date').copy()
    valid = validate_frame(d) & session_mask(d.date)
    expected = set(sessions(start, end).strftime('%Y-%m-%d'))
    observed = set(d.loc[valid, 'date'])
    unit_error = float((d.adj_close/d.close-1).abs().max())
    return d, dict(first=d.date.min(), last=d.date.max(), rows=len(d),
        invalid_rows=int((~valid).sum()), missing_sessions=sorted(expected-observed),
        maximum_adjusted_to_raw_deviation=unit_error,
        price_window_checks_pass=bool(valid.all() and expected == observed and unit_error <= 1e-8))


def main():
    source = P.parent/'dynamic_top100/additional_sources/free_discovery/qstk/GENZ.csv'
    provenance = json.loads(source.with_suffix('.csv.provenance.json').read_text())
    if checksum(source) != provenance['sha256']:
        raise ValueError('Changed historical source')
    proof = P/'sources/oct08_validation/genzyme_2010_10k_reviewed_facts.json'
    facts = json.loads(proof.read_text())
    frame = reviewed_frame(source)
    d, coverage = audit_window(frame)
    quarters = compare_quarters(frame, facts['quarters'])
    yahoo = P/'sources/yahoo_prices/GENZ.csv'
    wrong = reviewed_frame(yahoo)
    wrong_quarters = compare_quarters(wrong, facts['quarters'])
    tiingo_meta = P.parent/'dynamic_top100/additional_sources/tiingo/GENZ_metadata.json'
    tiingo = json.loads(tiingo_meta.read_text())
    if 'VANECK' not in tiingo['name'].upper() or 'ETF' not in tiingo['name'].upper():
        raise ValueError('ETF identity evidence changed')
    etf = P/'sources/oct08_validation/genz_etf_lineage.html'
    etf_prov = json.loads(etf.with_suffix('.provenance.json').read_text())
    if checksum(etf) != etf_prov['sha256']:
        raise ValueError('Changed ETF source')
    prior_exclusion = json.loads((P/'AVAILABLE_RESEARCH_READINESS.json').read_text())
    def excluded(value):
        if isinstance(value, dict):
            if value.get('symbol') == 'GENZ' and value.get('reason') == 'unresolved_identity':
                return True
            return any(excluded(v) for v in value.values())
        return isinstance(value, list) and any(excluded(v) for v in value)
    if not excluded(prior_exclusion):
        raise ValueError('Cannot confirm completed research excluded GENZ')
    dest = P/'recovery_candidates'; dest.mkdir(exist_ok=True)
    candidate = dest/'GENZ_qstk_2008_2011.csv.gz'
    d['issuer_id'] = 'SEC_CIK_0000732485'
    d['security_scope'] = 'historical_Genzyme_common_not_VanEck_ETF'
    d['entry_admitted'] = False
    d.to_csv(candidate, index=False, compression={'method':'gzip','mtime':0})
    cvr_source = P/'sources/oct08_validation/GCVRZ_tiingo_prices.json'
    cvr_proof = json.loads(cvr_source.with_suffix('.provenance.json').read_text())
    if checksum(cvr_source) != cvr_proof['sha256']:
        raise ValueError('Changed CVR quote cache')
    cvr = pd.DataFrame(json.loads(cvr_source.read_text())).rename(columns={'adjClose':'adj_close'})
    cvr['date'] = pd.to_datetime(cvr.date, utc=True).dt.strftime('%Y-%m-%d')
    # SEC identifies April4 as the first regular trading date; earlier vendor
    # observations are retained in the raw cache, not silently called regular.
    cvr = cvr.loc[cvr.date.ge('2011-04-04')].copy()
    if not (validate_frame(cvr) & session_mask(cvr.date)).all():
        raise ValueError('Invalid CVR bars')
    if not cvr.adj_close.eq(cvr.close).all():
        raise ValueError('CVR adjustment reconciliation needed')
    quote = cvr.loc[cvr.date.eq('2011-04-11')]
    if len(quote) != 1 or quote.volume.iloc[0] <= 0:
        raise ValueError('Missing CVR execution-day quote')
    cvr_dest = dest/'GCVRZ_tiingo_regular_2011.csv.gz'
    cvr.to_csv(cvr_dest,index=False,compression={'method':'gzip','mtime':0})
    completion = P/'sources/oct08_validation/genzyme_completion_reviewed_facts.json'
    event = dict(event_id='GENZ_20110408_cash_and_CVR_research_candidate',
        security_id='GENZ', effective_session='2011-04-11',
        legal_completion_date='2011-04-08', execution_session='2011-04-11',
        known_at='2011-04-08T23:59:59-04:00', terms_verified=True,
        source_file=str(completion.relative_to(P)), source_sha256=checksum(completion),
        price_basis='raw_with_explicit_actions', policy='liquidate_first_tradable_open',
        cash_per_share=74., payment_session=None,
        components=[dict(security_id='GCVRZ',ratio=1.)],
        research_assumption='Liquidate CVR at next exchange-session opening proxy after common stock ceases trading. Actual entitlement delivery is not observed.',
        live_or_backtest_admitted=False)
    (P/'GENZYME_TERMINAL_CANDIDATE.json').write_text(json.dumps(event,indent=2)+'\n')
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
        historical_issuer='Genzyme Corporation', historical_cik='0000732485',
        historical_symbol='GENZ', source_file=str(source.relative_to(P.parent)),
        source_sha256=checksum(source), source_provenance=provenance,
        quarterly_primary_check=quarters, coverage=coverage,
        source_corroborated=bool(all(r['matched'] for r in quarters) and coverage['price_window_checks_pass']),
        replacement_candidate=str(candidate.relative_to(P)), replacement_sha256=checksum(candidate),
        rejected_current_ticker_sources={
            'yahoo_file':str(yahoo.relative_to(P)), 'yahoo_sha256':checksum(yahoo),
            'yahoo_quarters_matching_Genzyme':sum(r['matched'] for r in wrong_quarters),
            'tiingo_metadata_file':str(tiingo_meta.relative_to(P.parent)),
            'tiingo_metadata_sha256':checksum(tiingo_meta), 'tiingo_name':tiingo['name'],
            'reason':'Current GENZ quotes belong to a VanEck fund relaunched from BJK, not old Genzyme common stock.'},
        evidence=[dict(source_file=str(f.relative_to(P)), source_sha256=checksum(f)) for f in
                  [proof, P/'sources/oct08_validation/genzyme_completion_reviewed_facts.json', etf]],
        cvr_quote=dict(source_file=str(cvr_source.relative_to(P)), source_sha256=checksum(cvr_source),
            candidate_file=str(cvr_dest.relative_to(P)), candidate_sha256=checksum(cvr_dest),
            regular_bars=len(cvr), execution_session='2011-04-11',
            opening_price=float(quote.open.iloc[0]), volume=float(quote.volume.iloc[0]),
            actual_entitlement_delivery_observed=False, price_independently_cross_checked=False),
        unresolved=['Observed cash and CVR entitlement settlement/delivery timing',
                    'Independent CVR quote corroboration',
                    'Candidate migration into a newly versioned research bundle and six-strategy rerun'],
        original_research_already_excluded_Genzyme=True,
        existing_results_modified=False, existing_accounts_modified=False,
        strict_complete_history_certified=False, entry_admitted=False,
        note='Quarterly extremes corroborate identity and units, not every daily quote. Post-publication SEC evidence is validation only, never a signal. The raw SEC request returned403; its fact receipts explicitly record web-parser provenance.')
    (P/'GENZYME_RECOVERY_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['source_corroborated','coverage','entry_admitted','unresolved']},indent=2))


if __name__ == '__main__':
    main()
