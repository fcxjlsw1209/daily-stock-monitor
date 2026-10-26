"""Inspect frozen source routes only; never infer historical security continuity."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from after156_event_scope_local_review import Sources, require

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/after156_price_identity_routes.json'
META = 'outputs/monthly_local_linear70/META.json'
META_SHA = '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4'
UNION = 'outputs/index_union/'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    src = Sources()
    meta = src.read(META, META_SHA)
    stocks = sorted(set(s for month in meta['symbols'] for s in month) - set(meta['etfs']))
    require(len(stocks) == 73, 'Original stock scope changed')
    source_hashes = src.read('outputs/monthly_2016_extension/SOURCE_HASHES.json')
    alias = src.read(UNION + 'alias_resolved_inputs/MANIFEST.json')
    raw_manifest = next(r for r in alias['sources'] if r['file'] == 'input_diagnostics/MANIFEST.json')
    inputs = src.read(UNION + raw_manifest['file'], raw_manifest['sha256'])
    # Bind the intermediate adjusted panel used by the original loader, not its values.
    adj = next(r for r in alias['fields'] if r['field'] == 'adj')
    adj_path = UNION + adj['file']
    require(source_hashes[str(ROOT / adj_path)] == adj['sha256'], 'Frozen intermediate route mismatch')
    src.read_bytes(adj_path, adj['sha256'])
    rows = []
    for symbol in stocks:
        proofs = [r for r in alias['identity_proofs'] if r['symbol'] == symbol]
        routes = [r for r in inputs['price_sources'] if r['symbol'] == symbol]
        proof_rows = []
        for proof in proofs:
            name = 'outputs/' + proof['source_file']
            value = src.read(name, proof['source_sha256'])
            proof_rows.append(dict(source=name, sha256=proof['source_sha256'],
                basis=proof['basis'], cik=str(value.get('cik', '')).zfill(10),
                current_tickers=value.get('tickers'), current_exchanges=value.get('exchanges')))
        for route in routes:
            for file_key, hash_key in (('input_file', 'input_sha256'), ('raw_file', 'raw_sha256')):
                if file_key in route:
                    src.read_bytes(UNION + route[file_key], route[hash_key])
        rows.append(dict(symbol=symbol, intermediate_issuer=alias['issuers'].get(symbol),
            exact_internal_symbol_source_routes=routes, current_identity_references=proof_rows,
            exact_internal_symbol_route_present=bool(routes),
            historical_price_security_binding_admitted=False,
            route_limit='Exact internal-symbol lookup only. Alias decisions and later overlays are not silently resolved; this is not a numeric reconstruction of final PRICES.npz.'))
    for name in ('work/after156_price_identity_routes.py',
                 'work/after156_event_scope_local_review.py',
                 'outputs/monthly_2016_extension/extend.py',
                 'outputs/monthly_guidance_reset/study.py',
                 'outputs/monthly_factor_search_v2/data_inputs.py',
                 'outputs/index_union/assemble_alias_resolved_inputs.py',
                 'run_local.sh', 'local_run.py'):
        src.read_bytes(name)
    src.recheck()
    result = dict(status='FROZEN_METADATA_SOURCE_ROUTES_REVIEWED_NO_SECURITY_ADMISSION',
        created_utc=datetime.now(timezone.utc).isoformat(),
        scope='All original 73 stock columns; existing intermediate metadata and source bytes only.',
        summary=dict(stocks=len(rows), direct_internal_symbol_routes=sum(bool(r['exact_internal_symbol_source_routes']) for r in rows),
            columns_with_current_identity_references=sum(bool(r['current_identity_references']) for r in rows),
            new_historical_security_bindings_admitted=0),
        findings=['The original intermediate manifest explicitly describes itself as alias_resolution_diagnostic_not_admitted.',
            'Current SEC corroboration and an unchanged internal price column are not independent historical security-series proof.',
            'Use existing source routes and dated issuer/class evidence to build event-specific joins; do not manufacture daily security epochs or assume later overlays share identity.',
            'This metadata audit neither invalidates nor recertifies completed research. Existing results retain their declared historical-universe and identity assumptions.'],
        next_required='Bind each consumed periodic filing to issuer/class and its actual vendor price series; inspect explicit ticker/class/ADR/successor transitions within the beta and formation interval.',
        rows=rows, input_hashes=src.hashes,
        activity=dict(numerical_price_values_interpreted=0, new_factors=0, new_orders=0,
                      new_returns=0, new_trials=0, network_requests=0, observer_changes=0))
    if args.write:
        with src.path(OUT).open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps(dict(path=OUT if args.write else None, summary=result['summary'],
                         files_verified=len(src.hashes))))


if __name__ == '__main__':
    main()
