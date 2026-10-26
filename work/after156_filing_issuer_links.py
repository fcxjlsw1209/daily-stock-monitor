"""Link canonical filing metadata to bounded dated identity observations only."""
import argparse
from collections import Counter, defaultdict
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from after156_event_scope_local_review import Sources, require
from after156_filing_universe_inventory import rows as sec_rows
from after156_identity_binding_batch import document, normalize
from after156_event_price_identity import original_ciks

SELF = 'work/after156_filing_issuer_links.py'
OUT = 'work/after156_filing_issuer_links.json'
CANONICAL = 'work/after156_canonical_chronology.json'
IDENTITY = 'work/after156_event_price_identity.json'
ROUTES = 'work/after156_price_identity_routes.json'
BINDING = 'work/after156_identity_binding_batch.json'
INVENTORY = 'work/after156_filing_universe_inventory.json'
PINS = {
    CANONICAL: '308bc7268ef36f1b8f4a4db3c76eeabddcb2c85cf65c439edccce6006379b796',
    IDENTITY: 'a1e4a42a72b79dc43c20836802e3f5df6265d1ab09c66699e59180fd7ce525fd',
    ROUTES: '4184ba45c51062c79ec41f8caabaad413749ae6dd0d68ae0738fabca2fa33ec7',
    BINDING: 'a381874f329871a0e602606e3325fd923274feb35b7d8f8c663ec0a8941c921f',
    INVENTORY: 'e5157e2c6da08253baa9e523be2465261b442b506fad3baf7216664f47ff7286',
}
ERAS = [('early53',0,53),('recent35',53,88),('first88',0,88),('later34',88,122),('all122',0,122)]


def relation(observation_date, event_date):
    return 'BEFORE_EVENT_DATE' if observation_date < event_date else 'AFTER_EVENT_DATE' if observation_date > event_date else 'SAME_REPORTED_DATE'


def exact_filing_document(url, event):
    # Only an original SEC archive route binds an accession. Never guess from a filename.
    needle = '/Archives/edgar/data/%d/%s/%s' % (int(event['cik']),
        event['accessionNumber'].replace('-', ''), event['primaryDocument'])
    return isinstance(url, str) and re.match(r'^https://(?:www\.)?sec\.gov/', url) is not None and url.split('?',1)[0].endswith(needle)


def tests():
    require(relation('2020-01-01','2020-01-02') == 'BEFORE_EVENT_DATE','Before')
    require(relation('2020-01-02','2020-01-01') == 'AFTER_EVENT_DATE','After')
    require(relation('2020-01-01','2020-01-01') == 'SAME_REPORTED_DATE','Same day')
    event=dict(cik='0000000001',accessionNumber='0000000001-20-000001',primaryDocument='x.htm')
    url='https://www.sec.gov/Archives/edgar/data/1/000000000120000001/x.htm'
    require(exact_filing_document(url,event),'Exact original accession URL')
    require(not exact_filing_document(url.replace('/1/','/2/'),event),'Different CIK rejected')
    require(not exact_filing_document(url.replace('sec.gov','example.com'),event),'Unrelated host rejected')
    require(not exact_filing_document(url.replace('x.htm','other.htm'),event),'Different primary document rejected')
    return 7


class Verified:
    def __init__(self, src):
        self.src, self.blobs, self.objects = src, {}, {}

    def blob(self, path, sha):
        if path not in self.blobs:
            self.blobs[path] = self.src.read_bytes(path, sha)
        require(self.src.hashes[path] == sha, 'Conflicting reference: '+path)
        return self.blobs[path]

    def obj(self, path, sha):
        if path not in self.objects:
            self.objects[path] = json.loads(self.blob(path,sha))
        else:
            require(self.src.hashes[path] == sha, 'Conflicting reference: '+path)
        return self.objects[path]

    def sec_proof(self, proof, input_hashes):
        raw = self.obj(proof['path'],proof['sha256'])
        if proof['source_receipt_verified']:
            path = proof['receipt']
            receipt = self.obj(path,input_hashes[path])
            require(receipt['sha256'] == proof['sha256'] and
                    receipt.get('source',receipt.get('url')) == proof['expected_source_url'], 'SEC receipt binding')
        return raw


def observation_details(checked, row, binding):
    original = checked.blob(row['original_path'],row['original_sha256'])
    checked.blob(row['original_receipt_path'],row['original_receipt_sha256'])
    require(binding['source']['original_bytes_receipt_verified'] and
            binding['source']['original_sha256'] == row['original_sha256'] and
            binding['source']['receipt_sha256'] == row['original_receipt_sha256'], 'Bound original receipt chain')
    if original.startswith(b'%PDF'):
        extraction = binding['extraction']
        text = normalize(checked.blob(extraction['path'],extraction['sha256']).decode('utf-8'))
        ciks = []
    else:
        text,_ = document(original)
        ciks = original_ciks(original)
    require(ciks == row['original_CIK_fields'], 'Original CIK fields changed')
    locators = [row[k] for k in ('original_issuer','original_ticker_exchange','original_security_class_literal') if row[k]]
    publication = binding['publication']
    if publication.get('literal'):
        locators.append(publication['literal'])
    if row['bridge']:
        locators += [v for v in row['bridge'].values() if isinstance(v,dict) and 'start' in v]
    for loc in locators:
        require(text[loc['start']:loc['end']] == loc['text'], 'Original text locator mismatch')
    current = row['current_CIK_reference']
    sec = checked.obj(current['path'],current['sha256'])
    receipt = checked.obj(current['receipt_path'],current['receipt_sha256'])
    require(str(sec['cik']).zfill(10) == current['cik'] and
            receipt['sha256'] == current['sha256'] and
            receipt.get('source',receipt.get('url')) == current['source_url'], 'Current SEC reference changed')
    require(publication['date'] == row['original_publication_date'], 'Publication date changed')
    return dict(binding_id=row['binding_id'],symbol=row['research_security_id'],
        observation_date=row['original_publication_date'],source_kind=row['source_kind'],
        original_source=dict(path=row['original_path'],sha256=row['original_sha256'],
            receipt_path=row['original_receipt_path'],receipt_sha256=row['original_receipt_sha256'],
            url=binding['source']['url']),
        original_issuer=row['original_issuer'],original_ticker_exchange=row['original_ticker_exchange'],
        class_bound_to_ticker=row['original_class_bound_to_ticker'],original_class=row['original_security_class_literal'],
        original_labelled_CIKs=[c['cik'] for c in ciks],current_CIK_reference=current,
        source_ticker=row['source_ticker'],provider_symbol=row['selected_provider_symbol'],
        dated_ticker_to_provider_edge=row['join'],explicit_same_security_bridge=row['bridge'],
        supported_scope='Dated observation point; explicit bridge only for its stated transition.',
        continuous_validity_period_inferred=False)


def era_summary(rows):
    return {name:dict(original_stock_slots=sum(lo<=r['signal_index']<hi for r in rows),
        fourteen_series_slots=sum(lo<=r['signal_index']<hi and r['in_fourteen_series_batch'] for r in rows),
        fourteen_series_source_closed_slots=sum(lo<=r['signal_index']<hi and r['in_fourteen_series_batch'] and r['canonical_source_closed'] for r in rows),
        fourteen_series_with_two_original_keys=sum(lo<=r['signal_index']<hi and r['in_fourteen_series_batch'] and r['original_report_keys']>=2 for r in rows),
        fourteen_series_slots_with_any_same_date_observation=sum(lo<=r['signal_index']<hi and r['same_date_observation_event_count']>0 for r in rows),
        admitted_historical_security_slots=0)
        for name,lo,hi in ERAS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    src = Sources()
    inputs = {p:src.read(p, h) for p,h in PINS.items()}
    identity, routes, binding = inputs[IDENTITY], inputs[ROUTES], inputs[BINDING]
    if args.inspect:
        chosen = {}
        for row in identity['bindings']:
            s = row['research_security_id']
            if s in ('MSFT','AMZN','research_equity_FB') and s not in chosen:
                chosen[s] = row
        print(json.dumps(dict(issuer_summary=identity['issuer_summary'], examples=chosen,
            vendor_example=identity['vendor_series']['META'], route_examples=routes['rows'][:2],
            binding_example=next(r for r in binding['bindings'] if r['symbol']=='research_equity_FB'),
            routes_summary=routes['summary']), ensure_ascii=False, indent=2))
        return
    checks=tests()
    canonical,inventory=inputs[CANONICAL],inputs[INVENTORY]
    checked=Verified(src)
    batch={r['research_security_id'] for r in identity['issuer_summary']}
    require(len(batch)==14 and len(identity['bindings'])==110, 'Fixed identity batch differs')
    stock_inventory={r['symbol']:r for r in inventory['stocks']}
    route_by_symbol={r['symbol']:r for r in routes['rows']}
    require(set(stock_inventory)==set(route_by_symbol) and len(stock_inventory)==73, 'Original stock universe differs')
    binding_by_id={r['binding_id']:r for r in binding['bindings']}
    observations={r['binding_id']:observation_details(checked,r,binding_by_id[r['binding_id']]) for r in identity['bindings']}
    by_symbol={s:[o for o in observations.values() if o['symbol']==s] for s in batch}
    # Verify frozen metadata manifests; market CSV/NPZ files are deliberately not reopened.
    alias_path='outputs/index_union/alias_resolved_inputs/MANIFEST.json'
    alias=checked.obj(alias_path,routes['input_hashes'][alias_path])
    input_path='outputs/index_union/input_diagnostics/MANIFEST.json'
    source_inputs=checked.obj(input_path,routes['input_hashes'][input_path])
    sec_proofs={}
    for snapshot in inventory['submission_snapshots']:
        sec_proofs[snapshot['source']['path']]=snapshot['source']
        for p in snapshot['advertised_pages']:
            if 'source' in p: sec_proofs[p['source']['path']]=p['source']
    table_cache={}
    events_by_snapshot={}
    for path,snapshot in canonical['snapshots'].items():
        events_by_snapshot[path]={e['event_id']:e for e in snapshot['events']}
        if not snapshot['events']: continue
        raw=checked.sec_proof(sec_proofs[path],inventory['input_hashes'])
        require(str(raw['cik']).zfill(10)==snapshot['cik'], 'Canonical CIK differs from source')
        for event in snapshot['events']:
            expected=tuple(event[k] for k in ('accessionNumber','form','filingDate','reportDate','raw_acceptanceDateTime','primaryDocument'))
            for occurrence in event['occurrences']:
                p=occurrence['source_path']
                if p not in table_cache:
                    original=checked.sec_proof(sec_proofs[p],inventory['input_hashes'])
                    table_cache[p]=sec_rows(original['filings']['recent'] if 'filings' in original else original)
                require(occurrence['source_sha256']==sec_proofs[p]['sha256'] and
                        table_cache[p][occurrence['row_index']]==expected, 'Canonical occurrence is not exact source row')
    providers={}
    for symbol,vendor in identity['vendor_series'].items():
        rec=checked.obj(vendor['receipt_path'],vendor['receipt_sha256'])
        require(rec['symbol']==symbol and rec['sha256']==vendor['original_sha256'] and
                rec['provider']==vendor['provider'], 'Provider request receipt changed')
        matches=[r for r in source_inputs['price_sources'] if r['symbol']==symbol]
        require(len(matches)==1 and matches[0]['input_sha256']==vendor['normalized_sha256'] and
                matches[0]['raw_sha256']==vendor['execution_sha256'], 'Frozen intermediate route differs')
        providers[symbol]=dict(vendor,market_files_reopened=False,
            prior_opaque_hash_chain_is_bound_by_identity_report=PINS[IDENTITY])
    # Audit all original stock columns' existing proof metadata, including the other 59.
    stock_routes={}
    for symbol,row in route_by_symbol.items():
        proofs=[]
        for proof in [p for p in alias['identity_proofs'] if p['symbol']==symbol]:
            path='outputs/'+proof['source_file']
            raw=checked.obj(path,proof['source_sha256'])
            item=dict(source_path=path,source_sha256=proof['source_sha256'],basis=proof['basis'],
                current_CIK=str(raw['cik']).zfill(10) if 'cik' in raw else None,
                current_tickers=raw.get('tickers'),proof_reusable_as_historical_epoch=False)
            if path in sec_proofs:
                checked.sec_proof(sec_proofs[path],inventory['input_hashes'])
                item['original_SEC_receipt_verified']=sec_proofs[path]['source_receipt_verified']
            proofs.append(item)
        direct=row['exact_internal_symbol_source_routes']
        alias_decisions=[a for a in alias['alias_decisions'] if a['security_id']==symbol]
        provider_symbol=next(iter({o['provider_symbol'] for o in by_symbol.get(symbol,[])}), None)
        route_status='DIRECT_FROZEN_INTERMEDIATE_ROUTE' if direct else 'EXPLICIT_FROZEN_ALIAS_ROUTE' if alias_decisions else 'NO_EXACT_FROZEN_ROUTE_IN_THIS_AUDIT'
        stock_routes[symbol]=dict(research_price_column=symbol,
            column_basis='Original META candidate identity; final extension/overlay security lineage is not newly certified.',
            intermediate_issuer=row['intermediate_issuer'],direct_routes=direct,alias_decisions=alias_decisions,
            route_status=route_status,selected_provider_symbol=provider_symbol,
            frozen_identity_proofs=proofs,current_metadata_ciks=stock_inventory[symbol]['exact_current_ciks'],
            unadmitted_locating_ciks=stock_inventory[symbol]['non_admitted_locating_ciks'],
            historically_admitted=False)
    event_uses=defaultdict(lambda:dict(snapshot_paths=set(),slot_indices=set()))
    event_objects={}
    for slot_index,row in enumerate(canonical['rows']):
        path=row['selected_snapshot']
        if path is None: continue
        for event_id in row['audit']['event_ids']:
            key=(row['symbol'],event_id)
            event_objects[key]=events_by_snapshot[path][event_id]
            event_uses[key]['snapshot_paths'].add(path)
            event_uses[key]['slot_indices'].add(slot_index)
    linked=[]
    for (symbol,event_id),event in sorted(event_objects.items()):
        in_batch=symbol in batch
        relations=[]
        for obs in by_symbol.get(symbol,[]):
            require(obs['current_CIK_reference']['cik']==event['cik'], 'Current-reference CIK conflicts with canonical event')
            relations.append(dict(binding_id=obs['binding_id'],observation_date=obs['observation_date'],
                relative_to_event_filing_date=relation(obs['observation_date'],event['filingDate']),
                exact_SEC_accession_primary_document=exact_filing_document(obs['original_source']['url'],event),
                original_labelled_CIK_matches=event['cik'] in obs['original_labelled_CIKs'],
                original_class_bound_to_ticker=obs['class_bound_to_ticker'],
                explicit_bridge_effective_date=obs['explicit_same_security_bridge']['effective_date'] if obs['explicit_same_security_bridge'] else None,
                bridge_effective_date_relative_to_event=relation(obs['explicit_same_security_bridge']['effective_date'],event['filingDate']) if obs['explicit_same_security_bridge'] else None))
        same=[r['binding_id'] for r in relations if r['relative_to_event_filing_date']=='SAME_REPORTED_DATE']
        exact=[r['binding_id'] for r in relations if r['exact_SEC_accession_primary_document'] and r['original_labelled_CIK_matches'] and r['original_class_bound_to_ticker']]
        labelled_same=[r['binding_id'] for r in relations if r['relative_to_event_filing_date']=='SAME_REPORTED_DATE'
                       and r['original_labelled_CIK_matches'] and r['original_class_bound_to_ticker']]
        before=[r['observation_date'] for r in relations if r['relative_to_event_filing_date']=='BEFORE_EVENT_DATE']
        after=[r['observation_date'] for r in relations if r['relative_to_event_filing_date']=='AFTER_EVENT_DATE']
        linked.append(dict(symbol=symbol,event_id=event_id,accessionNumber=event['accessionNumber'],form=event['form'],
            filingDate=event['filingDate'],reportDate=event['reportDate'],cik=event['cik'],
            cik_basis='CIK field in the selected receipt-bound SEC submissions root; not a historical price-security join.',
            canonical_snapshot_references=[dict(path=p,sha256=sec_proofs[p]['sha256']) for p in sorted(event_uses[(symbol,event_id)]['snapshot_paths'])],
            frozen_price_route_ref=symbol,in_fourteen_series_batch=in_batch,
            observation_relations=relations,same_date_dated_ticker_observation_ids=same,
            exact_filing_CIK_class_original_ids=exact,same_date_labelled_CIK_class_original_ids=labelled_same,
            nearest_prior_observation_date=max(before) if before else None,
            nearest_later_observation_date=min(after) if after else None,
            direct_date_observation_edge_exists=bool(same),
            continuity_assumption_required_for_event_to_price_identity=in_batch and not bool(exact),
            join_status='EXACT_FILING_IDENTITY_POINT_WITH_PRICE_INTERVAL_STILL_PENDING' if exact else
                'DATED_TICKER_POINTS_PLUS_CURRENT_CIK_CONTINUITY_ASSUMPTION' if in_batch else 'OUTSIDE_FIXED_FOURTEEN_SERIES_IDENTITY_BATCH',
            historical_security_admitted=False,continuous_validity_period_inferred=False))
    link_index={(r['symbol'],r['event_id']):r for r in linked}
    cal_path='outputs/monthly_paired_target143/CALENDAR.json'
    calendar=checked.obj(cal_path,canonical['input_hashes'][cal_path])['sessions']
    slots=[]
    for row in canonical['rows']:
        symbol=row['symbol'];t=bisect_left(calendar,row['signal_date'])
        require(calendar[t]==row['signal_date'] and t>=504,'Price-window date axis')
        keys=[(symbol,e) for e in row['audit']['event_ids']] if row['audit'] else []
        lo,hi=calendar[t-504],calendar[t-42]
        slots.append(dict(signal_index=row['signal_index'],candidate_index=row['candidate_index'],symbol=symbol,
            signal_date=row['signal_date'],in_fourteen_series_batch=symbol in batch,
            canonical_source_closed=row['selected_snapshot'] is not None,
            original_report_keys=row['audit']['original_report_key_count'] if row['audit'] else 0,
            canonical_event_count=len(keys),
            same_date_observation_event_count=sum(link_index[k]['direct_date_observation_edge_exists'] for k in keys),
            exact_filing_identity_event_count=sum(bool(link_index[k]['exact_filing_CIK_class_original_ids']) for k in keys),
            consumed_price_date_window_metadata=[lo,hi],
            dated_observations_inside_price_window=[o['binding_id'] for o in by_symbol.get(symbol,[]) if lo<=o['observation_date']<=hi],
            price_window_identity_continuity_proven=False,historical_security_admitted=False))
    remaining=[]
    for symbol in sorted(set(stock_inventory)-batch):
        route=stock_routes[symbol]
        needs=['Dated original filing/issuer evidence tying the event CIK to the listed class, ticker and exchange at the used event dates.',
               'Bounded event-to-consumed-price-series identity review; point observations must not be expanded into invented valid-from/through periods.']
        if route['frozen_identity_proofs']:
            needs.append('Existing frozen identity_proofs can locate/reuse exact original SEC bytes; their stated current-CIK continuity assumption remains an assumption, not a dated class bridge.')
        else:
            needs.append('No exact-symbol frozen identity_proof in this manifest; identify a dated original issuer/security reference before joining this column.')
        if route['route_status']=='NO_EXACT_FROZEN_ROUTE_IN_THIS_AUDIT':
            needs.append('Resolve the explicit vendor request/alias route for this research column from frozen provenance; do not substitute a similar ticker.')
        if route['unadmitted_locating_ciks'] or route['alias_decisions']:
            needs.append('Adjudicate any relevant predecessor, class, ADR or ticker change with explicit original bridge evidence; do not concatenate candidate CIK histories.')
        remaining.append(dict(symbol=symbol,canonical_event_count=sum(r['symbol']==symbol for r in linked),
            original_stock_slots=sum(r['symbol']==symbol for r in slots),
            source_closed_slots=sum(r['symbol']==symbol and r['canonical_source_closed'] for r in slots),
            frozen_identity_proofs=route['frozen_identity_proofs'],route_status=route['route_status'],
            minimum_remaining_evidence=needs))
    require(len(slots)==3904 and len(remaining)==59, 'Fixed scope changed')
    batch_events=[r for r in linked if r['in_fourteen_series_batch']]
    summary=dict(original_stock_columns=73,dated_identity_series=14,dated_identity_observations=len(observations),
        canonical_symbol_event_pairs=len(linked),canonical_events_in_fourteen_series=len(batch_events),
        fourteen_series_events_with_same_date_ticker_observation=sum(bool(r['same_date_dated_ticker_observation_ids']) for r in batch_events),
        exact_filing_CIK_class_edges=sum(bool(r['exact_filing_CIK_class_original_ids']) for r in batch_events),
        same_date_labelled_CIK_class_edges=sum(bool(r['same_date_labelled_CIK_class_original_ids']) for r in batch_events),
        fourteen_series_events_requiring_continuity_assumption=sum(r['continuity_assumption_required_for_event_to_price_identity'] for r in batch_events),
        events_with_observations_before_and_after=sum(r['nearest_prior_observation_date'] is not None and r['nearest_later_observation_date'] is not None for r in batch_events),
        explicit_bridge_series=sorted({o['symbol'] for o in observations.values() if o['explicit_same_security_bridge']}),
        remaining_columns=59,remaining_columns_with_frozen_identity_proofs=sum(bool(r['frozen_identity_proofs']) for r in remaining),
        remaining_columns_with_exact_frozen_routes=sum(r['route_status']=='DIRECT_FROZEN_INTERMEDIATE_ROUTE' for r in remaining),
        new_security_admissions=0)
    for path in (SELF,'work/after156_event_scope_local_review.py','work/after156_identity_binding_batch.py',
                 'work/after156_event_price_identity.py','work/after156_filing_universe_inventory.py',
                 'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):
        src.read_bytes(path,identity['input_hashes'].get(path))
    src.recheck()
    result=dict(schema_version=1,status='CANONICAL_FILING_TO_DATED_IDENTITY_LINKS_WITH_CONTINUITY_ASSUMPTIONS_EXPLICIT',
        created_utc=datetime.now(timezone.utc).isoformat(),summary=summary,slot_summary=era_summary(slots),
        pointwise_dated_observations=observations,stock_price_routes=stock_routes,vendor_request_routes=providers,
        canonical_event_links=linked,stock_month_slots=slots,remaining_fifty_nine_columns=remaining,
        assumptions=['Current SEC CIK matching and no observed issuer change are explicitly continuity assumptions, not historical security proof.',
            'All before/after/same comparisons use raw reported filingDate versus original observation publication date; no acceptance clock or intraday order is inferred.',
            'Same reported date is a point observation only; an earnings release is not silently treated as the original 10-Q/10-K cover.',
            'The Meta Class-A/CUSIP bridge supports only the stated FB-to-META transition, not an invented all-history security epoch.',
            'Frozen route metadata and prior receipt-bound opaque price hashes are reused. This run reads no market CSV/NPZ bytes and does not recertify later extension/overlay identity.'],
        validation=dict(synthetic_checks=checks,canonical_source_rows_reconstructed=True,original_observation_text_locators_rechecked=True,
            source_receipt_and_current_SEC_references_rehashed=True,all_consumed_input_hashes_rechecked=True,
            original_3904_slot_grid_preserved=True),
        activity=dict(market_price_files_opened=0,price_values_read=0,acceptance_instants_interpreted=0,
            factors=0,orders=0,returns=0,new_trials=0,network_requests=0,observer_changes=0,frozen_files_modified=0),
        input_hashes=dict(sorted(src.hashes.items())))
    output=None
    if args.write:
        payload=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        with src.path(OUT).open('xb') as handle: handle.write(payload)
        require(src.path(OUT).read_bytes()==payload,'Output round-trip')
        output=dict(path=OUT,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    print(json.dumps(dict(summary=summary,slot_summary=result['slot_summary'],checks=checks,
        script_sha256=src.hashes[SELF],output=output),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
