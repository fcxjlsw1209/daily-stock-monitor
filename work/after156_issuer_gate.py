"""Reviewable issuer-link policy over frozen metadata, never certification."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from after156_event_scope_local_review import Sources, require

SELF='work/after156_issuer_gate.py'
OUT='work/after156_issuer_gate.json'
LINKS='work/after156_filing_issuer_links.json'
INVENTORY='work/after156_filing_universe_inventory.json'
PRIOR='work/monthly_profitability141_identity_review.json'
REVIEW='work/profitability141_identity_admission_review.json'
DELAYED='work/after156_delayed_partition.json'
ROUTES='work/after156_price_identity_routes.json'
PINS={
    LINKS:'cb85f9a88a65e451cbc6e60559c7960cf7f9eb7042dcaf9d30aef4af46d4ba93',
    INVENTORY:'e5157e2c6da08253baa9e523be2465261b442b506fad3baf7216664f47ff7286',
    PRIOR:'a67ac9706ce39cee9a461c97c920af092e413640a977b20794e3e4ede793fd50',
    REVIEW:'590446b750bb2d96d79056be4f0d86fe5cc61ec360cbfe48e75295f584c5f171',
    DELAYED:'11fb070a64723752ffedfb81b6fbbb00aacb0ce1ec5e3269e0d560c6e30e9a92',
    ROUTES:'4184ba45c51062c79ec41f8caabaad413749ae6dd0d68ae0738fabca2fa33ec7',
}
ERAS=[('early53',0,53),('recent35',53,88),('first88',0,88),('later34',88,122),('all122',0,122)]
RULES={
    'version':1,
    'scope':'All original73 stock columns and3904 stock-months; auxiliary-event mask only, never remove stocks from the original universe.',
    'event_source':'Use only the newly bound delayed-partition snapshots and each actual row audit event_ids/context_original_event_ids. Old issuer-links supplies identity observations/routes, never the event set.',
    'review_status':'A proposed, explicit continuity-assumption screen for review; no certified identity, clock adoption, strategy registration or factual claim that a current CIK proves historical continuity.',
    'ordinary':'A unique current SEC CIK plus exact frozen research-column/provider route may support a bounded same-issuer/class continuity assumption when this reviewed evidence records no specific contradiction. Every release need not repeat its share class. Current retrieval dates stay current.',
    'known_successors':'AVGO,DIS,XOM remain quarantined for this batch: the bound prior review records unresolved legal-entity/CIK successor history. Do not invent successor dates or stitch predecessors. Unreviewed extra CIK leads alone are not treated as proven successors.',
    'google':'The original dated2015-10-22 release states Alphabet succeeded Google after the2015-10-02 reorganization. It supports issuer succession, not a per-class conversion ratio. Permit only windows whose full consumed price interval starts on/after2015-10-22, under the explicit same-GOOGL-class/provider continuity assumption. Earlier crossing windows remain blocked.',
    'meta':'Use the explicit original same-Class-A/unchanged-CUSIP FB-to-META transition and the frozen research_equity_FB->META route. Reject the reused current FB ETF route. This is a retrospective bridge; retain2022-05-31 publication and2022-06-09 effective date, not an invented earlier known-at.',
    'berkshire':'BRK.B means frozen Class-B spelling of SEC BRK-B only. Never substitute BRK-A or treat CIK-level issuer identity as share-class equality.',
    'shell':'Original evidence supports issuer-level A/B ADS assimilation only. Permit a continuity-assumption candidate only if the entire consumed price interval starts on/after2022-01-31; do not choose or splice a pre-assimilation A/B price series.',
    'skhy':'Original listing evidence dates the Nasdaq ADR to2026-07-10. Block any consumed window reaching earlier; never substitute Korean ordinary shares. Missing current CIK remains unresolved.',
    'china_mobile':'Local original NYSE suspension evidence and frozen Tiingo CHL route are retained. No active current-CIK link is inferred; require the stated historic CIK proof before joining any event. Do not treat later OTC history as NYSE.',
    'missing_eleven':'Inspect frozen later loader/provider metadata for BABA,BUD,CHL,HSBC,NVO,NVS,SAP,SHEL,SKHY,TM,TSM; absence from the intermediate alias manifest alone is not absence of a route. Read no market data file.',
    'per_event':'Apply the declared expected-CIK rule to every actual delayed event and context record and preserve its snapshot/original occurrence references. Missing IDs or contradictory CIKs block the slot; no favorable fallback snapshot.',
    'coverage':'Report issuer-policy candidates, candidates with delayed source closure, and candidates satisfying the already computed hypothetical partition. All remain conditional; no acceptance timestamps or price values are processed.',
    'non_actions':'No network, market-file reads, factors, ranks, orders, returns, central-state changes, observer changes, or frozen-file modifications.'}


def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def day_relation(day, signal):
    return 'BEFORE' if day<signal else 'AFTER' if day>signal else 'SAME_DATE'


def classify(symbol,route,current_ciks):
    if symbol in ('AVGO','DIS','XOM'):
        return dict(category='QUARANTINE_UNRESOLVED_CIK_SUCCESSOR',candidate=False,minimum_price_window_start=None)
    if symbol=='CHL':
        return dict(category='HISTORICAL_ADS_CIK_JOIN_UNRESOLVED',candidate=False,minimum_price_window_start=None)
    if symbol=='SKHY':
        return dict(category='ADR_LISTING_AND_CIK_RESTRICTION',candidate=False,minimum_price_window_start='2026-07-10')
    if not route:
        return dict(category='FROZEN_ROUTE_UNRESOLVED',candidate=False,minimum_price_window_start=None)
    if len(current_ciks)!=1:
        return dict(category='CURRENT_CIK_MISSING_OR_AMBIGUOUS',candidate=False,minimum_price_window_start=None)
    category,boundary=('GOOGL_POST_REORGANIZATION_CONTINUITY_ASSUMPTION','2015-10-22') if symbol=='GOOGL' else (
        ('SHEL_POST_ADS_ASSIMILATION_CONTINUITY_ASSUMPTION','2022-01-31') if symbol=='SHEL' else (
        ('META_EXPLICIT_CLASS_A_BRIDGE_PLUS_CONTINUITY_ASSUMPTION',None) if symbol=='research_equity_FB' else (
        ('BRK_B_EXPLICIT_SPELLING_PLUS_CLASS_B_CONTINUITY_ASSUMPTION',None) if symbol=='BRK.B' else
        ('SAME_CIK_EXACT_ROUTE_CONTINUITY_ASSUMPTION',None))))
    return dict(category=category,candidate=True,minimum_price_window_start=boundary)


def slot_decision(column,start,ciks):
    reasons=[]
    if not column['candidate']: reasons.append(column['category'])
    bound=column['minimum_price_window_start']
    if bound and start<bound: reasons.append('CONSUMED_PRICE_WINDOW_CROSSES_UNBRIDGED_HISTORY')
    if any(c not in column['expected_current_ciks'] for c in ciks): reasons.append('ACTUAL_EVENT_CIK_CONTRADICTS_COLUMN')
    return not reasons,reasons


def tests():
    require(not classify('XOM',True,['1'])['candidate'],'Known successor cannot pass generic continuity')
    require(classify('AAPL',True,['1'])['candidate'],'Ordinary declared assumption need not require every release class')
    require(not classify('AAPL',True,[])['candidate'],'Missing CIK')
    google=dict(classify('GOOGL',True,['1']),expected_current_ciks=['1'])
    require(not slot_decision(google,'2015-10-21',['1'])[0],'Prebridge window blocked')
    require(slot_decision(google,'2015-10-22',['1'])[0],'Boundary inclusive')
    require(not slot_decision(google,'2016-01-01',['2'])[0],'Actual event CIK mismatch')
    require(classify('research_equity_FB',True,['1'])['candidate'],'Explicit Meta bridge')
    return 7


def supplements(src,data):
    routes=data[ROUTES]
    source_path='outputs/monthly_2016_extension/SOURCE_HASHES.json'
    hashes=src.read(source_path,routes['input_hashes'][source_path])
    result={}
    foreign_path='outputs/monthly_reconstruction_v2/foreign/ETF_MANIFEST.json'
    foreign=src.read(foreign_path,hashes[str(src.path(foreign_path))])
    for row in foreign:
        if row['symbol'] not in ('BABA','NVO','SAP','TM','TSM'): continue
        require(hashes[row['file']]==row['sha256'],'Foreign loader source hash')
        result[row['symbol']]=dict(kind='FROZEN_FOREIGN_LOADER_MANIFEST',provider='Yahoo chart API',
            provider_symbol=row['symbol'],metadata_path=foreign_path,metadata_sha256=src.hashes[foreign_path],
            market_source_path=str(Path(row['file']).relative_to(src.root)),market_source_sha256=row['sha256'],
            url=row['url'],source_note=row['source'],reported_first=row['first'],reported_last=row['last'],
            retrieval_date_in_this_metadata=None,market_file_read=False)
    for symbol,folder in [('BUD','monthly_2016_extension'),('HSBC','monthly_2016_extension'),('NVS','monthly_2016_extension'),
                           ('SHEL','monthly_published44_reconstruction'),('SKHY','monthly_published44_reconstruction')]:
        path='outputs/'+folder+'/prices/'+symbol+'.source.json';row=src.read(path)
        price='outputs/'+folder+'/prices/'+symbol+'.csv'
        require(row['symbol']==symbol and hashes[str(src.path(price))]==row['sha256'],'Later loader source hash')
        result[symbol]=dict(kind='FROZEN_LATER_LOADER_RECEIPT',provider=row['source'],provider_symbol=symbol,
            metadata_path=path,metadata_sha256=src.hashes[path],market_source_path=price,market_source_sha256=row['sha256'],
            retrieval_date_in_this_metadata=row['retrieved_utc'],reported_first=row.get('first'),reported_last=row.get('last'),market_file_read=False)
    price='outputs/dynamic_top100/additional_sources/tiingo/CHL_prices.json'
    rec_path='outputs/dynamic_top100/additional_sources/tiingo/CHL_prices.provenance.json';rec=src.read(rec_path)
    require(rec['sha256']==hashes[str(src.path(price))],'CHL source hash')
    result['CHL']=dict(kind='FROZEN_EXPLICIT_TIINGO_CHL_LOADER',provider='Tiingo',provider_symbol='CHL',
        metadata_path=rec_path,metadata_sha256=src.hashes[rec_path],market_source_path=price,
        market_source_sha256=rec['sha256'],receipt=rec,market_file_read=False,
        loader_restriction='monthly_2016_extension/extend.py limits this CHL series to dates before2021-01-01.')
    for path in ('outputs/monthly_2016_extension/extend.py','outputs/monthly_factor_search_v2/data_inputs.py'):
        src.read_bytes(path)
    require(len(result)==11,'Eleven supplemental metadata routes')
    return result


def special_evidence(src,links):
    base='outputs/monthly_earnings_repair/alphabet/raw/f63e0d074ae30f691be9'
    raw=src.read_bytes(base+'.pdf','87a1c0665b7d300a3fdfa67134c71cbc04530c795dbbecc2a3cd3be22b203e47')
    rec=src.read(base+'.meta.json','6c8cdc9df27a1557b16aead51720504dea79d6b434dd1cab3d97b551a8091564')
    require(rec['sha256']==hashlib.sha256(raw).hexdigest(),'Alphabet original receipt')
    text=' '.join(src.read_bytes(base+'.txt','c526305c0f170e058904d9f254814e9b34137fc5afac7ea914b8018757e3bbd8').decode().split())
    required='the successor issuer to, and parent holding company of, Google Inc.'
    require(required in text and 'became effective on October 2, 2015' in text and 'October 22, 2015' in text,'Alphabet succession excerpt')
    google=dict(original_path=base+'.pdf',original_sha256=rec['sha256'],receipt_path=base+'.meta.json',
        retrieved_at=rec['retrieved_at'],publication_date='2015-10-22',stated_reorganization_effective='2015-10-02',
        original_source_url=rec['source_url'],excerpt=required,
        limit='Issuer succession supported. No per-class exchange ratio is inferred. PDF text is an existing pinned extraction.')
    meta=next(o for o in links['pointwise_dated_observations'].values() if o['symbol']=='research_equity_FB')
    for p,h in ((meta['original_source']['path'],meta['original_source']['sha256']),
                (meta['original_source']['receipt_path'],meta['original_source']['receipt_sha256'])):src.read_bytes(p,h)
    require(meta['explicit_same_security_bridge']['same_class_A_and_unchanged_CUSIP_supported'],'Meta explicit bridge')
    shell=src.read('outputs/monthly_earnings_repair/batch_health_energy_foreign/SHEL_IDENTITY.json')
    src.read_bytes(shell['evidence_path'],shell['sha256'])
    skhy_path='outputs/monthly_earnings_repair/batch_technology_industrial/SKHY/raw/adr_identity.meta.json'
    skhy=src.read(skhy_path);src.read_bytes(skhy_path.replace('.meta.json','.bin'),skhy['sha256'])
    policy_path='outputs/monthly_earnings_repair/batch_health_energy_foreign/IDENTITY_POLICY.json'
    policies=src.read(policy_path);chl=next(r for r in policies if r['ticker']=='CHL')
    for e in chl['evidence']:src.read_bytes(e['evidence_path'],e['sha256'])
    return dict(GOOGL=google,META=meta,SHEL=shell,SKHY=skhy,CHL=chl)


def current_sources(src,inventory,stock):
    """Recheck the actual current link and preserve its real observation time."""
    snapshots={s['source']['path']:s for s in inventory['submission_snapshots']}
    result=[]
    ticker={'research_equity_FB':'META','BRK.B':'BRK-B'}.get(stock['symbol'],stock['symbol'])
    for cik in stock['exact_current_ciks']:
        for path in stock['local_snapshot_paths'][cik]:
            proof=snapshots[path]['source']
            raw=src.read(path,proof['sha256'])
            require(str(raw['cik']).zfill(10)==cik,'Current source CIK differs')
            # Same-CIK alternate snapshots can omit the ticker. Such snapshots
            # do not independently establish the current ticker-to-CIK link.
            rec=None
            if proof['source_receipt_verified']:
                rec=src.read(proof['receipt'],inventory['input_hashes'][proof['receipt']])
                require(rec['sha256']==proof['sha256'],'Current source receipt differs')
            result.append(dict(proof,cik=cik,sec_name=raw['name'],current_tickers=raw.get('tickers',[]),
                directly_lists_requested_current_ticker=ticker in raw.get('tickers',[]),
                original_receipt=rec,observation_is_historical_effective_interval=False))
    require(not stock['exact_current_ciks'] or any(r['directly_lists_requested_current_ticker'] for r in result),
        'Declared current CIK lacks actual ticker link')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--inspect',action='store_true');p.add_argument('--rules-only',action='store_true')
    p.add_argument('--write',action='store_true');p.add_argument('--rules-sha256');p.add_argument('--scope-sha256');args=p.parse_args()
    src=Sources();data={path:src.read(path,sha) for path,sha in PINS.items()}
    links=data[LINKS]; prior={r['symbol']:r for r in data[PRIOR]['rows']}
    missing=[s for s,r in links['stock_price_routes'].items() if r['route_status']=='NO_EXACT_FROZEN_ROUTE_IN_THIS_AUDIT']
    if args.inspect:
        print(json.dumps(dict(missing_eleven=missing, details={s:dict(route=links['stock_price_routes'][s],prior=prior[s]) for s in missing},
            focused_prior={s:prior[s] for s in ('GOOGL','AVGO','XOM','research_equity_FB','BRK.B')},
            explicit_conflict_statuses={s:r for s,r in prior.items() if 'conflict' in r['status'] or 'unresolved' in r['status']}),ensure_ascii=False,indent=2))
        return
    checks=tests();extra=supplements(src,data);special=special_evidence(src,links)
    inventory={r['symbol']:r for r in data[INVENTORY]['stocks']}
    columns={}
    for symbol,route in sorted(links['stock_price_routes'].items()):
        current_evidence=current_sources(src,data[INVENTORY],inventory[symbol])
        current_by_path={r['path']:r for r in current_evidence}
        proof_records=[]
        for proof in route['frozen_identity_proofs']:
            raw=src.read(proof['source_path'],proof['source_sha256'])
            require(str(raw.get('cik')).zfill(10)==proof['current_CIK'],'Frozen CIK proof differs')
            observed=current_by_path.get(proof['source_path'])
            proof_records.append(dict(proof,retrieval_time_is_current_only=True,
                original_observation_source=observed,
                observation_time_unavailable_in_current_inventory=observed is None))
        has_route=bool(route['direct_routes'] or route['alias_decisions'] or symbol in extra)
        ciks=inventory[symbol]['exact_current_ciks']
        classification=classify(symbol,has_route,ciks)
        columns[symbol]=dict(classification,expected_current_ciks=ciks,
            direct_frozen_route=route['direct_routes'],explicit_alias_route=route['alias_decisions'],
            supplemental_route=extra.get(symbol),frozen_identity_proofs=proof_records,
            current_SEC_source_evidence=current_evidence,
            dated_observation_ids=[o['binding_id'] for o in links['pointwise_dated_observations'].values() if o['symbol']==symbol],
            prior_review_status=prior[symbol]['status'],prior_note=prior[symbol].get('note',prior[symbol].get('foreign_reporting_note')),
            explicit_assumption='The selected frozen provider column represents the same listed class and issuer over only the consumed event/beta/formation interval; no known contradiction in this bounded review is proof of continuity.',
            certified_identity=False)
    scope_sha=digest(columns);rules_sha=digest(RULES)
    if args.rules_only:
        print(json.dumps(dict(rules=RULES,rules_sha256=rules_sha,scope_sha256=scope_sha,
            candidate_scope={s:{k:r[k] for k in ('category','candidate','minimum_price_window_start')} for s,r in columns.items()},
            eleven_metadata_routes=sorted(extra),synthetic_checks=checks),ensure_ascii=False,indent=2));return
    require(args.rules_sha256==rules_sha and args.scope_sha256==scope_sha,'Bind rules and candidate scope before computing coverage')
    delayed=data[DELAYED]
    cal_path='outputs/monthly_paired_target143/CALENDAR.json'
    sessions=src.read(cal_path,delayed['input_hashes'][cal_path])['sessions']
    catalogs={p:{e['event_id']:e for e in s['events']} for p,s in delayed['snapshots'].items()}
    proof_by_path={}
    for s in data[INVENTORY]['submission_snapshots']:
        proof_by_path[s['source']['path']]=s['source']
        for p in s['advertised_pages']:
            if 'source' in p:proof_by_path[p['source']['path']]=p['source']
    verified=set();rows=[];actual_events={};old_events={(e['symbol'],e['event_id']) for e in links['canonical_event_links']}
    for source_row in delayed['rows']:
        symbol=source_row['symbol'];t=source_row['signal_session_index'];path=source_row['selected_snapshot']
        lo,hi=sessions[t-504],sessions[t-42]
        ids=[] if source_row['audit'] is None else list(dict.fromkeys(source_row['audit']['event_ids']+source_row['audit']['context_original_event_ids']))
        events=[catalogs[path][i] for i in ids] if path else []
        allowed,reasons=slot_decision(columns[symbol],lo,[e['cik'] for e in events])
        for event in events:
            for occurrence in event['occurrences']:
                p=occurrence['source_path']
                if p not in verified:
                    proof=proof_by_path[p];src.read_bytes(p,proof['sha256'])
                    require(proof['source_receipt_verified'],'Actual event source lacks original receipt')
                    rec=src.read(proof['receipt'],data[INVENTORY]['input_hashes'][proof['receipt']])
                    require(rec['sha256']==proof['sha256'],'Actual event receipt mismatch');verified.add(p)
            key=(symbol,event['event_id'])
            record=dict(symbol=symbol,event_id=event['event_id'],cik=event['cik'],
                accessionNumber=event['accessionNumber'],form=event['form'],filingDate=event['filingDate'],
                reportDate=event['reportDate'],original_occurrences=event['occurrences'],
                was_in_old_issuer_links=key in old_events,classification=columns[symbol]['category'],
                same_CIK_as_declared_column=event['cik'] in columns[symbol]['expected_current_ciks'],
                dated_observation_relations=[dict(binding_id=o['binding_id'],observation_date=o['observation_date'],
                    relative_to_actual_event=day_relation(o['observation_date'],event['filingDate']))
                    for o in links['pointwise_dated_observations'].values() if o['symbol']==symbol],
                certified_identity=False)
            if key not in actual_events:
                actual_events[key]=dict(record,selected_snapshot_paths=[path])
            else:
                existing=actual_events[key]
                require(all(existing[k]==record[k] for k in ('cik','accessionNumber','form','filingDate','reportDate')),
                    'Same event has conflicting actual delayed metadata')
                if path not in existing['selected_snapshot_paths']:existing['selected_snapshot_paths'].append(path)
                for occurrence in event['occurrences']:
                    if occurrence not in existing['original_occurrences']:existing['original_occurrences'].append(occurrence)
        partition_pass=bool(source_row['partition'] and source_row['partition']['hypothetical_partition_pass'])
        rows.append(dict(signal_index=source_row['signal_index'],candidate_index=source_row['candidate_index'],symbol=symbol,
            signal_date=source_row['signal_date'],selected_snapshot=path,actual_delayed_event_ids=source_row['audit']['event_ids'] if source_row['audit'] else [],
            context_original_event_ids=source_row['audit']['context_original_event_ids'] if source_row['audit'] else [],
            consumed_price_window_metadata=[lo,hi],issuer_assumption_screen_candidate=allowed,issuer_block_reasons=reasons,
            delayed_source_closed=path is not None,delayed_hypothetical_partition_pass=partition_pass,
            combined_hypothetical_candidate=allowed and partition_pass,actual_identity_admitted=False,actual_clock_admitted=False))
    require(len(rows)==3904,'Original denominator')
    check=[r for r in rows if r['symbol']=='C' and r['signal_index']==4 and r['candidate_index']==7]
    require(len(check)==1 and any(actual_events[('C',i)]['accessionNumber']=='0000831001-15-000157' for i in check[0]['actual_delayed_event_ids']),
            'Known newly included delayed C filing must be joined')
    summaries={}
    for name,start,end in ERAS:
        group=[r for r in rows if start<=r['signal_index']<end];passed=[r for r in group if r['combined_hypothetical_candidate']]
        summaries[name]=dict(original_slots=len(group),issuer_assumption_candidates=sum(r['issuer_assumption_screen_candidate'] for r in group),
            source_closed_issuer_candidates=sum(r['issuer_assumption_screen_candidate'] and r['delayed_source_closed'] for r in group),
            combined_hypothetical_candidates=len(passed),combined_fraction=len(passed)/len(group),
            distinct_symbols=len({r['symbol'] for r in passed}),block_reason_counts=dict(Counter(x for r in group for x in r['issuer_block_reasons'])),
            actual_admitted_slots=0)
    for p in (SELF,'AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb'):src.read_bytes(p)
    src.recheck()
    report=dict(status='PROPOSED_BOUNDED_ISSUER_CONTINUITY_ASSUMPTION_SCREEN_NOT_CERTIFICATION',schema_version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),rules=RULES,rules_sha256=rules_sha,scope_sha256=scope_sha,
        columns=columns,special_original_evidence=special,summary=summaries,
        column_category_counts=dict(Counter(c['category'] for c in columns.values())),
        actual_delayed_event_links=list(actual_events.values()),rows=rows,
        event_summary=dict(actual_symbol_event_pairs=len(actual_events),new_pairs_not_in_old_links=sum(k not in old_events for k in actual_events),
            known_new_C_filing_verified=True),
        remaining=['This is an explicitly reviewable assumption policy, not consent/adoption of assumptions or a positive issuer certificate.',
            'AVGO/DIS/XOM successor histories and CHL historic CIK linkage remain quarantined; GOOGL/SHEL crossing windows remain blocked.',
            'Known-at/cutoff exceptions are left to the separate preflight; current retrieval times and retrospective bridges are preserved.',
            'The delayed date-clock hypothesis, full chronology, positive-finite prices and feature/strategy admission remain separate.'],
        validation=dict(synthetic_checks=checks,delayed_ids_and_context_used=True,actual_event_original_hashes_rechecked=True,
            all_consumed_inputs_rehashed=True,known_C_regression_passed=True),
        activity=dict(price_files_read=0,price_values_read=0,acceptance_instants_interpreted=0,new_factors=0,orders=0,returns=0,
            network_requests=0,registrations=0,observer_changes=0,frozen_files_modified=0),input_hashes=dict(sorted(src.hashes.items())))
    output=None
    if args.write:
        payload=(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        with src.path(OUT).open('xb') as f:f.write(payload)
        require(src.path(OUT).read_bytes()==payload,'Output roundtrip')
        output=dict(path=OUT,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
    print(json.dumps(dict(summary=summaries,column_category_counts=report['column_category_counts'],event_summary=report['event_summary'],
        rules_sha256=rules_sha,scope_sha256=scope_sha,script_sha256=src.hashes[SELF],output=output),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
