"""Join dated identity evidence to opaque price-source hashes, never price values."""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re

from after156_identity_binding_batch import Inputs, document, literal, normalize, require, timestamp

ROOT = Path(__file__).resolve().parents[1]
OUT = 'work/after156_event_price_identity.json'
BINDINGS = 'work/after156_identity_binding_batch.json'
ROUTES = 'work/after156_price_identity_routes.json'
UNION = 'outputs/index_union/'
PINS = {
    BINDINGS: 'a381874f329871a0e602606e3325fd923274feb35b7d8f8c663ec0a8941c921f',
    ROUTES: '4184ba45c51062c79ec41f8caabaad413749ae6dd0d68ae0738fabca2fa33ec7',
    UNION + 'NEW_PRICE_VALIDATION.json': 'a2a74ced299329bd29b3a175519bb9b20b71608a97976570ff83a7baa4fb7d56',
    UNION + 'sources/yahoo_identity/META.json': '9e99e83b444e98e3627fc42ada4d5a7b7d98e3db019aaf05fa5003bbc666768d',
    UNION + 'sources/yahoo_identity/FB.json': 'ee78b840f10cfa10c2bf13530a567caea3b3cbb3bef9b295cf86e52ac2c202bf',
    'work/after156_identity_binding_batch.py': '326bdc09a2deff561e726bd9d3a65b5a7be5f9e7c3d476c0114758fe8633d2a7',
}


class CikFields(HTMLParser):
    """Only labelled document fields qualify; URLs and unrelated numbers do not."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.active = None
        self.parts = []
        self.fields = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == 'dei:entitycentralindexkey' or attrs.get('name', '').lower() == 'dei:entitycentralindexkey':
            self.active, self.parts = tag, []

    def handle_endtag(self, tag):
        if tag == self.active:
            text = normalize(''.join(self.parts))
            if re.fullmatch(r'\d{1,10}', text):
                self.fields.append(dict(cik=text.zfill(10), basis='Original labelled dei:EntityCentralIndexKey field', text=text))
            self.active = None

    def handle_data(self, text):
        if self.active:
            self.parts.append(text)


def original_ciks(raw):
    text = raw.decode('utf-8', errors='replace')
    parser = CikFields()
    parser.feed(text)
    fields = list(parser.fields)
    for match in re.finditer(r'CENTRAL INDEX KEY\s*:\s*(\d{1,10})\b', text, re.I):
        fields.append(dict(cik=match.group(1).zfill(10), basis='Original labelled CENTRAL INDEX KEY', text=match.group()))
    return fields


def within(day, first, last):
    return date.fromisoformat(first) <= date.fromisoformat(day) <= date.fromisoformat(last)


def edge_assessment(source_ticker, provider_symbol, metadata, bridge=None):
    if metadata and metadata.get('symbol') != provider_symbol:
        return 'REJECT_VENDOR_METADATA_SYMBOL_CONFLICT'
    if metadata and metadata.get('instrumentType') not in (None, 'EQUITY'):
        return 'REJECT_VENDOR_INSTRUMENT_TYPE_CONFLICT'
    if source_ticker == provider_symbol:
        return 'DIRECT_DATED_TICKER_TO_VENDOR_REQUEST'
    if bridge and bridge.get('from_ticker') == source_ticker and bridge.get('to_ticker') == provider_symbol and bridge.get('same_class_A_and_unchanged_CUSIP_supported') is True:
        return 'EXPLICIT_SAME_SECURITY_TICKER_BRIDGE_TO_VENDOR_REQUEST'
    return 'REJECT_UNPROVED_ALIAS'


def verify_receipt(receipt, raw, symbol):
    require(receipt['sha256'] == hashlib.sha256(raw).hexdigest(), 'Raw vendor file differs from receipt')
    require(receipt['symbol'] == symbol, 'Receipt requested a different vendor symbol')
    require(receipt['provider'] == 'Yahoo Finance via yfinance', 'Unreviewed provider')
    require(date.fromisoformat(receipt['start']) < date.fromisoformat(receipt['end_exclusive']), 'Bad requested interval')
    return timestamp(receipt['retrieved_at'])


def tests():
    checks = []
    def ok(name, value):
        require(value, name); checks.append(name)
    def bad(name, fn):
        try: fn()
        except (ValueError, KeyError): checks.append(name)
        else: raise ValueError(name)
    ok('ordinary labelled CIK', original_ciks(b'<ix:nonNumeric name="dei:EntityCentralIndexKey">789019</ix:nonNumeric>')[0]['cik'] == '0000789019')
    ok('labelled header CIK', original_ciks(b'CENTRAL INDEX KEY: 0000789019')[0]['cik'] == '0000789019')
    ok('CIK URL is not document field', not original_ciks(b'<a href="/CIK-0000789019/a">Report</a>'))
    ok('numeric amount is not CIK', not original_ciks(b'<p>789019</p>'))
    ok('invalid labelled CIK stays absent', not original_ciks(b'<dei:EntityCentralIndexKey>789019x</dei:EntityCentralIndexKey>'))
    ok('direct ticker is only dated request edge', edge_assessment('MSFT', 'MSFT', None) == 'DIRECT_DATED_TICKER_TO_VENDOR_REQUEST')
    ok('same name cannot bridge ticker', edge_assessment('FB', 'META', {'symbol':'META','instrumentType':'EQUITY'}) == 'REJECT_UNPROVED_ALIAS')
    bridge = dict(from_ticker='FB', to_ticker='META', same_class_A_and_unchanged_CUSIP_supported=True)
    ok('explicit class bridge accepted', edge_assessment('FB','META',{'symbol':'META','instrumentType':'EQUITY'},bridge).startswith('EXPLICIT_'))
    ok('bridge missing class support rejected', edge_assessment('FB','META',None,dict(bridge,same_class_A_and_unchanged_CUSIP_supported=False)) == 'REJECT_UNPROVED_ALIAS')
    ok('reused ticker ETF rejected', edge_assessment('FB','FB',{'symbol':'FB','instrumentType':'ETF'},bridge) == 'REJECT_VENDOR_INSTRUMENT_TYPE_CONFLICT')
    ok('metadata symbol conflict rejected', edge_assessment('MSFT','MSFT',{'symbol':'OTHER','instrumentType':'EQUITY'}) == 'REJECT_VENDOR_METADATA_SYMBOL_CONFLICT')
    ok('range is inclusive only for reported observations', within('2020-01-02','2020-01-02','2020-01-03') and not within('2020-01-04','2020-01-02','2020-01-03'))
    bad('invalid date rejected', lambda: within('2020-02-30','2020-01-01','2020-03-01'))
    raw=b'opaque synthetic bytes'
    receipt=dict(sha256=hashlib.sha256(raw).hexdigest(), symbol='MSFT', provider='Yahoo Finance via yfinance', start='2020-01-01', end_exclusive='2020-01-03', retrieved_at='2020-01-04T01:00:00Z')
    ok('opaque bytes receipt verified without parsing prices', verify_receipt(receipt,raw,'MSFT') == '2020-01-04T01:00:00+00:00')
    bad('vendor bytes conflict rejected', lambda:verify_receipt(receipt,b'other','MSFT'))
    bad('vendor request symbol conflict rejected', lambda:verify_receipt(receipt,raw,'NVDA'))
    bad('naive retrieval rejected', lambda:verify_receipt(dict(receipt,retrieved_at='2020-01-04T01:00:00'),raw,'MSFT'))
    return checks


def vendor_chain(src, route, validation, executions):
    symbol = route['symbol']
    require(route['provider'] == 'yahoo', 'Fixed batch expects Yahoo source')
    candidates = [x for x in validation['files'] if x['symbol'] == symbol and x['provider'] == 'yahoo' and x['normalized_file'] == route['input_file']]
    require(len(candidates) == 1, 'Ambiguous normalization provenance')
    norm = candidates[0]
    require(norm['normalized_sha256'] == route['input_sha256'], 'Normalization route mismatch')
    source = str(Path(norm['source']).relative_to(ROOT))
    raw = src.read(source, norm['sha256'])  # Opaque hashing only: no CSV or numeric parser.
    receipt_path = str(Path(source).with_suffix('.provenance.json'))
    receipt = src.obj(receipt_path)
    retrieved = verify_receipt(receipt, raw, symbol)
    src.read(UNION + route['input_file'], route['input_sha256'])
    src.read(UNION + route['raw_file'], route['raw_sha256'])
    matches = [x for x in executions['files'] if x['symbol'] == symbol]
    require(len(matches) == 1, 'Ambiguous execution source')
    execution = matches[0]
    require(execution['input_file'] == route['input_file'] and execution['input_sha256'] == route['input_sha256'] and execution['file'] == route['raw_file'] and execution['sha256'] == route['raw_sha256'], 'Execution transformation route mismatch')
    metadata_path = UNION + 'sources/yahoo_identity/' + symbol + '.json'
    metadata = src.obj(metadata_path, PINS.get(metadata_path)) if src.path(metadata_path).is_file() else None
    require(metadata is None or set(metadata) <= {'symbol','shortName','longName','exchangeName','fullExchangeName','instrumentType','firstTradeDate'}, 'Unexpected vendor metadata schema; do not read quote fields')
    return dict(provider='Yahoo Finance via yfinance', provider_symbol=symbol,
        original_source=source, original_sha256=norm['sha256'], receipt_path=receipt_path,
        receipt_sha256=src.hashes[receipt_path], actual_retrieved_at=retrieved,
        requested_from=receipt['start'], requested_end_exclusive=receipt['end_exclusive'],
        reported_observation_bounds=dict(first=norm['first'],last=norm['last'],basis='Frozen normalization inventory metadata; dates not recomputed from price values'),
        normalized_source=UNION+route['input_file'], normalized_sha256=route['input_sha256'],
        execution_source=UNION+route['raw_file'], execution_sha256=route['raw_sha256'],
        opaque_byte_hash_chain_verified=True, original_receipt_identity_verified=receipt.get('identity_verified'),
        vendor_identity_metadata=metadata, vendor_identity_metadata_path=metadata_path if metadata else None,
        vendor_identity_metadata_sha256=src.hashes.get(metadata_path),
        vendor_identity_metadata_separate_receipt_present=False,
        limits=['Vendor requested-symbol receipt and transformation hashes prove data provenance, not historical class identity.',
                'Requested date range is not a trading-life assertion. Reported observation bounds do not prove every session exists.',
                'This ends at the frozen intermediate source route; later extension/overlay identity has not been newly certified.'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--test-only', action='store_true')
    args=parser.parse_args()
    check_names=tests()
    if args.test_only:
        print(json.dumps(dict(passed=True,checks=len(check_names),names=check_names))); return
    src=Inputs()
    binding=src.obj(BINDINGS,PINS[BINDINGS])
    routes=src.obj(ROUTES,PINS[ROUTES])
    require(len(binding['bindings'])==110, 'Fixed binding scope changed')
    for name in ('AGENTS.md','LOCAL_SETUP.md','run_local.sh','local_run.py','local.sandbox.sb','work/after156_event_price_identity.py','work/after156_identity_binding_batch.py'):
        src.read(name,PINS.get(name))
    alias=src.obj(UNION+'alias_resolved_inputs/MANIFEST.json',routes['input_hashes'][UNION+'alias_resolved_inputs/MANIFEST.json'])
    inputs=src.obj(UNION+'input_diagnostics/MANIFEST.json',routes['input_hashes'][UNION+'input_diagnostics/MANIFEST.json'])
    validation=src.obj(UNION+'NEW_PRICE_VALIDATION.json',PINS[UNION+'NEW_PRICE_VALIDATION.json'])
    executions=src.obj(UNION+'EXECUTION_CANDIDATES.json')
    for name in ('normalize_new_prices.py','build_execution_candidates.py','collect_missing_yahoo.py','consolidate_security_aliases.py','assemble_alias_resolved_inputs.py'):
        src.read(UNION+name,routes['input_hashes'].get(UNION+name))
    route_by_symbol={x['symbol']:x for x in routes['rows']}
    vendors={}
    output=[]
    for item in binding['bindings']:
        symbol=item['symbol']
        original=item['source']
        raw=src.read(original['path'],original['original_sha256'])
        src.read(original['receipt_path'],original['receipt_sha256'])
        if raw.startswith(b'%PDF'):
            extraction=item['extraction']
            text=normalize(src.read(extraction['path'],extraction['sha256']).decode('utf-8'))
            ciks=[]
        else:
            text,_=document(raw); ciks=original_ciks(raw)
        for key in ('issuer_name','exchange_ticker','security_class_literal'):
            proof=item['identity'].get(key)
            if proof:
                require(text[proof['start']:proof['end']]==proof['text'], 'Original identity locator differs')
        route_row=route_by_symbol[symbol]
        bridge=item.get('transition')
        if symbol=='research_equity_FB':
            choices=[x for x in alias['alias_decisions'] if x['security_id']==symbol]
            require(len(choices)==1 and choices[0]['symbols']==['FB','META'] and choices[0]['price_provider_symbol']=='META','Frozen Meta alias changed')
            vendor_symbol='META'
            candidate_routes=[x for x in inputs['price_sources'] if x['symbol']==vendor_symbol]
            routing=dict(kind='Frozen explicit alias decision', alias_decision=choices[0], current_route_audit_has_direct_symbol=False)
            source_ticker='FB'
        else:
            candidate_routes=route_row['exact_internal_symbol_source_routes']
            vendor_symbol=symbol; source_ticker=symbol
            routing=dict(kind='Frozen direct symbol route', intermediate_issuer=route_row['intermediate_issuer'], current_route_audit_has_direct_symbol=True)
        require(len(candidate_routes)==1, 'Missing or ambiguous fixed vendor route')
        if vendor_symbol not in vendors:
            vendors[vendor_symbol]=vendor_chain(src,candidate_routes[0],validation,executions)
        vendor=vendors[vendor_symbol]
        edge=edge_assessment(source_ticker,vendor_symbol,vendor['vendor_identity_metadata'],bridge)
        require(not edge.startswith('REJECT'), 'Selected source identity conflicts: '+symbol)
        current=item['current_registration_reference']
        sec=src.obj(current['path'],current['sha256'])
        src.read(current['receipt_path'],current['receipt_sha256'])
        require(str(sec['cik']).zfill(10)==current['cik'], 'Current reference CIK changed')
        require(not ciks or {x['cik'] for x in ciks}=={current['cik']}, 'Original/current CIK conflict needs adjudication')
        registration_class=item['identity']['exchange_ticker_basis']=='Single registration-table header followed by class, ticker, exchange row'
        class_bound=registration_class or bool(bridge)
        gaps=[]
        if not ciks: gaps.append('No directly labelled CIK in this original document; current SEC reference remains a separate current-only link.')
        if not class_bound: gaps.append('No explicit class-to-listed-ticker binding identified in this original; a generic common-stock mention is not sufficient.')
        if vendor['vendor_identity_metadata'] is None: gaps.append('No dedicated local vendor issuer/exchange/instrument identity metadata in the inspected yahoo_identity directory.')
        else: gaps.append('Vendor identity metadata is an unreceipted local snapshot; current EQUITY/name/exchange compatibility does not certify historical class continuity.')
        gaps.append('Event-to-consumed-price-interval identity and later overlay routes need bounded review; no daily universal security master is required.')
        bounds=vendor['reported_observation_bounds']
        output.append(dict(binding_id=item['binding_id'], research_security_id=symbol,
            source_kind=item['kind'], original_publication_date=item['publication']['date'],
            original_path=original['path'], original_sha256=original['original_sha256'],
            original_receipt_path=original['receipt_path'], original_receipt_sha256=original['receipt_sha256'],
            original_actual_retrieved_at=original['retrieved_at'],
            original_issuer=item['identity']['issuer_name'], original_ticker_exchange=item['identity']['exchange_ticker'],
            original_security_class_literal=item['identity']['security_class_literal'], original_class_bound_to_ticker=class_bound,
            original_CIK_fields=ciks, current_CIK_reference=current,
            source_ticker=source_ticker, selected_provider_symbol=vendor_symbol,
            join=edge, frozen_route=routing, bridge=bridge,
            source_supported_scope='This dated document observation only' if not bridge else 'Explicit same-Class-A/CUSIP ticker transition effective '+bridge['effective_date'],
            source_use_window_is_security_epoch=False,
            original_date_inside_vendor_reported_bounds=within(item['publication']['date'],bounds['first'],bounds['last']),
            identity_evidence_gaps=gaps, historical_price_security_admitted=False, filing_gate_satisfied=False))
    # Inspect the tempting discarded legacy ticker explicitly, without loading its price data.
    discarded=src.obj(UNION+'sources/yahoo_identity/FB.json',PINS[UNION+'sources/yahoo_identity/FB.json'])
    rejected=dict(candidate_provider_symbol='FB',metadata=discarded,
        metadata_path=UNION+'sources/yahoo_identity/FB.json',metadata_sha256=src.hashes[UNION+'sources/yahoo_identity/FB.json'],
        assessment=edge_assessment('FB','FB',discarded),
        scope='Negative compatibility evidence from local metadata; no separate retrieval receipt. Frozen selected route is META independently of this rejection.')
    require(rejected['assessment']=='REJECT_VENDOR_INSTRUMENT_TYPE_CONFLICT','Expected explicit legacy FB conflict changed')
    issuer_summary=[]
    for symbol in sorted({x['research_security_id'] for x in output}):
        rows=[x for x in output if x['research_security_id']==symbol]
        issuer_summary.append(dict(research_security_id=symbol,
            selected_provider_symbol=rows[0]['selected_provider_symbol'],
            dated_observations=len(rows),
            first_original_date=min(x['original_publication_date'] for x in rows),
            last_original_date=max(x['original_publication_date'] for x in rows),
            original_CIK_fields_found=sum(bool(x['original_CIK_fields']) for x in rows),
            original_class_to_ticker_bindings=sum(x['original_class_bound_to_ticker'] for x in rows),
            source_to_vendor_request_route_verified=True,
            date_range_is_observation_span_not_certified_epoch=True,
            temporal_continuity_certified=False))
    src.recheck()
    summary=dict(bindings=len(output),research_securities=len({x['research_security_id'] for x in output}),
        provider_series=len(vendors),verified_opaque_vendor_chains=sum(x['opaque_byte_hash_chain_verified'] for x in vendors.values()),
        joins=dict(Counter(x['join'] for x in output)),
        original_documents_with_labelled_CIK=sum(bool(x['original_CIK_fields']) for x in output),
        original_documents_with_class_bound_to_ticker=sum(x['original_class_bound_to_ticker'] for x in output),
        dated_documents_in_reported_vendor_bounds=sum(x['original_date_inside_vendor_reported_bounds'] for x in output),
        provider_identity_metadata_present=sum(x['vendor_identity_metadata'] is not None for x in vendors.values()),
        exact_rejected_alias_candidates=1,historical_security_admissions=0,input_files_hashed=len(src.hashes))
    result=dict(schema_version=1,status='DATED_EVENT_TO_VENDOR_SOURCE_ROUTES_BOUND_WITH_EXPLICIT_IDENTITY_GAPS',
        created_utc=datetime.now(timezone.utc).isoformat(),summary=summary,issuer_summary=issuer_summary,
        bindings=output,vendor_series=vendors,rejected_candidates=[rejected],
        rules=['A direct edge requires original dated issuer/ticker/exchange evidence and an exact requested-symbol to opaque-source-hash transformation chain; it is evidence, not automatic class admission.',
            'A ticker mismatch requires an explicit source-supported same-security transition. Similar names, internal labels and current CIK are insufficient.',
            'A class or instrument conflict rejects the candidate. A generic common-stock phrase is retained but does not establish the traded class.',
            'Identity need only cover the actual consumed event and price interval; do not invent long epochs or require an unrelated all-history daily master.',
            'Existing file date bounds, source-use windows and current retrieval time are not evidence of historic identity validity.'],
        concrete_findings=['The 24 MSFT/NVDA annual observations in the fixed binding batch are earnings-release documents, not original 10-K cover pages.',
            'All fixed source observations now connect to a receipt-bound vendor request and frozen normalization/execution route; this resolves source lineage, not the stated identity gaps.',
            'The frozen research_equity_FB alias explicitly selects META. Its original unchanged-Class-A/CUSIP transition supports this ticker bridge. The current FB metadata identifies a different ETF and is rejected.',
            'Price bytes were hashed opaquely. No CSV value was decoded, no price transformation replayed, and no final extended/overlaid price panel was recertified.'],
        minimal_remaining=['For actual periodic filings, attach directly evidenced filing CIK and listed class/ticker/exchange to this vendor route, reusing original cover fields when present.',
            'Where vendor identity is absent or a class/alias/ADR/successor question arises, supply bounded vendor/reference evidence for that event and consumed price interval. Do not extrapolate dated observations into unbounded epochs.',
            'Keep independent filing-clock and complete relevant original/amendment chronology gates separate; this artifact changes neither gate nor the strategy design.'],
        synthetic_validation=dict(passed=True,checks=len(check_names),names=check_names),input_hashes=dict(sorted(src.hashes.items())),
        activity=dict(network_requests=0,numeric_price_values_decoded=0,financial_values_extracted=0,new_factors=0,new_orders=0,new_returns=0,strategy_registrations=0,existing_files_modified=0))
    if args.write:
        with src.path(OUT).open('x') as file:
            json.dump(result,file,indent=2,sort_keys=True,allow_nan=False);file.write('\n')
    print(json.dumps(dict(summary=summary,checks=len(check_names),output=OUT if args.write else None,
        output_sha256=hashlib.sha256(src.path(OUT).read_bytes()).hexdigest() if args.write else None),indent=2))


if __name__=='__main__':
    main()
