"""Metadata-only transition ledger. No price/volume/NPZ/result decoding."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/after248_volume_breadth_TRANSITION_LEDGER.json'
PREFLIGHT=ROOT/'work/after248_volume_breadth_source_preflight.json'
PREFLIGHT_SHA='5b189d6a52f7b05197081727809673dde12253218ff3e0a47e8846e60277f306'
PINS={
 'outputs/monthly_stability_monitor/inputs.py':'dd97fb4a9f7b2102d2c7f497825526a484628fa86ee9891312bfb5abad7c8ded',
 'outputs/index_union/price_units.py':'353e2f039d022a03cf90be06028d680b8dd389fad7bb216727cc2ecbda226291',
 'work/volume195_source_review.py':'1d7b0f6c8f20acf90e1453cf028d427e29a3ba5f2f514c3ed54b245be4bb1d64',
 'outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json':'8f0d33ef1494804f88034a8d84618b07cb0999c184418584c8592617e2a0415d',
}
META_NAMES={
 'work/after248_volume_breadth_source_preflight.json',
 'outputs/index_union/DATED_SYMBOL_IDENTITIES.json',
 'outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json',
 'outputs/index_union/REVIEWED_TERMINAL_EVENTS.json',
 'outputs/index_union/alias_resolved_inputs/MANIFEST.json',
 'outputs/monthly_2016_extension/prices/BUD.source.json',
 'outputs/monthly_2016_extension/prices/HSBC.source.json',
 'outputs/monthly_2016_extension/prices/NVS.source.json',
 'outputs/monthly_top35_extension/prices/HSBC.source.json',
 'outputs/monthly_top35_extension/prices/NVS.source.json',
}
CODE_NAMES={'outputs/index_union/consolidate_security_aliases.py','outputs/index_union/consolidate_issuer_classes.py','outputs/index_union/normalize_new_prices.py','outputs/index_union/assemble_alias_resolved_inputs.py'}
def sha(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p
 return hashlib.sha256(p.read_bytes()).hexdigest()
def meta(name):
 if name not in META_NAMES:raise ValueError('Not allowlisted metadata')
 return json.loads((ROOT/name).read_text())
def main():
 if OUT.exists():raise ValueError('Write-once ledger exists')
 if sha(PREFLIGHT)!=PREFLIGHT_SHA:raise ValueError('Changed source preflight')
 for p,h in PINS.items():
  if sha(p)!=h:raise ValueError('Changed source pin '+p)
 pre=meta('work/after248_volume_breadth_source_preflight.json');events=pre['calendar']['events'];chains=pre['source_lineage']['chains'];names=set(chains)
 aliases=meta('outputs/index_union/DATED_SYMBOL_IDENTITIES.json');evidence=meta('outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json')
 terminal=meta('outputs/index_union/REVIEWED_TERMINAL_EVENTS.json')
 manifest=meta('outputs/index_union/alias_resolved_inputs/MANIFEST.json')
 # Each row below reports an evidenced metadata boundary, not a guessed volume jump.
 records=[]
 for b in pre['listing']['bounds']:
  records.append(dict(id='listing_'+b['symbol'],category='security_identity_not_before',symbols=[b['symbol']],effective_date=b['not_before'],date_semantics='Conservative listed-security history lower bound, not necessarily first actual trade.',
   evidence_known_date=b['available_from'],available_from=b['available_from'],known_at_utc=None,visibility_basis='Inherited admitted source-evidence next-calendar-day rule; copied unchanged.',
   affected_fields=['adj','volume'],before='No admitted earlier history for this named US security.',after='Identity bound alone does not admit any numeric source.',unit_change='No numeric conversion asserted.',
   resolution='QUERY_VISIBLE_IDENTITY_GATE',automatic_action='Positive consumed source observation before visible bound is a hard identity contradiction; ordinary source absence is whole-event fallback. Check315-volume separately from253-price.',
   unresolved_unit_transition=False,known_unresolved_fallback_eligible=False,metadata_source='work/after248_volume_breadth_source_preflight.json',metadata_source_sha256=PREFLIGHT_SHA,source_evidence_id=b['evidence_id'],issuer_key=b['issuer_key']))
 records.append(dict(id='chl_ptr_terminal_mask_20210101',category='availability_cutoff_not_unit_transition',symbols=['CHL','PTR'],effective_date='2021-01-01',date_semantics='Literal loader rule: retain dates strictly before2021-01-01, mask all fields on/after.',
   evidence_known_date=None,available_from=None,known_at_utc=None,visibility_basis='Archived code rule; historical public known_at is not established.',affected_fields=['adj','open','close','high','low','volume'],
   before='Tiingo adjClose with raw open/close/OHLC and volume, not adjVolume.',after='All fields missing by explicit mask; no replacement provider or unit inferred.',unit_change='No unit transition claimed at cutoff.',resolution='KNOWN_LOADER_AVAILABILITY_RULE',
   automatic_action='Do not forward fill. Required masked cells will fail ordinary complete-source support after actual scan; this row is not an additional unit-break fallback trigger.',unresolved_unit_transition=False,known_unresolved_fallback_eligible=False,
   metadata_source='outputs/monthly_stability_monitor/inputs.py',metadata_source_sha256=PINS['outputs/monthly_stability_monitor/inputs.py'],locator='load_panel CHL/PTR override and index<2021-01-01 mask',
   applicable_inventory_symbols=sorted(names & {'CHL','PTR'})))
 fb=next(z for z in evidence['aliases'] if z['old']=='FB' and z['new']=='META')
 fbrows=[z for z in aliases['rows'] if z['security_id']=='research_equity_FB']
 if len(fbrows)!=2 or any(z['evidence_sha256']!=PINS['outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json'] for z in fbrows):raise ValueError('FB alias evidence mismatch')
 records.append(dict(id='fb_meta_same_security_20220609',category='resolved_ticker_rename_not_provider_splice',symbols=['research_equity_FB'],effective_date=fb['effective'],date_semantics='Published effective ticker rename.',evidence_known_date=fb['published'],available_from=None,known_at_utc=None,
   visibility_basis='Public date retained; exact intraday release time unknown. Not used as a numeric factor or automatic fallback.',affected_fields=['symbol_label','membership_epoch'],before='FB ticker',after='META ticker, same reviewed security_id.',unit_change='No documented share/volume-unit change; do not invent one from ticker rename.',
   resolution='RETROSPECTIVE_SAME_EQUITY_IDENTITY_RESOLVED_PIT_NOT_CERTIFIED',automatic_action='No unit-transition fallback. Frozen consolidator chooses one price/volume provider for entire identity and merges membership epochs; it does not splice volume providers on rename date.',unresolved_unit_transition=False,known_unresolved_fallback_eligible=False,
   metadata_source='outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json',metadata_source_sha256=PINS['outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json'],source_url=fb['source'],identity_rows=fbrows))
 cm=next(z for z in evidence['aliases'] if z['old']=='CMCSK' and z['new']=='CMCSA')
 records.append(dict(id='comcast_class_conversion_20151214',category='documented_class_event_not_proven_signal_unit_break',symbols=['CMCSA'],effective_date=cm['effective'],date_semantics='Evidence metadata distinguishes conversion_at from next effective trading label date.',conversion_at=cm['conversion_at'],evidence_known_date=cm['published'],available_from=None,known_at_utc=None,
   visibility_basis='Public publication date only; intraday availability unknown.',affected_fields=['share_class_identity'],before='CMCSK class',after='One-for-one conversion into CMCSA class.',unit_change='Metadata says one-for-one class conversion. No dated change to consumed CMCSA stored Volume field/basis is established.',
   resolution='CORPORATE_EVENT_NOT_AN_ADMITTED_SOURCE_DISCONTINUITY',automatic_action='Informational only. Do not treat every corporate action or economic activity change as a source-unit break. A new explicit source/basis contradiction would need evidence and a prospective ledger revision.',unresolved_unit_transition=False,known_unresolved_fallback_eligible=False,
   metadata_source='outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json',metadata_source_sha256=PINS['outputs/index_union/SECURITY_ALIAS_EVIDENCE_02.json'],source_url=cm['source']))
 undated=[]
 for symbol in ('BUD','HSBC','NVS'):
  chain=chains[symbol];early=next(z for z in chain if z['operation']=='leading_adj_gap_before_first_overlap')
  early_receipt='outputs/monthly_2016_extension/prices/'+symbol+'.source.json';receipt=meta(early_receipt)
  if receipt['sha256']!=early['sha256']:raise ValueError('Early source receipt mismatch')
  older=[z for z in chain if z['operation']=='add_security'];receipts=[dict(path=early_receipt,sha256=sha(early_receipt),retrieved_utc=receipt['retrieved_utc'],data_file_sha256=receipt['sha256'])]
  if older:
   rpath='outputs/monthly_top35_extension/prices/'+symbol+'.source.json';r=meta(rpath)
   if r['sha256']!=older[0]['sha256']:raise ValueError('Overlay source receipt mismatch')
   receipts.append(dict(path=rpath,sha256=sha(rpath),retrieved_utc=r['retrieved_utc'],reported_first=r.get('first'),reported_first_role='Receipt first row is not proof of first nonmissing common adjusted-price overlap.',data_file_sha256=r['sha256']))
  undated.append(dict(id='undated_early_bridge_'+symbol,symbols=[symbol],category='known_loader_splice_rule_effective_date_unknown',effective_date=None,effective_date_status='UNKNOWN_NOT_INFERRED_FROM_REQUEST_START_OR_RECEIPT_FIRST',evidence_known_date=None,available_from=None,known_at_utc=None,
    retrospective_archive_receipts=receipts,affected_fields=['adj','open','close','high','low','volume'],operation='Use early data only in leading adj gaps before first common valid adjusted-price overlap. Multiply early adj by overlap ratio; copy early Volume without that ratio.',
    first_overlap_rule="first=a.index[a.notna() & z['Adj Close'].notna()][0] when any overlap; otherwise end+1",unit_continuity='UNKNOWN; two Yahoo vintages do not by themselves certify identical within-window volume conventions.',
    automatic_action='Cannot compute window intersection or automatic known-date fallback from present metadata. Do not substitute first requested/first reported row for actual overlap. Keep date unknown under declared-assumption policy, or admit a separately reviewed source-only bridge-date receipt before registration.',known_unresolved_fallback_eligible=False,
    blocker='No certified effective date and no complete basis-consistency evidence; current metadata cannot implement exhaustive bridge-intersection gate.',source_code='outputs/monthly_stability_monitor/inputs.py',source_code_sha256=PINS['outputs/monthly_stability_monitor/inputs.py'],locator='early2016 extension first common overlap and leading-gap assignment'))
 # Metadata-only calendar intersections. Never use them to assert numeric support.
 for r in records:
  r['window_intersections']=[]
  for e in events:
   if e['initial_partial']:continue
   syms=sorted(set(e['symbols']) & set(r['symbols']))
   if not syms:continue
   effective=r['effective_date'];p=e['price_start']<=effective<=e['price_end'];v=e['volume_start']<=effective<=e['volume_end']
   if p or v:
    known=r['evidence_known_date'];r['window_intersections'].append(dict(query_index=e['query_index'],date=e['date'],symbols=syms,price_window_contains_date=p,volume_window_contains_date=v,
      evidence_published_strictly_before_query=None if known is None else known<e['date'],evidence_same_date_clock_unknown=None if known is None else known==e['date'],numeric_support='UNKNOWN_NOT_SCANNED'))
 fallback_records=[r['id'] for r in records if r['known_unresolved_fallback_eligible']]
 unrelated_terminal=[z for z in terminal if z.get('symbol') in names]
 if unrelated_terminal:raise ValueError('New relevant terminal metadata requires explicit review')
 pins={name:sha(name) for name in sorted(set(PINS)|META_NAMES|CODE_NAMES)};pins[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
 report=dict(schema='after248_volume_breadth_transition_ledger_v1',created_utc=datetime.now(timezone.utc).isoformat(),
   status='BOUNDED_METADATA_LEDGER_DECLARED_ASSUMPTION_ONLY_NOT_CONTINUITY_CERTIFICATION',research_policy='Root selected archived stored-volume declared-assumption research; strictPIT=false and continuity UNKNOWN.',strict_pit=False,source_admitted=False,registered=False,
   source_preflight=dict(path=str(PREFLIGHT.relative_to(ROOT)),sha256=PREFLIGHT_SHA,raw64_hash_closure='Already verified there; not rerun here'),
   records=records,undated_known_loader_rules=undated,automatic_known_unresolved_transition_fallback_ids=fallback_records,
   automatic_fallback_scope='No sufficiently evidenced dated unresolved consumed-field/unit/provider discontinuity was found in this bounded metadata subset. Empty list does not certify absence. Missing original coverage and undated early bridges remain UNKNOWN.',
   gate_contract=dict(dated_unresolved='A future properly evidenced entry requires affected security/field, effective date, query-visible evidence time or explicitly retrospective data-integrity basis, and unresolved source-unit/identity/provider change. If the consumed315-volume or253-price window intersects it, both experimental arms fall back for the entire49 event before ranking.',
     identity='Keep known visible SPCX/SKHY not-before gates separate: positive consumed source before bound hardfails. Their mere calendar intersection is not a source numeric contradiction.',
     missing='Ordinary missing remains all49 whole-event fallback. Scan all members before fallback so missing cannot conceal a hard contradiction.',
     unknown='Undated/uncertified continuity is an explicit research assumption, not imputed continuity and not a silently fabricated date. Root must bind this exact limited ledger before actual feature computation; no outcome-dependent additions/removals.',
     disclosure='Field/source code, archived retrieval time, public evidence date and economic effective date are distinct. Archive acquisition in2026 does not certify availability at an earlier query.'),
   static_source_rules_not_calendar_transitions=[
     'Alias consolidation chooses one provider for the whole identity by retrospective coverage; it does not establish a dated provider switch.',
     'Manifest whole-column overwrites are loader precedence, not dated within-series unit changes. Their source basis can remain UNKNOWN.',
     'raw_to_split_adjusted multiplies Volume by subsequent split multiplier where documented. No split calendar or ratio was decoded here; action completeness is UNKNOWN.',
     'A ticker rename or corporate reclassification alone does not prove the consumed security Volume field changes units.',
     'Date requested, first reported row, CSV retrieval time and raw-price overlap parity must not be substituted for certified volume-splice effective date or unit continuity.'],
   inspected_metadata_scope=dict(inventory_symbols=len(names),query_events=len(events),known_listing_records=2,known_loader_mask_records=1,resolved_alias_records=1,corporate_event_only_records=1,undated_early_bridge_records=3,relevant_reviewed_terminal_event_matches=0,complete_corporate_action_or_ADR_ledger=False),
   blockers_and_limits=[
     'Cannot implement exhaustive automatic gating for BUD/HSBC/NVS early-splice windows: first common valid overlap depends on uninspected source values and is not proved by available metadata.',
     'No complete point-in-time split/ADR/provider transition ledger exists in the inspected scope; empty automatic list is not an admission certificate.',
     'True positivity/coverage and prelisting contamination remain registered source checks; no current numeric observation was examined.',
     'If the final rule requires every possible unit discontinuity to be dated and quarantined, current evidence is insufficient. Under root approved declared-assumption policy, only evidenced known gates can be enforced and UNKNOWN must remain in reporting.',
     'This ledger must not authorize post-outcome source repairs, turnover claims or strictPIT promotion.'],
   evidence_sha256=pins,work_performed=dict(metadata_and_code_only=True,price_volume_rows_decoded=0,npz_decoded=0,features_computed=0,orders_computed=0,returns_computed=0,network_requests=0,new_trials=0,old_files_modified=0,numeric_sources_rehashed=False))
 with OUT.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),dated_metadata_records=len(records),undated_rules=len(undated),automatic_fallback_rules=len(fallback_records),numeric_data_read=False)))
if __name__=='__main__':main()
