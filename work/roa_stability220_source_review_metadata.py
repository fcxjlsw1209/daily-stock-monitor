"""One-shot metadata/hash-only round220 source review. Run via run_local.sh."""
from pathlib import Path
from datetime import datetime, timezone, date
from collections import Counter
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'work/roa_stability220_SOURCE_REVIEW.json'
CONTEXT = ROOT/'work/monthly_roa_stability220_context.json'
CONTEXT_SHA = '673aa44502d61490fef26a16a426cefa94469cdae9f059948bb471458e7370e5'
PACKAGE = ROOT/'outputs/monthly_roa_stability_source220_v1'
MANIFEST_SHA = '328b19f5e30406dcec2094b0302f83d97a7c303fae5fa9ca0f8500ee861dffac'
FIELDS = ('annual_net_income','opening_total_assets','closing_total_assets')
pins = {}

def check(ok, message):
    if not ok: raise ValueError(message)

def path(name):
    p = Path(name)
    p = p if p.is_absolute() else ROOT/p
    check(p.is_relative_to(ROOT), 'Nonlocal path: '+str(p))
    check('CloudStorage' not in str(p), 'Cloud path forbidden')
    return p

def sha(p):
    result = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): result.update(block)
    return result.hexdigest()

def bind(name, expected=None):
    p = path(name); digest = sha(p)
    check(expected is None or digest == expected, 'Digest mismatch '+str(p))
    check(str(p) not in pins or pins[str(p)] == digest, 'Pin conflict')
    pins[str(p)] = digest
    return digest

def read(name, expected=None):
    p = path(name); bind(p, expected)
    return json.loads(p.read_text())

def iso(value):
    check(type(value) is str and date.fromisoformat(value).isoformat() == value, 'Canonical ISO date')

check(not OUT.exists(), 'Write-once review already exists')
context = read(CONTEXT, CONTEXT_SHA)
evidence = {}
for name, expected in context['evidence_sha256'].items():
    evidence[name] = bind(name, expected)
check(context['dose'] == .01 and context['source_policy']['strict_pit'] is False, 'Context contract')
manifest = read(PACKAGE/'MANIFEST.json', MANIFEST_SHA)
check(manifest['source_only'] is True and manifest['write_once'] is True, 'Immutable source package')
check(set(manifest['files']) == {'POLICY.json','DATASET.json','REVIEW_EVIDENCE.json','CLOCK_REVIEW.json','VALIDATION.json'}, 'Exact package members')
package = {name:read(PACKAGE/name, digest) for name,digest in manifest['files'].items()}
check(len(manifest['input_sha256']) == 207, 'Source input closure')
for name, expected in manifest['input_sha256'].items(): bind(name, expected)
dataset, validation, clock, policy = [package[x] for x in ('DATASET.json','VALIDATION.json','CLOCK_REVIEW.json','POLICY.json')]
check(dataset['source_pins'] == manifest['input_sha256'], 'Dataset source closure differs')
check(validation['passed'] is True and validation['source_only'] is True, 'Source validation incomplete')
for flag in ('raw_digest_bindings','own_report_three_field_bindings','original_field_admissions_preserved','exact_identity_no_ticker_join','reported_NI_and_assets_checked_independently','admitted_edge_exact_scaled_source_equality','event_versions_not_deduplicated_by_raw_sha','complete_supplied_catalog_preserved','no_first_public_PIT_certification'):
    check(validation['checks'][flag] is True, 'Missing source validation '+flag)
for key in ('ROA_calculations','means','standard_deviations','factors','ranks','monthly_support_matches','orders','returns','network_requests','source_admission_updates_outside_new_directory','observer_changes'):
    check(type(validation['checks'][key]) is int and validation['checks'][key] == 0, 'Source computation boundary '+key)
for key in ('first_public_version_verified','complete_correction_history_verified','certified_daily_security_master'):
    check(policy[key] is False and dataset[key] is False, 'Source limitations changed')
check(clock['counts'] == {'verified_calendar_rule_only':161,'outside_pinned_calendar':30}, 'Source clock counts')
check(clock['inherited_clock_review_absent_count'] == clock['absent_clock_independently_verified_count'] == 44, 'Independent clock review coverage')
check(clock['actual_signal_queries'] == 0 and clock['no_outside_calendar_clock_invented'] is True, 'Source clocks did queries')
bind(clock['calendar_ref'],clock['calendar_sha256'])
catalog=dataset['catalog']; reports=dataset['reports']; chains=dataset['chains']; edges=dataset['edges']
check(len(catalog)==191 and len({x['event_id'] for x in catalog})==191,'Complete distinct catalog')
check(Counter(x['original_field_state'] for x in catalog)=={'admitted':91,'held':38,'unprocessed':62},'Catalog admission states')
check(len(reports)==91 and all(x['inherited_single_report_fields_admitted'] is True for x in reports),'Original single-report admissions')
check(len(edges)==70 and sum(x['conditional_allowed'] for x in edges)==51,'Conditional edges')
check(len(chains)==51 and sum(x['conditional_allowed'] for x in chains)==40,'Conditional chains')
reportmap={x['report_version_id']:x for x in reports}
check(len(reportmap)==91 and len({x['catalog_version_id'] for x in catalog})==191,'Distinct versions')
ni_scope=set(); missing_cell_date=0; explicit_cell_date=0; currency_iso=set(); currency_literal=set()
for report in reports:
    check(set(report['fields'])==set(FIELDS),'Three own-report fields')
    check(report['corrected_original_values_substituted'] is False,'Original field substitution')
    for field in report['fields'].values():
        cell=field['source_cell']; iso(field['as_of'])
        check(bool(field['original_field_path']) and bool(cell['locator']),'Original source locator')
        check(cell['source']['path']==report['source']['path'] and cell['source']['sha256']==report['source']['sha256'],'Original source field binding')
        check(type(cell['scale']) is str and type(cell['value_decimal']) is str,'Original decimal field schema')
        if 'as_of' in cell:
            explicit_cell_date+=1; check(cell['as_of']==field['as_of'],'Cell date mismatch')
        else: missing_cell_date+=1
        currency_iso.add(cell.get('currency_iso'));currency_literal.add(cell['currency_literal'])
    ni_scope.add(report['fields']['annual_net_income']['ni_attribution'])
check(ni_scope=={'reported_consolidated_total_NI_before_NCI_deduction'},'NI attribution mismatch')
check(currency_iso=={None} and currency_literal=={'$'},'Unverified currency scope changed')
for chain in chains:
    refs=chain['nine_required_fields']
    check(len(refs)==9 and len({(r['report_version_id'],r['field']) for r in refs})==9,'Nine distinct bindings')
    for ref in refs:
        report=reportmap[ref['report_version_id']]
        check(ref['report_version_id'] in chain['report_version_ids'] and ref['event_id']==report['event_id'] and ref['source_sha256']==report['source']['sha256'] and ref['original_field_path']==report['fields'][ref['field']]['original_field_path'],'Nine field reference mismatch')
for edge in edges:
    refs=edge['evidence_refs']
    check(len(refs)==2,'Original two-source edge package')
    for identifier in (edge['earlier_report_version_id'],edge['later_report_version_id']):
        report=reportmap[identifier]; selected=[r for r in refs if r['report_version_id']==identifier]
        check(len(selected)==1,'Missing original edge evidence')
        ref=selected[0];iso(ref['assumed_available_from'])
        check(ref['event_id']==report['event_id'] and ref['source']['path']==report['source']['path'] and ref['source']['sha256']==report['source']['sha256'] and ref['assumed_available_from']==report['known_at'],'Edge source/version/clock mismatch')
    check(edge['known_at']==max(r['assumed_available_from'] for r in refs),'Edge evidence clock')
local=ROOT/'outputs/monthly_local_linear70'
plan=read(local/'PLAN.json','0016175aee1df43b734010d80bd543b881c57036a7f8c6ab73428340203afc96')
raw=plan['raw_source_hashes']
prior=read('work/risk_consensus219_SOURCE_REVIEW.json')
check(len(raw)==64 and raw==prior['raw_source_hashes'],'Original64 source map')
original=read(local/'INPUT_HASHES.json','5ac2b639fa8c944688787eba47ea1b420bfbe9c6e1e051a3d827d8b2b32b264d')
for name, expected in raw.items():
    check(original.get(name)==expected,'Raw source not original registered input')
    bind(name,expected)
meta=read(local/'META.json','3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4')
check(len(meta['dates'])==len(meta['ends'])==len(meta['symbols'])==122,'Original metadata events')
check(meta['dates']==sorted(set(meta['dates'])) and meta['dates'][0]=='2016-08-17' and meta['dates'][-1]=='2026-08-31' and meta['ends'][-1]=='2026-09-24','Original metadata dates')
check(all(len(r)==49 and len(set(r))==49 for r in meta['symbols']) and len(meta['etfs'])==len(set(meta['etfs']))==17,'Original candidate axes')
for value in meta['dates']+meta['ends']: iso(value)
for name in ('CACHE_HASHES.json','INCUMBENT_POLICY.json'):
    read(local/name)
# Pin static integration code, but do not import any module or decode data arrays.
static_files=('work/monthly_roa_stability220_kernel.py','work/monthly_roa_stability220_scoring.py','work/roa_stability_preflight_math.py','work/after219_roa_stability_source_admission_v1.py','outputs/monthly_local_linear70/study.py','outputs/monthly_core_timing70_research/core_adapter.py','outputs/monthly_path70_research/incumbent_adapter.py','work/monthly_stock_comp149_runner.py')
for name in static_files: evidence[name]=bind(name)
evidence[str(Path(__file__).relative_to(ROOT))]=bind(Path(__file__))
flags=('source_only','source_admission_metadata_only','raw_source_hashes_verified','raw_source_map_matches_original','financial_package_hashes_verified','complete_supplied_catalog_preserved','original_single_report_admissions_preserved','conditional_chain_policy_reviewed','nine_field_source_bindings_reviewed','source_clock_assumptions_explicit','integration_source_verified','metadata_axes_verified')
review=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,context_sha256=CONTEXT_SHA,scope='Streaming local SHA and JSON/source-code metadata only. No source-selection invocation, strategy module imports, numeric arrays, financial ratios, ranks, orders or returns.',input_sha256=dict(sorted(pins.items())),evidence_sha256=dict(sorted(evidence.items())),raw_source_hashes=raw,financial_source_manifest_sha256=MANIFEST_SHA,financial_source_dataset_sha256=manifest['files']['DATASET.json'],source_counts=validation['counts'],metadata=dict(events=122,width=49,etfs=17,first_decision=meta['dates'][0],last_decision=meta['dates'][-1],last_outcome_end=meta['ends'][-1]),field_schema=dict(original_field_names=list(FIELDS),ni_attribution=list(ni_scope),cell_as_of_absent=missing_cell_date,cell_as_of_explicit=explicit_cell_date,wrapper_as_of_from_original_period_and_source_locator=True,currency_literal='$',currency_iso=None,currency_ISO_certified=False),static_integration_review=dict(latest_catalog_before_admission=True,fiscal_anchor_conflicts_block=True,held_unknown_unprocessed_latest_no_fallback=True,exact_identity_join=True,original_report_reference_per_edge_required=True,extra_evidence_dates_canonical_and_in_clock_max=True,own_report_original_fields_and_positive_assets=True,negative_NI_not_excluded=True,all_nine_source_refs_checked=True,full_core_support_and_universe_preserved=True,fixed_bounded_delta=.01,only_eligible_supported_stocks=True,no_new_cross_sectional_ranking=True,one_original_episode_per_arm=True,static_only_not_actual_order_parity=True),limitations=['Conditional as-reported own-year NI and average assets; no constant business perimeter claim; PYPL2015 combined-and-consolidated explicitly allowed.','Document-date plus following pinned session is an availability assumption, not certified first-publication PIT.','40 reviewed chains are source policy units, not actual signal eligibility; monthly support is uncomputed.','191 entries are the complete supplied15-issuer catalog, not a worldwide complete report/correction history.','Source currency literals are dollar with explicit scale; ISO is not independently certified.','Original execution inputs are hash-verified, not freshly reconstructed PIT membership or price evidence.'],actual_new_features_computed=False,actual_new_orders_computed=False,actual_new_returns_computed=False,actual_monthly_support_computed=False,npz_decoded=False,strict_pit=False,first_public_version_verified=False,complete_correction_history_verified=False,no_observer_changes=True,source_byte_hashes_only=True)
review.update({key:True for key in flags})
with OUT.open('x') as stream: json.dump(review,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),verified_files=len(pins),raw_source_files=len(raw),source_inputs=len(manifest['input_sha256']),source_only=True,monthly_queries=0)))
