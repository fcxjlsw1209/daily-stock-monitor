"""Round225 metadata/source-byte audit only; never decode numeric arrays.

Run through run_local.sh. Old209 complete-path evidence is inspected, not rerun.
New225 cohort, labels, forecasts and outcomes remain gated by its future PLAN.
"""
from pathlib import Path
from datetime import date, datetime, timezone
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT/'work/monthly_relative_duration225_context.json'
CONTEXT_SHA = '0392086957d2cd7a843811de6971cf0544788cdb484aa2fdcd7f74b3b195d5d8'
COVERAGE = ROOT/'work/after224_relative_duration_coverage.json'
OUTPUT = ROOT/'work/relative_duration225_SOURCE_REVIEW.json'
PRIOR = ROOT/'outputs/monthly_relative_path209'
SOURCE209 = ROOT/'work/relative_path209_SOURCE_REVIEW.json'
SOURCE209_SHA = '976f4b210176e3a4173bddd28e17004a951b0c5c9f0735929df180d0d6da177e'
PLAN155 = ROOT/'outputs/monthly_endpoint155/PLAN.json'
PLAN155_SHA = 'f10890b99c5b27d176eaa9467152358a2c3bd5b845c0bb4c9aa6865512f29201'
LOCAL = ROOT/'outputs/monthly_local_linear70'
CALENDAR = ROOT/'outputs/monthly_paired_target143/CALENDAR.json'
CALENDAR_SHA = '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622'
EXTERNAL = {'/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9':'6b24be8ba00c4abc8b3e7e2620bd710e2cf7a4312d00c147f8f05fbb6f3d297a'}
PINS = {}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def path(p):
    p = Path(p)
    p = p if p.is_absolute() else ROOT/p
    require((p.is_relative_to(ROOT) and p.resolve().is_relative_to(ROOT)) or str(p) in EXTERNAL,
            'Nonlocal source '+str(p))
    require('CloudStorage' not in str(p), 'Cloud path')
    return p

def sha(p):
    p = path(p)
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def bind(p, expected=None):
    p = path(p)
    current = sha(p)
    require(expected is None or current == expected, 'Changed source '+str(p))
    require(str(p) not in PINS or PINS[str(p)] == current, 'Conflicting source '+str(p))
    PINS[str(p)] = current
    return current

def read(p):
    return json.loads(path(p).read_text())

def bind_map(mapping):
    for p, expected in mapping.items():
        bind(p, expected)

def write_once(p, obj):
    require(not p.exists(), 'Write-once output already exists '+str(p))
    with p.open('x') as stream:
        stream.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')

def stamp():
    return datetime.now(timezone.utc).isoformat()

def definitions():
    # These are bounded source-definition findings, not performance selection.
    rows = [
      dict(round=163, path='work/run163_kernel.py', function='_terminal_run',
        definition='211 prices t-252..t-42; ten disjoint21-session endpoint signs; signed terminal equal-sign run length/10, zero last sign gives0.',
        distinction='Descriptive past absolute block run ending t-42; not future whole-month cumulative-SPY-relative daily run nor learned target.'),
      dict(round=166, path='work/followthrough166_kernel.py', function='_paired_correlations',
        definition='212 adjusted prices t-253..t-42 and231 volume values; 210 lagged daily logreturn pairs. Candidate volume-return interaction partial correlation with next return after removing current return/activity; control lag1 daily-return correlation.',
        distinction='Past local lag correlation/volume interaction; neither consecutive cumulative-relative state duration nor next-month target.'),
      dict(round=175, path='work/monthly_transition175_kernel.py', function='measure',
        definition='1261 prices t-1302..t-42;60 disjoint21-session blocks,59 prior sign-to-next-block logreturn transitions. Current last-block sign selects matching transitions; minimum10, prior mass20 toward unconditional59-payoff mean.',
        distinction='Already matured own-history conditional mean payoff, not longest future targetmonth relative underwater run or fraction of lagging sessions.'),
      dict(round=191, path='work/monthly_relative_severity191_model.py', function='relative_events/build_endpoint_targets',
        definition='Next naturalmonth terminal relative severity max(0,1-asset_gross/SPY_gross) versus binary I[relative_gross<1], exact binary64 rational endpoints.',
        distinction='Terminal depth/incidence ignores ordering and duration inside targetmonth; both225 targets are new.'),
      dict(round=197, path='work/monthly_recovery_durability197_kernel.py', function='measure',
        definition='253 prices t-294..t-42; negative daily anchors1..210, first recovery within21 sessions and following21-session all-above-anchor durability. Candidate durable/negative anchors; control recovered/negative anchors.',
        distinction='Past absolute recovery after local drawdowns, not signalanchored cumulative SPY-relative duration target.'),
      dict(round=209, path='work/monthly_relative_path209_model.py', function='relative_path/build_endpoint_targets',
        definition='Full next-naturalmonth stock/SPY path; candidate average max(0,1-relative_wealth_d) across target sessions, control terminal relative shortfall; rational deficits until final rounding.',
        distinction='Already combines depth and time, so path-risk family is not new.225 removes depth and isolates temporal clustering via longest run versus same-path total lag incidence.'),
      dict(round=217, path='work/monthly_persistence217_model.py', function='build_forecasts',
        definition='Exact previous formal calendar month/sameID/originalU and cached availability; preserve local-minus-pooled increment if current/prior signs agree, otherwise pooled; matchedcontrol halfincrement on same support.',
        distinction='Two-month existing forecast-sign confirmation, not a new whole futuremonth relative-price-path target.')]
    for item in rows:
        item['sha256'] = bind(item['path'])
    return rows

def coverage(context):
    bind_map(context['evidence_sha256'])
    rows = definitions()
    old = read('work/after208_path_shortfall_coverage.json')
    unknown = list(old['unknown_definitions'])
    implementation = bind(__file__)
    report = dict(schema='relative_duration225_bounded_coverage_v1', created_utc=stamp(),
        purpose='Prospective executable-definition comparison before new225 labels/fits/orders/returns.',
        context_sha256=CONTEXT_SHA, evidence_sha256=context['evidence_sha256'],
        reviewed_definitions=rows, family_novelty=False,
        exact_historical_overlap_status='No exact longest-run/total-incidence pair found in this bounded reviewed set; unlocated older numerical definitions remain UNKNOWN.',
        unknown_definitions=[dict(name=name,status='UNKNOWN',evidence='Inherited unlocated-definition list in pinned after208_path_shortfall_coverage.json; no exhaustive recovery claimed.') for name in unknown],
        contrast=dict(candidate='Longest uninterrupted run of cumulative stock-under-SPY states during complete next naturalmonth, divided by official target-session count.',
            matched_control='Total count of the same cumulative-relative lag states divided by the identical session count.',
            economic_mechanism='At equal lag-session incidence, clustered continuous relative weakness differs from interrupted recoveries; this may distinguish fragile leaders without changing the dominant momentum core.',
            counterexample='111000 and101010 share incidence3/6 and terminal0, while maximumrun is3/6 versus1/6.',
            nonclaims=['Not daily excess-return sign runs.','Not an untouched research family or independent holdout.','Not a guarantee of better nextmonth return.','No225 target equals the209 depth-average or terminal target, so no oldcontrol prediction parity.']),
        source_reuse='209/155 source closure and complete-path admission only; future225 preparation must compare dates/ends/symbols/base_good/good exactly with209.',
        parameter_search=False, actual_new_labels=False, actual_new_features=False,
        actual_new_support_computed=False, actual_model_fits_computed=False, actual_orders_computed=False,
        actual_returns_computed=False, npz_decoded=False, strategy_registered=False,
        strict_point_in_time=False, observers_modified=False,
        source_sha256=dict(sorted(PINS.items())), implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation))
    write_once(COVERAGE, report)
    return report

def day(s):
    require(isinstance(s,str) and date.fromisoformat(s).isoformat()==s, 'Canonical ISO date required')
    return s

def next_month(s):
    y,m = int(s[:4]),int(s[5:7])
    return '%04d-%02d'%((y+1,1) if m==12 else (y,m+1))

def axes(meta, expected_months, width):
    require(all(isinstance(meta.get(k),list) and len(meta[k])==expected_months for k in ('dates','ends','symbols')), 'Metadata month axes')
    identities = []
    for i,(s,e,ids) in enumerate(zip(meta['dates'],meta['ends'],meta['symbols'])):
        day(s);day(e)
        require(e>s and (i==0 or s>meta['dates'][i-1]), 'Metadata time ordering')
        require(isinstance(ids,list) and len(ids)==width and len(set(ids))==width and
            all(isinstance(x,str) and x and x.strip()==x for x in ids), 'Metadata security IDs')
        identities.extend((s,x,e) for x in ids)
    require(len(set(identities))==len(identities), 'Duplicate metadata identity')
    return identities

def source(context):
    bind_map(context['evidence_sha256'])
    bind(COVERAGE); cov = read(COVERAGE)
    require(cov['context_sha256']==CONTEXT_SHA and cov['npz_decoded'] is False, 'Coverage binding')
    bind_map(cov['source_sha256'])
    bind(SOURCE209,SOURCE209_SHA); oldreview=read(SOURCE209)
    require(oldreview['passed'] is True and oldreview['source_only'] is True, '209 source receipt')
    bind_map(oldreview['source_sha256'])
    bind(PLAN155,PLAN155_SHA); endpoint=read(PLAN155)
    require(endpoint['round']==155 and endpoint['status']=='REGISTERED','155 registered source')
    bind_map(endpoint['hashes'])
    for section in ('raw_source_hashes','independent_prepare_source_hashes','independent_outcome_source_hashes'):
        require(all(endpoint['hashes'].get(p)==h for p,h in endpoint[section].items()), '155 incomplete closure '+section)
    oldplan=read(PRIOR/'PLAN.json')
    bind_map(oldplan['input_sha256'])
    original=read(LOCAL/'PLAN.json')
    raw=original['raw_source_hashes']
    require(len(raw)==64 and raw==endpoint['raw_source_hashes']==oldplan['raw_source_hashes']==oldreview['raw_source_hashes'], 'Original raw64 maps differ')
    bind_map(raw)
    required = dict(meta='outputs/monthly_local_linear70/META.json',policy='outputs/monthly_local_linear70/INCUMBENT_POLICY.json',
        original_orders='outputs/monthly_local_linear70/local_linear_ORDERS.json',prices='outputs/monthly_2016_extension/PRICES.npz',
        train_inputs='outputs/monthly_target_definition70/TRAIN_INPUTS.npz',training_meta='outputs/monthly_earnings_factor_research/META.json',
        calendar='outputs/monthly_paired_target143/CALENDAR.json')
    for key,p in required.items():
        ref=endpoint['inputs'][key]
        require(path(ref['path'])==ROOT/p and endpoint['hashes'].get(str(ROOT/p))==ref['sha256'], 'Wrong155 source wiring '+key)
        require(ref==oldplan['endpoint_source_binding']['inputs'][key], '209 source wiring differs '+key)
    require(endpoint['calendar']==original['calendar']==oldplan['endpoint_source_binding']['calendar'], 'Calendar provenance ancestry')
    prepared=read(PRIOR/'prepared/COMPLETION.json')
    require(prepared['status']=='COMPLETE' and prepared['plan_sha256']==sha(PRIOR/'PLAN.json') and prepared['new_returns_computed'] is False and prepared['original_array_count']==27, '209 prepared contract')
    expectedfiles={'FEATURES.npz','TRANSFORMS_DELTAS.npz','LABELS.npz','LABEL_AUDIT.json','MODEL_AUDIT.json','RANK_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','IDENTITY_AUDIT.json','CALENDAR.json','INDEPENDENT_PREPARE.json','CONTROL_PARITY.json'}
    require(set(prepared['artifacts_sha256'])==expectedfiles, '209 exact prepared manifest')
    for name,h in prepared['artifacts_sha256'].items():bind(PRIOR/'prepared'/name,h)
    independence=read(PRIOR/'prepared/INDEPENDENT_PREPARE.json')
    require(independence['source_cohort_exact'] is True and independence['original_arrays_unchanged'] is True and independence['original_training_arrays_unchanged'] is True, '209 cohort/immutability audit')
    for flag in ('passed','full_official_path_checked','endpoint_calendar_checked','audit_checked','no_query_future_outcomes'):
        require(independence['model'][flag] is True, '209 independent source flag '+flag)
    identity=read(PRIOR/'prepared/IDENTITY_AUDIT.json')
    require(identity==oldreview['source_identity'] and identity['exact_all9_source_arrays'] is True and identity['historical_metadata_equal_architecture'] is True and identity['explicit_symbols_offsets_dates_ends'] is True,'209 prior identity receipt')
    parity=read(PRIOR/'prepared/SOURCE_PARITY.json')
    require(parity['raw_source_hashes']==raw and parity['independent_raw_prices_verified'] is True and parity['training_identity_verified'] is True and parity['official_calendar_independently_verified'] is True, '209 prior raw/calendar parity')
    meta=read(LOCAL/'META.json'); trainall=read(endpoint['inputs']['training_meta']['path']); training=trainall['training']
    require(trainall['monthly']==meta==read(PRIOR/'prepared/META.json'), 'Monthly metadata identity differs')
    architecture=read('outputs/monthly_architecture_restart/META.json')
    require(architecture['training']==training, 'Original training metadata differs from architecture')
    qids=axes(meta,122,49);tids=axes(training,180,49)
    bind(CALENDAR,CALENDAR_SHA);cal=read(CALENDAR);sessions=cal['sessions']
    require(cal==read(PRIOR/'prepared/CALENDAR.json') and cal['provenance']==endpoint['calendar'], 'Calendar record/provenance differs')
    for i,s in enumerate(sessions):require(day(s)==s and (i==0 or s>sessions[i-1]), 'Calendar date ordering')
    pos={s:i for i,s in enumerate(sessions)};last={s[:7]:s for s in sessions}
    require(len(pos)==len(sessions) and len(sessions)>0,'Calendar uniqueness')
    clock=[]; oldcoverage=oldreview['training_path_coverage'];oldrecords=oldcoverage['records']
    labelaudit=read(PRIOR/'prepared/LABEL_AUDIT.json');months=labelaudit['months']
    require(len(oldrecords)==len(months)==180,'Prior training path month count')
    for i,(s,e) in enumerate(zip(training['dates'],training['ends'])):
        require(last[s[:7]]==s and last[next_month(s)]==e and sessions[pos[e]+1][:7]>e[:7], 'Full official targetmonth endpoints')
        wanted=sessions[pos[s]:pos[e]+1];oldr=oldrecords[i];labelr=months[i]
        require(oldr['signal']==s and oldr['end']==e and oldr['rows']==49 and oldr['official_path_dates']==wanted, '209 old source dates mismatch')
        require(labelr['signal_date']==s and labelr['label_end']==e and labelr['source_dates']==wanted and labelr['target_dates']==wanted[1:] and labelr['target_session_count']==len(wanted)-1,'209 saved label calendar mismatch')
        require(oldr['base_rows']==labelr['base_rows']==labelr['good_rows'] and oldr['issues']==[], '209 saved complete-path certificate mismatch')
        clock.append(dict(signal_date=s,label_end=e,target_session_count=len(wanted)-1,next_month_witness=sessions[pos[e]+1]))
    for i,(s,e) in enumerate(zip(meta['dates'],meta['ends'])):
        require(s in pos and e in pos, 'Original query/evaluation session missing')
        if i==0:require(s=='2016-08-17' and e=='2016-08-31','Original initial partial')
        else:require(last[s[:7]]==s and (e==last[next_month(s)] or (i==121 and e=='2026-09-24')), 'Original query endpoint')
    require(oldcoverage['training_rows']==8820 and oldcoverage['training_months']==180 and oldcoverage['base_rows']==oldcoverage['candidate191_good_rows']==oldcoverage['path_complete_base_rows']==labelaudit['base_rows']==labelaudit['good_rows']==identity['eligible_rows']==independence['model']['good_training_rows'],'209 admitted certificate counts differ')
    annual=[]
    for y in sorted({int(s[:4]) for s in meta['dates']}):
        cutoff='%04d-12-31'%(y-1);lower='%04d-12-31'%(y-6)
        candidate_months=[s for s,e in zip(training['dates'],training['ends']) if s>lower and e<=cutoff]
        require(all(int(s[:4])<y for s in candidate_months),'Annual maturity metadata')
        annual.append(dict(year=y,cutoff=cutoff,exclusive_signal_lower_bound=lower,metadata_candidate_months=len(candidate_months),numeric_good_support_computed=False))
    implementation=bind(__file__)
    certificate=dict(training_rows=oldcoverage['training_rows'],training_months=oldcoverage['training_months'],
        admitted_good_rows=oldcoverage['candidate191_good_rows'],final_good_mask_sha256=oldcoverage['final_good_mask_sha256'],
        base_rows=oldcoverage['base_rows'],path_complete_base_rows=oldcoverage['path_complete_base_rows'],
        provenance='Pinned prior209 source coverage and independent completed preparation; no225 support recomputed.',
        prior209_certificate=True,fresh225_support=False,source_review_sha256=SOURCE209_SHA,
        prepared_completion_sha256=sha(PRIOR/'prepared/COMPLETION.json'),labels_archive_sha256=prepared['artifacts_sha256']['LABELS.npz'])
    report=dict(schema='relative_duration225_source_metadata_v1',created_utc=stamp(),passed=True,source_only=True,
        context_sha256=CONTEXT_SHA,coverage_sha256=sha(COVERAGE),evidence_sha256=context['evidence_sha256'],
        source_admission_metadata_only=True,metadata_axes_verified=True,calendar_clock_verified=True,
        raw_source_hashes_verified=True,raw_source_map_matches_original=True,endpoint155_wiring_verified=True,
        prior209_complete_path_certificate_verified=True,prior209_prepared_manifest_verified=True,
        annual_cutoff_calendar_verified=True,canonical_identity_scope_reviewed=True,
        source_sha256=dict(sorted(PINS.items())),raw_source_hashes=raw,calendar_sha256=CALENDAR_SHA,
        endpoint_source_binding=dict(plan=dict(path=str(PLAN155),sha256=PLAN155_SHA),inputs={k:endpoint['inputs'][k] for k in required},calendar=endpoint['calendar']),
        training_path_coverage=certificate,source_identity=identity,
        metadata_axes=dict(query_months=122,width=49,query_identities=len(qids),training_months=180,training_identities=len(tids),plain10_width=10,
            original_plain10_shape_evidence='Frozen209source_inputs/IDENTITY_AUDIT; no numeric arrays decoded in225preflight.',
            training_source='Original155 target_definition70 TRAIN_INPUTS + earnings_factor_research META; not169 rebuildcache.',
            first_training_date=training['dates'][0],last_training_date=training['dates'][-1],last_training_label_end=training['ends'][-1]),
        calendar_metadata=dict(sessions=len(sessions),first=sessions[0],last=sessions[-1],training_windows=clock,annual_windows=annual),
        post_registration_required_cohort_parity=['dates','ends','symbols','base_good','good'],
        post_registration_requirements=['225 exact cohort dtype/shape/bytes versus209 and saved goodmask digest.','Originalplain10 and all9original training arrays must retain frozen identity.','Compute both newlabels, both annualRidge arms, independent scalar labels/fits and truncatedprefix checks only after immutable225PLAN.'],
        old_numeric_label_parity_claimed=False,old_control_prediction_parity_claimed=False,old_control_order_parity_claimed=False,
        both225_targets_new=True,source_code_imported=False,npz_decoded=False,
        actual_source_coverage_recomputed=False,actual_new_support_computed=False,actual_features_computed=False,
        actual_labels_computed=False,actual_model_fits_computed=False,actual_orders_computed=False,actual_returns_computed=False,
        strategy_registered=False,observers_modified=False,central_modified=False,strict_point_in_time=False,
        source_validation=dict(unique_verified_pins=len(PINS),raw64_count=len(raw),prepared_files=len(expectedfiles),all_hashes_passed=True),
        limitations=['209 source helper originally read cached masks and actual price support;225 only verifies its immutable prior certificate and JSON calendar metadata, not its numeric computations anew.',
            'Exact stored security IDs and canonical columns are research identity, not a new permanentissuer/tickerreuse certificate. Inherited alias continuity and vendor revisions remain.',
            'Frozen loader includes retrospective source overrides and a2016 leading-gap price splice; no current source/PIT upgrade is claimed.',
            'The original estimated historical Top32/theme17 universe and repeated research sample remain; strict point-in-time vendor vintages and untouched validation are not certified.',
            'Original final outcome cutoff2026-09-24 does not shorten any future naturalmonth traininglabel. Both newtarget forecasts are not oldcontrol equivalents.'],
        host_not_pinned_to_avoid_cycle=True,implementation=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=implementation))
    write_once(OUTPUT,report)
    return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['coverage','source'],required=True);parser.add_argument('--context-sha256',required=True)
    args=parser.parse_args();require(args.context_sha256==CONTEXT_SHA,'Explicit context differs');bind(CONTEXT,CONTEXT_SHA);context=read(CONTEXT)
    report=coverage(context) if args.mode=='coverage' else source(context)
    out=COVERAGE if args.mode=='coverage' else OUTPUT
    print(json.dumps(dict(passed=True,mode=args.mode,output=str(out),sha256=sha(out),implementation_sha256=sha(__file__),pins=len(PINS),training_path_coverage=report.get('training_path_coverage')),indent=2))

if __name__=='__main__':main()
