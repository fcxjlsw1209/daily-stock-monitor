"""Round200 source-only review and bounded historical coverage record.

Run through run_local.sh. Only the pinned original load_panel is imported.
No cached NPZ arrays, indicators, sigma, ranks, deltas, orders or returns.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / 'outputs'
CONTEXT = ROOT / 'work/monthly_cmf_state200_context.json'
CONTEXT_SHA = 'bccab811afc6d2a1fa27d26e205b30b9d0b75df8ca83b69e03734177217d80a0'
COVERAGE = ROOT / 'work/after199_cmf_state_coverage.json'
OUT = ROOT / 'work/cmf200_SOURCE_REVIEW.json'
PRIOR = ROOT / 'work/nested199_SOURCE_REVIEW.json'
PRIOR_SHA = 'd12cdcfc032404e297f1d95a2753785175db9892b71c0e8b49f241470bfad911'
VOLUME = ROOT / 'work/volume195_SOURCE_REVIEW.json'
VOLUME_SHA = 'ea5d8c96d8c340cf01d6a6db4ec424a1379c03a6545446039821d2e17828fe40'
CHAIN = (
    'outputs/monthly_regime_monitor/strategy.py',
    'outputs/monthly_stability_monitor/strategy.py',
    'outputs/monthly_quality_research/explore.py',
    'outputs/monthly_state_rank_research/state_rank.py',
    'outputs/monthly_target_definition70/study.py',
    'outputs/monthly_model_scope70/study.py',
    'outputs/monthly_local_linear70/study.py',
    'outputs/monthly_daily_episodes/episodes.py',
    'outputs/monthly_path70_research/incumbent_adapter.py',
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def pin(pins, path, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(ROOT), 'Source must be in project')
    digest = sha(path)
    require(expected is None or digest == expected, 'Changed source: ' + str(path))
    require(str(path) not in pins or pins[str(path)] == digest, 'Conflicting pin')
    pins[str(path)] = digest
    return digest


def write_once(path, value, pins):
    for p, h in pins.items():
        require(sha(p) == h, 'Source changed during review: ' + p)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(passed=True, path=str(path), sha256=sha(path), source_pins=len(pins))))


def flags():
    return dict(actual_features_computed=False, actual_orders_computed=False,
        actual_returns_computed=False, actual_portfolio_returns_computed=False,
        actual_sigma_computed=False, actual_state_cdf_computed=False,
        actual_cmf_computed=False, actual_cmf_ranks_computed=False,
        actual_ranks_computed=False, actual_deltas_computed=False,
        actual_signal_returns_computed=False, new_model_fits=0,
        production_files_modified=False, strategy_registered_by_reviewer=False,
        original_sources_modified=False, observers_accessed=False)


def context_and_chain(pins):
    pin(pins, Path(__file__))
    pin(pins, CONTEXT, CONTEXT_SHA)
    context = read(CONTEXT)
    require(context['round'] == 200 and context['name'] == 'monthly_cmf_state200', 'Wrong context')
    require(context['pre_registration_actual_new_features'] == 0
        and context['pre_registration_actual_orders'] == 0
        and context['pre_registration_actual_returns'] == 0, 'Nonzero prospective counters')
    for path, digest in context['evidence_sha256'].items():
        pin(pins, ROOT/path, digest)
    chain = {p: pin(pins, ROOT/p) for p in CHAIN}
    state = (ROOT/CHAIN[3]).read_text()
    target = (ROOT/CHAIN[4]).read_text()
    scope = (ROOT/CHAIN[5]).read_text()
    local = (ROOT/CHAIN[6]).read_text()
    episode = (ROOT/CHAIN[7]).read_text()
    require("support=(a['baseline'][:n]-core)/(1-a['core_weight'][:n])" in state
        and "result+=b*scale*(a[z['key']][:n]-support)" in state,
        'Original support budget changed')
    require("amount=.025, state='calm10'" in target and "amount=.05" in target
        and "state='weak_narrow'" in target
        and "pre_score=ep.old.score(prior_policy,original,len(m['dates']))" in target,
        'Original cached pre score chain changed')
    require("active=use[:,None]&available&common&eligible&np.isfinite(rank)" in scope
        and "delta[active]=spec['budget']*(rank[active]-support[active])" in scope
        and "SOURCE_SCOPE_SPEC=dict(id=3,base='incumbent67',scope='calm_nonweak',budget=.01)" in local
        and 'delta[active]=coefficient[active]*(rank[active]-current[active])' in local,
        'Original scope/local correction changed')
    require("support = (a['baseline'][:n] - a['core'][:n]) / (1 - a['core_weight'][:n])" in episode,
        'Original episode support changed')
    return context, chain


def coverage():
    require(not COVERAGE.exists(), 'Write-once coverage already exists')
    pins = {}
    context, chain = context_and_chain(pins)
    paths = [
        'outputs/monthly_target70_research/transition.py',
        'outputs/monthly_target70_research/PLAN.json',
        'work/monthly_latent_state134_proposal.json',
        'work/monthly_latent_state134_model.py',
        'work/monthly_latent_state134_runner.py',
        'outputs/monthly_latent_state134/COMPLETION.json',
        'outputs/monthly_latent_state134/STATE_AUDIT.json',
        'outputs/monthly_latent_state134/INDEPENDENT_PREPARE.json',
        'outputs/monthly_latent_state134/INDEPENDENT_OUTCOME.json',
        'work/monthly_absolute_state135_proposal.json',
        'work/monthly_absolute_state135_model.py',
        'work/monthly_absolute_state135_runner.py',
        'outputs/monthly_absolute_state135/PLAN.json',
        'work/vvix160_proposal.json', 'work/vvix160_kernel.py',
        'outputs/monthly_vvix160/COMPLETION.json',
        'work/monthly_relative_severity191_context.json',
        'work/monthly_relative_severity191_model.py',
        'work/monthly_relative_severity191_central_before_review.json',
        'outputs/monthly_relative_severity191/COMPLETION.json',
        'outputs/monthly_relative_severity191/results/COMPLETION.json',
        'work/monthly_vol_reallocation185_context.json',
    ]
    for p in paths: pin(pins, ROOT/p)
    latent = read(O/'monthly_latent_state134/COMPLETION.json')
    latent_outcome = read(O/'monthly_latent_state134/INDEPENDENT_OUTCOME.json')
    vvix = read(O/'monthly_vvix160/COMPLETION.json')
    severity = read(O/'monthly_relative_severity191/COMPLETION.json')
    require(latent['status'] == 'completed_independently_verified'
        and latent['new_model_fits'] == 11 and latent_outcome['passed'] is True
        and latent_outcome['execution_paths_verified'] == 9, '134 execution evidence differs')
    require(vvix['independent_execution_paths'] == 9 and vvix['new_model_fits'] == 0,
        '160 execution evidence differs')
    require(severity['status'] == 'COMPLETED_INDEPENDENTLY_VERIFIED'
        and severity['new_model_fits'] == 22, '191 execution evidence differs')
    missing135 = ['outputs/monthly_absolute_state135/'+x for x in
        ('COMPLETION.json','INDEPENDENT_PREPARE.json','INDEPENDENT_OUTCOME.json')]
    require(all(not (ROOT/p).exists() for p in missing135), '135 evidence availability changed; re-review needed')
    findings = [
        dict(subject='Original hard states and pre-score coefficient',
            formula='r=+1 iff SPY rolling63 sample std > its trailing252 median, else -1. Both include query. calm is separately sigma63*sqrt252<.10. q=avg_pct_rank(-CMF63).fillna(.5)-.5 over original49 before eligibility. Baseline a=.075 calm; otherwise .075*(1-.10*I[r>0]/.48).',
            direct_pre_coefficient='a_eff=a*K; K=1-(.025*calm+.05*weak+.01*active)/(1-cw), active=calm & ~weak & extension_available & forecast_available & eligible & finite(rank_log_excess).',
            derivation='state_rank.score subtracts (.025calm+.05weak)*support; its cap scale is1 because maximum .075. Target-definition weak correction changes ranks only. Scope config3 additionally subtracts .01support on active. Local-linear config4 changes ranks only. Original support=(baseline-core)/(1-cw). All other frozen arrays held fixed.',
            integration='Candidate delta=a_eff*q*(g-r); fixed half-sign delta=-.5*a_eff*q*r. Apply only to complete original config4 pre-score; original boundary episode retains its original support, effects and policy. This softens only direct pre-stage CMF, not all indirect model/episode dependence.'),
        dict(subject='Early fixed calm transition',
            formula='calm_share=clip((high-annualized_sigma63)/(high-low),0,1); bands [.09,.11] and [.08,.12], versus hard<.10. Mix complete calm/noncalm scores, core and core weights; relative CMF r and industry hard-state logic remain.',
            execution_evidence='Source/PLAN and TRANSITION.npz exist; historical191 ledger snapshot records monthly_target70_research completed with85 trials. Detailed original per-trial/closure files missing: individual execution verification UNKNOWN.',
            duplicate_risk='Smooth market-state mixing already implemented, but not CMF-only empirical state or fixed half-CMF.'),
        dict(subject='134 prior-trained HMM and135 absolute quiet probability',
            formula134='Annual two-state Gaussian model trained on prior5calendar years through Dec31(Y-1), min756 returns. Forward filtered low-variance-state p mixes raw_skip42 and riskadjusted_skip42 ranks as p*raw+(1-p)*risk; stationary-prior memoryless matched control. Original core weight and auxiliary hard states preserved.',
            execution134=dict(status=latent['status'], new_model_fits=11, independent_paths=9,
                evidence='COMPLETION + STATE_AUDIT + INDEPENDENT_PREPARE + INDEPENDENT_OUTCOME'),
            formula135='Reuse134 fits. State annualized sigma_j=scaler.sd*sqrt(var_j)*sqrt252; q=sum(p_j*I[sigma_j<.10]); q=1 bothquiet,0 nonequiet,p_low onlylowquiet. Same core mixture, not CMF sign.',
            execution135='PLAN plus three partial exports and historical191 ledger completion claim exist. Current completion/independent receipts missing; full current verification UNKNOWN.',
            missing_evidence=missing135,
            duplicate_risk='Probability-based regime mixing was tested for core134. Do not claim HMM novelty or general untested soft-state family.'),
        dict(subject='160 continuous empirical market percentile times auxiliary score',
            formula='State=(2*less+equal-252)/251 on252 VIX or VVIX closes ending t-1; delta=-.02*state*original_vol126_rank on Local67 pre-score with original episode.',
            execution=dict(independent_paths=9,new_model_fits=0,protected_files=vvix['protected_files']),
            duplicate_risk='Closest mechanism family already executed.200 differs in SPY realized63vol source, including-t window, /252 scale, exact-median neutrality, CMF multiplier and matched half-sign control.'),
        dict(subject='191 relative severity;185 auxiliary reallocation',
            formula191='R=(stock_end/stock_start)/(SPY_end/SPY_start). Candidate max(0,1-R) versus indicator[R<1], exact rational comparisons; annual prior5year Ridge alpha10, rank negative predictions, .01 pre-score correction.',
            execution191=dict(status=severity['status'],new_model_fits=22,protected_files=severity['protected_files']),
            formula185='delta=lambda*((1-cw)*V-(B-C)), lambda .025/.05/.10; adjusts all nonvol auxiliary contributions, not only market-state CMF.',
            duplicate_risk='191 is a continuous training-target severity, not market volatility state.185 is broad auxiliary reallocation, not fixedhalfCMF.'),
        dict(subject='Narrow exact-coverage conclusion',
            proposed_gate='g=0 at exact sigma==window_median; otherwise (2*less+equal)/252-1; control=.5*r. Median-tie neutralization avoids sign reversal from asymmetric tied blocks.',
            exact_spy63_midcdf_cmf_gate_found=False, exact_fixed_half_cmf_found=False,
            scope='Targeted readable relevant work source/context/proposals and outputs source/PLAN; no extra broad search. No199 outcome data inspected. Historical execution evidence read only as metadata.',
            novelty='Mechanism refinement/retest; no exhaustive never-tested claim. Missing older definitions and135 detailed receipts leave historical exhaustive coverage UNKNOWN.',
            scope_bound='No new features, scores, orders or outcomes generated; no registration or central modification.')
    ]
    result = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        context_sha256=CONTEXT_SHA, review_code_sha256=sha(__file__), findings=findings,
        evidence_sha256={str(Path(p).relative_to(ROOT)):h for p,h in pins.items()},
        chain_sha256=chain, strict_pit=False, historical_exhaustive_coverage='UNKNOWN', **flags())
    write_once(COVERAGE, result, pins)


def quality(values):
    a = np.asarray(values, dtype=float)
    finite = np.isfinite(a)
    return dict(observations=int(a.size), finite_positive=int((finite & (a>0)).sum()),
        zero=int((finite & (a==0)).sum()), negative=int((finite & (a<0)).sum()),
        nan=int(np.isnan(a).sum()), infinite=int(np.isinf(a).sum()))


def window(calendar, day, length):
    t = calendar.index(day)
    require(t >= length-1, 'Official source history too short')
    dates = calendar[t-length+1:t+1]
    require(len(dates) == length and dates[-1] == day, 'Wrong source window')
    return dates


def source():
    require(not OUT.exists(), 'Write-once source review already exists')
    require(quality([1.,0.,-1.,np.nan,np.inf,-np.inf]) == dict(observations=6,
        finite_positive=1,zero=1,negative=1,nan=1,infinite=2), 'Synthetic count partition')
    require(window(list(range(400)),350,315) == list(range(36,351)), 'Synthetic315 offsets')
    require(window(list(range(400)),350,63) == list(range(288,351)), 'Synthetic63 offsets')
    try: window(list(range(400)),313,315)
    except ValueError: pass
    else: raise ValueError('Short source history must fail')
    pins = {}
    context, chain = context_and_chain(pins)
    pin(pins, COVERAGE)
    historical = read(COVERAGE)
    require(historical['passed'] is True and historical['context_sha256'] == CONTEXT_SHA,
        'Coverage must bind current context')
    for p,h in historical['evidence_sha256'].items(): pin(pins,ROOT/p,h)
    pin(pins, PRIOR, PRIOR_SHA); pin(pins,VOLUME,VOLUME_SHA)
    previous, volume = read(PRIOR), read(VOLUME)
    for reviewed in (previous,volume):
        require(reviewed['passed'] is True and reviewed['raw_source_parity']['exact_source_map_equal'],
            'Previous original-source audit unavailable')
        for p,h in reviewed['source_sha256'].items(): pin(pins,p,h)
    plan_path = O/'monthly_local_linear70/PLAN.json'
    pin(pins,plan_path)
    raw = read(plan_path)['raw_source_hashes']
    require(raw == previous['raw_source_hashes'] == volume['raw_source_hashes'] and len(raw)==64,
        'Original64 source closure differs')
    for p,h in raw.items(): pin(pins,p,h)
    meta_path = O/'monthly_local_linear70/META.json'
    calendar_path = O/'monthly_paired_target143/CALENDAR.json'
    pin(pins,meta_path);pin(pins,calendar_path);pin(pins,O/'monthly_paired_target143/PLAN.json')
    meta, record = read(meta_path),read(calendar_path)
    dates, rows, calendar = meta['dates'],meta['symbols'],record['sessions']
    require(calendar == sorted(set(calendar)), 'Invalid official calendar')
    require(record['provenance'] == read(O/'monthly_paired_target143/PLAN.json')['calendar'],
        'Calendar provenance differs')
    for p,h in record['provenance']['source_hashes'].items(): pin(pins,p,h)
    require(len(dates)==len(rows)==122 and dates==sorted(set(dates)), 'Original query axis differs')
    require(len(set(meta['etfs']))==len(meta['etfs'])==17, 'ETF identities differ')
    for i,(day,names) in enumerate(zip(dates,rows)):
        require(len(names)==len(set(names))==49 and set(meta['etfs']).issubset(names), 'Candidate identity differs')
        t=calendar.index(day)
        require((i==0 and day=='2016-08-17') or calendar[t+1][:7]!=day[:7], 'Not an original monthend')
    names=sorted({s for row in rows for s in row})
    loader=O/'monthly_stability_monitor/inputs.py'
    require(str(loader) in raw, 'Original loader not pinned')
    spec=importlib.util.spec_from_file_location('cmf200_original_source_only_loader',loader)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    panel,info=module.load_panel(meta['etfs'])
    used=dict(info['source_hashes']);used[str(loader)]=sha(loader)
    require(used==raw, 'Original loader source map changed')
    fields=('adj','close','high','low','volume')
    for key in fields:
        frame=panel[key]
        require(frame.index.is_unique and frame.index.is_monotonic_increasing and frame.columns.is_unique,
            'Bad source axes '+key)
        require(frame.index.equals(panel['adj'].index) and frame.columns.equals(panel['adj'].columns)
            and set(names).issubset(frame.columns), 'Source-field alignment differs '+key)
    spy_rows=[]; field_rows=[]
    totals={key:dict(observations=0,finite_positive=0,zero=0,negative=0,nan=0,infinite=0) for key in fields[1:]}
    for day,row in zip(dates,rows):
        sessions=window(calendar,day,315);index=pd.to_datetime(sessions)
        original_span=panel['adj'].loc[index[0]:index[-1]].index
        require(original_span.equals(index), 'Original SPY source axis differs from official315 sessions')
        counts=quality(panel['adj'].reindex(index=index)['SPY'].to_numpy())
        require(counts['finite_positive']==315, 'Incomplete positive SPY315 source window '+day)
        spy_rows.append(dict(date=day,start=sessions[0],end=sessions[-1],sessions=315,
            finite_positive=315,complete=True))
        cmf_sessions=window(calendar,day,63);idx=pd.to_datetime(cmf_sessions)
        require(panel['adj'].loc[idx[0]:idx[-1]].index.equals(idx), 'Original CMF63 source calendar differs')
        observed={}
        for key in fields[1:]:
            counts=quality(panel[key].reindex(index=idx,columns=row).to_numpy())
            require(sum(counts[k] for k in ('finite_positive','zero','negative','nan','infinite'))==counts['observations'],
                'Quality count partition '+key)
            observed[key]=counts
            for k,v in counts.items(): totals[key][k]+=v
        field_rows.append(dict(date=day,start=cmf_sessions[0],end=cmf_sessions[-1],
            sessions=63,candidates=49,field_counts=observed))
    require(all(v['observations']==122*49*63 for v in totals.values()), 'CMF source count accounting')
    central=read(O/'monthly_target70_research/CONTINUATION_STATUS.json')
    require(central['current']=='monthly_nested_trend199' and central['registered_queue']==[]
        and len(central['completed'])==197 and sum(x['trials'] for x in central['completed'])==1108,
        'Review must precede round200 registration')
    result=dict(schema_version=1,created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        source_only=True,conditional_research_sources_ready=True,unresolved_blockers=[],
        context_sha256=CONTEXT_SHA,review_code_sha256=sha(__file__),coverage_sha256=sha(COVERAGE),
        raw_source_hashes=raw,source_sha256=pins,raw_source_parity=dict(passed=True,
            exact_source_map_equal=True,count=64,current_loader_sha256=sha(loader),
            method='Execute only frozen original load_panel and hash all sources; no strategy/factor builder.'),
        axes=dict(monthly_queries=122,symbols_per_query=49,unique_candidate_symbols=len(names),etf_count=17,
            first_query=dates[0],last_query=dates[-1],original_panel_rows=len(panel['adj']),
            original_panel_columns=len(panel['adj'].columns),all_noninitial_queries_monthend=True,
            all_source_windows_end_at_query=True,loader_data_as_of=info['data_as_of']),
        spy_state_source_coverage=dict(window_sessions=315,earliest_offset=-314,latest_offset=0,
            complete_positive_windows=122,incomplete_windows=0,repeated_observations=122*315,
            first_window_start=spy_rows[0]['start'],last_window_end=spy_rows[-1]['end'],by_query=spy_rows,
            no_sigma_or_median_or_cdf_computed=True,
            interpretation='315 consecutive official-session positive prices provide source support for252 rolling63-return sigma values. Their arithmetic finiteness and canonical equality remain registered implementation obligations.'),
        cmf_input_source_coverage=dict(window_sessions=63,field_observation_counts=totals,by_query=field_rows,
            new_availability_mask_applied=False,
            interpretation='Descriptive original close/high/low/volume source counts only. Missing/zero values do not create a new filter; original zero-range, rolling and missing-neutral semantics must remain. No CMF or final-feature availability computed.'),
        source_lineage=previous['source_lineage'],volume_source_lineage=volume['source_lineage'],
        alias_rules=previous['alias_rules'],normalization_facts=[
            'adj is original stored adjusted close. close/high/low/volume are original frozen signal fields, not uniformly raw-share or execution OHLCV.',
            'Alias base fields retain original bar-valid/quarantine masks. Provider choice uses retrospective history coverage; META->research_equity_FB and BRK-B->BRK.B remain fixed.',
            'Yahoo archives declare split-adjusted OHLCV and dividend-adjusted close. Raw Tiingo/WIKI signal normalization can use future split factors through vendor anchors. These facts do not certify historical first-public vintages.',
            'Fixed manifest overrides, archived securities and leading-gap extension remain unchanged. Leading adj is spliced using the original first-overlap ratio; raw other fields including Volume are copied without that ratio.',
            'CHL/PTR final Tiingo overrides use adjClose and raw close/high/low/volume (not adjVolume), masking dates>=2021-01-01.',
            'No new rescaling, fetch, replacement, forward fill, imputation, clipping or provider reconciliation. Round195 positive-volume eligibility is NOT inherited; its reviewed source lineage alone is reused.'
        ],integration_source_verification=dict(passed=True,numerical_integration_verified=False,
            hash_bound_dependencies=chain,
            pre_stage_coefficient=historical['findings'][0],
            method='Source-text algebra only; all cached state/rank/model arrays remain unread.'),
        strict_pit=False,strict_pit_vintage=False,strict_pit_membership=False,strategy_admitted=False,
        actual_source_coverage_computed=True,pre_registration_actual_new_features=0,
        observed_central_state=dict(completed_rounds=197,cumulative_trials=1108,current=central['current'],
            registered_queue=[],role='Observation only; central file is not pinned or modified.'),
        limitations=[
            'Existing conditional research source coverage only; proxyTop32/theme17 membership, retrospective aliases and vendor/action revisions preclude certified strictPIT.',
            'No199 performance outcomes read. Predecessor context evidence files are hash-verified without extracting return values.',
            'No sigma, empiricalCDF, CMF/rank, actual K, score delta, cache parity, orders or returns computed. Source readiness does not certify numeric feature construction or selection.',
            'CMF smoothing applies only to its direct pre-stage contribution; original episode support and indirect frozen model features remain unchanged by design.',
            'Registration, tests, independent implementation review and actual preparation remain separate obligations.'
        ],synthetic_source_checks_passed=4,**flags())
    write_once(OUT,result,pins)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('coverage','source'))
    globals()[parser.parse_args().phase]()
