from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'work/monthly_sector_recurrence192_host.py').read_text()
for a,b in [('monthly_sector_recurrence192','monthly_volume_cluster195'),('sectorrecurrence192','volumecluster195'),('MONTHLY_SECTOR_RECURRENCE192','MONTHLY_VOLUME_CLUSTER195'),('Round192','Round195'),('sector_relative_recurrence','volume_cluster'),('absolute_recurrence_control','volume_frequency_control'),('monthly_relative_severity191','monthly_trade_wc194'),('Round191','Round194')]:s=s.replace(a,b)
def function(name,body):
 global s
 node=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==name)
 lines=s.splitlines(keepends=True);s=''.join(lines[:node.lineno-1])+body.rstrip()+'\n'+''.join(lines[node.end_lineno:])
s=s.replace('frozen Local67 plus signed 21-session own-sector recurrence','frozen Local67 plus unsigned high-volume event clustering')
s=s.replace('Twelve exact endpoint signs are counts, not fitted means.','Rolling medians are descriptive estimates; no predictive model fits.')
start=s.index('TRAIN, DOSE');end=s.index('TEST_RECEIPT =',start)
s=s[:start]+"""TRAIN, DOSE, MAX_RAW, MAX_DELTA = 88, .01, .5, .005
FEATURE_PREFIXES = (0, 1, 53, 88, 122)
ESTIMATE_FIELDS = ('event_observations', 'adjacent_pair_observations', 'rolling_medians_computed')
MODEL_FIT_ACCOUNTING = 'Zero new predictive model fits; each available cell uses210 descriptive prior63 medians and209 adjacent binary-event pairs. Main counts exclude independent and prefix verification reruns.'
CONTEXT = ROOT / 'work/monthly_volume_cluster195_context.json'
CONTEXT_SHA = '1a6e06499d35667ae435887f3d2d2e56fce4ea4194531032b2ab4969b046b0a0'
SOURCE_REVIEW = ROOT / 'work/volume195_SOURCE_REVIEW.json'
SOURCE_REVIEW_SHA = 'PENDING_SOURCE195'
PRIOR_EXPECTED = {
 'PLAN.json': '6bac3e3aedf4aa9350cbc1b9583c0d7231236f44aa88b1b5d5276aad10bd531c',
 'COMPLETION.json': 'ab8a8171a27dd7218408633145bcc8aaee6398ccb16f52b3d58d6926f6ee8219',
 'results/COMPLETION.json': '670802971a2cf2d4975b3cc387f3f470a338835d6db03ba086bca99638a0bf16',
}
FEATURE_FLAGS = ('passed','independent_scalar_rebuild','strict_prior63_median_checked','unsigned_volume_events_checked','adjacent_covariance_checked','matched_frequency_checked','full273_volume_support_checked','official_session_windows_checked','monthend_queries_checked','exact_common_availability','audit_checked','no_future_or_outcome_inputs','centered_high_tie_ranks_checked')
"""+s[end:]
function('contract',"""def contract():
    require(sha(CONTEXT)==CONTEXT_SHA, 'Changed prospective context')
    context=read(CONTEXT)
    return dict(context=context,prospective_context_sha256=CONTEXT_SHA,arms=list(NAMES),candidate=NAMES[1],control_nonpromotable=NAMES[2],period=['2016-08-17','2026-09-24'],training_months=88,later_months=34,dose=DOSE,max_absolute_raw=MAX_RAW,max_absolute_delta=MAX_DELTA,
        source_window=context['formation'],scoring='Higher centered average-tie ranks on common available original candidates, .01 additive to full Local67 pre-score before exactly one original episode. Initial partial-row delta zero.',feature_prefix_source='Physically truncate volume through prefix signal date, including empty prefix0.',verification=dict(independent_all_cells=True,feature_prefixes=list(FEATURE_PREFIXES),all123_order_prefixes=True),limitation=context['limitations'],historical_best_tracking='First88 selects; full-period best is descriptive among previously training-selected research candidates, not pristine OOS or observer promotion.',new_model_fits=0,new_predictive_model_fits=0,predictive_model_fits=0,model_fit_accounting=MODEL_FIT_ACCOUNTING,new_arms=2,registered_trials=3,execution_paths=9,no_observer_writes=True,original_pool_preserved=True,no_parameter_search=True)
""")
s=s.replace("'price_distribution171_independent.py')]","'price_distribution171_independent.py', 'monthly_sector_recurrence192_independent.py')]")
function('sector_source_binding',"""def volume_source_binding(raw):
    require(sha(SOURCE_REVIEW)==SOURCE_REVIEW_SHA,'Changed source review')
    review=read(SOURCE_REVIEW)
    require(review.get('passed') is True,'Volume source admission failed')
    require(review.get('context_sha256')==CONTEXT_SHA,'Source concept changed')
    require(review.get('raw_source_hashes')==raw,'Volume raw source lineage changed')
    for key in ('actual_features_computed','actual_orders_computed','actual_returns_computed'):
        require(review.get(key) is False or type(review.get(key)) is int and review[key]==0,'Source audit computed strategies: '+key)
    pins={str(SOURCE_REVIEW):SOURCE_REVIEW_SHA}
    for name,digest in review['source_sha256'].items():
        bind(pins,Path(name) if Path(name).is_absolute() else ROOT/name,digest)
    verify(pins)
    return dict(path=str(SOURCE_REVIEW),sha256=SOURCE_REVIEW_SHA,input_sha256=pins,source_column='Original stored native volume; no re-adjustment or imputation',strict_pit=False)
""")
function('sector_inputs',"""def volume_inputs(source,meta):
    require(len(meta['etfs'])==17 and len(set(meta['etfs']))==17,'Original17 ETF identities required')
    require(isinstance(source.get('volume'),pd.DataFrame),'Original stored volume required')
    return {'volume':source['volume']}
""")
function('estimate_counts',"""def estimate_counts(check):
    available=check.get('available_cells')
    require(type(available) is int and 0<=available<=122*49,'Invalid available-cell count')
    counts={key:check.get(key) for key in ESTIMATE_FIELDS}
    expected=dict(event_observations=210*available,adjacent_pair_observations=209*available,rolling_medians_computed=210*available)
    require(all(type(value) is int and value==expected[key] for key,value in counts.items()),'Descriptive event/median counts differ')
    return counts
""")
s=s.replace("{'adj', 'sectors'}","{'volume'}").replace('including classifications','with full original identities').replace('Original adjusted prices and sector classifications required','Original stored volume frame required')
s=s.replace("== 189 and sum(r['trials'] for r in state['completed']) == 1084","== 192 and sum(r['trials'] for r in state['completed']) == 1093").replace('Expected189 completed rounds /1084 trials','Expected192 completed entries /1093 trials')
s=s.replace("== 190 and sum(r['trials'] for r in state['completed']) == 1087","== 193 and sum(r['trials'] for r in state['completed']) == 1096")
s=s.replace('source_binding = sector_source_binding(raw)\n    bind(pins, MAPPING_PATH, MAPPING_SHA)',"source_binding = volume_source_binding(raw)\n    numerical_review = code_review_binding()\n    for name,digest in source_binding['input_sha256'].items(): bind(pins,Path(name),digest)\n    for name,digest in numerical_review['input_sha256'].items(): bind(pins,Path(name),digest)")
s=s.replace('sector_source_binding=source_binding,','volume_source_binding=source_binding,numerical_code_review=numerical_review,')
s=s.replace('actual_relative_sign_observations_before_registration=0, actual_absolute_sign_observations_before_registration=0,','actual_volume_events_before_registration=0, actual_rolling_medians_before_registration=0,')
s=s.replace('registered_before_new_sector_recurrence_features','registered_before_new_volume_cluster_features').replace('Prepare fixed relative and absolute signed-block recurrence','Prepare fixed unsigned volume clustering and matched frequency')
s=s.replace("require(p['sector_source_binding'] == sector_source_binding(p['raw_source_hashes']), 'Registered sector source binding changed')","require(p['volume_source_binding'] == volume_source_binding(p['raw_source_hashes']), 'Registered volume source binding changed')\n    require(p['numerical_code_review']==code_review_binding(),'Registered independent code review changed')")
function('scale_and_delta',"""def scale_and_delta(values,n):
    require(set(values)=={'candidate','control','available','candidate_rank','control_rank'},'Feature keys changed')
    available=np.asarray(values['available']);shape=available.shape
    require(len(shape)==2 and shape[1]>0 and type(n) is int and 0<=n<=shape[0] and available.dtype==np.dtype(bool),'Invalid feature prefix/mask')
    transforms,deltas={},{}
    for arm,key in RAW.items():
        raw=np.asarray(values[key]);rank=np.asarray(values[key+'_rank'])
        require(raw.shape==rank.shape==shape,'Feature shape')
        x=np.asarray(raw[:n],float);r=np.asarray(rank[:n],float)
        require(not np.isinf(x).any() and np.array_equal(np.isfinite(x),available[:n]),'Common raw availability')
        lo,hi=(-.25,.25) if key=='candidate' else (0.,1.)
        require(((x[available[:n]]>=lo)&(x[available[:n]]<=hi)).all(),'Raw event statistic bounds')
        require(np.isfinite(r).all() and (np.abs(r)<=MAX_RAW).all() and (r[~available[:n]]==0.).all(),'Rank bounds/missing neutrality')
        expected=np.zeros_like(r)
        for i in range(n):
            ids=np.flatnonzero(available[i])
            if len(ids)>1:
                for j in ids:
                    less=np.count_nonzero(x[i,ids]<x[i,j]);equal=np.count_nonzero(x[i,ids]==x[i,j])
                    expected[i,j]=(less+(equal-1)/2)/(len(ids)-1)-.5
        np.testing.assert_array_equal(r,expected)
        delta=DOSE*r
        if n:delta[0]=0.
        require(np.max(np.abs(delta),initial=0.)<=MAX_DELTA,'Score budget exceeded')
        transforms[arm],deltas[arm]=r.copy(),delta
    return transforms,deltas
""")
s=s.replace("    require(check.get('mapping_source_sha256') == MAPPING_SHA, 'Independent mapping source changed')\n",'')
s=s.replace("'no_cross_sectional_ranking'","'centered_cross_sectional_ranking'")
s=s.replace('panel, mapper = sector_inputs(study, source, meta)','panel = volume_inputs(source, meta)')
s=s.replace('kernel.build_features(panel, meta, calendar, mapper)','kernel.build_features(panel, meta, calendar)')
s=s.replace('independent.verify_features(panel, meta, calendar, mapper, values, audit)','independent.verify_features(panel, meta, calendar, values, audit)')
s=s.replace('kernel.build_features(prefix_panel, meta, calendar, mapper, limit=n)','kernel.build_features(prefix_panel, meta, calendar, limit=n)')
s=s.replace("sector_source_binding=plan['sector_source_binding'], feature_panel='original source adj and sectors'","volume_source_binding=plan['volume_source_binding'], feature_panel='original source native volume'")
s=s.replace('close_vs_absolute_recurrence','close_vs_frequency').replace('open_vs_absolute_recurrence','open_vs_frequency')
s=s.replace('sector_relative_signed_block_recurrence=True','unsigned_volume_cluster=True')
s=s.replace('Review relative versus absolute signed-block recurrence identification and results','Review unsigned high-volume clustering versus frequency and source limitations')
s=s.replace("'Zero new predictive or descriptive model fits.","'Zero new predictive model fits.")
# Freeze independent code-review receipt at registration, outside the six reviewed files.
a=(ROOT/'work/monthly_trade_wc194_host.py').read_text();node=next(x for x in ast.parse(a).body if isinstance(x,ast.FunctionDef) and x.name=='code_review_binding')
f=''.join(a.splitlines(keepends=True)[node.lineno-1:node.end_lineno]).replace('outputs/monthly_working_capital194_source/independent/CODE_REVIEW.json','work/volume195_CODE_REVIEW.json').replace("'actual_financial_ratios_computed', ", '')
s=s.replace('def register():',f+'\n\ndef register():')
assert 'MAPPING_' not in s and 'mapper' not in s and 'sector_source_binding' not in s
ast.parse(s)
out=ROOT/'work/monthly_volume_cluster195_host.py';assert not out.exists();out.write_text(s)
print(out)
