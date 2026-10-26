"""One-time local source transformation; no data or financial computation."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def common(text):
    for a,b in (
        ('monthly_asset_growth147_prepare_review','monthly_session_cluster148_prepare_independent'),
        ('monthly_asset_growth147_outcome_review','monthly_session_cluster148_outcome_independent'),
        ('monthly_asset_growth147_review_contract','monthly_session_cluster148_review_contract'),
        ('monthly_asset_growth147','monthly_session_cluster148'),
        ('round147','round148'),('Round147','Round148'),('==147','==148'),
        ("'earnings_control', 'low_accrual_control', 'low_asset_growth'","'event_frequency_control', 'low_event_clustering'"),
        ("'earnings_control','low_accrual_control','low_asset_growth'","'event_frequency_control','low_event_clustering'"),
        ('all_four_score','all_three_score'),('Four exact','Three exact'),('Exactly four','Exactly three'),
        ('Four result','Three result'),('NAMES[3]','NAMES[2]'),('488','366'),
        ('strict_greater_than_all_three_controls','strict_greater_than_both_controls'),
        ('noninferior_incumbent_strict_greater_than_two_controls','noninferior_incumbent_strict_greater_than_frequency_control'),
    ):text=text.replace(a,b)
    return text

p=common((root/'work/monthly_asset_growth147_prepare_review.py').read_text())
start=p.index("METHOD=");end=p.index("CHECKS=")
p=p[:start]+"""METHOD=contract.METHOD
EXTRA=contract.EXTRA
HANDS=contract.HANDS
features_oracle=contract.features_oracle
"""+p[end:]
p=p.replace(",'admitted_dataset','source_admission'","")
start=p.index('def text(');end=p.index('def verify_original_arrays(')
p=p[:start]+"""def score_oracle(arrays,meta,frequency_rank,cluster_rank,n=None):
 helper=old.original_helper();x=helper.original_prefix(arrays,meta,n);n=x['n'];shape=x['core'].shape
 hands=[np.asarray(h)[:n] for h in (frequency_rank,cluster_rank)]
 for hand in hands:
  require(hand.shape==shape and hand.dtype==np.dtype('float64') and np.isfinite(hand).all(),'Finite rank hand')
  require((np.abs(hand)<=.5).all(),'Half-rank bound')
 scores={};choices={};deltas={}
 for k,name in enumerate(NAMES):
  pre=x['local'].copy();delta=np.zeros(shape)
  if k:
   hand=hands[k-1];active=hand!=0
   delta[active]=.01*hand[active]
   pre[active]+=delta[active]
  choice,final,_=helper.episode(pre,x)
  scores[name]=final;choices[name]=choice;deltas[name]=delta
 return scores,choices,deltas


"""+p[end:]
p=p.replace('Original 27 plus four new arrays','Original 27 plus five new arrays')
start=p.index('def verify_audit(');end=p.index('def ordered_orders(')
p=p[:start]+"""def verify_audit(saved,expected,meta):
 contract.feature_checks(saved,meta)
 require(saved['months']==expected['months'],'Exact skip42 window metadata')
 require(saved['cells']==expected['cells'],'Exact independent event counts and ranks')


"""+p[end:]
start=p.index('def source_gate_oracle(');end=p.index('def audit(')
p=p[:start]+"""def synthetic_receipt(result,root=None):
 root=ROOT if root is None else Path(root).resolve()
 reviewed=[root/'work'/name for name in ('monthly_session_cluster148_runner.py','monthly_session_cluster148_factor.py',
  'monthly_session_cluster148_proposal.json','monthly_session_cluster148_concept.json')]
 return dict(schema_version=1,stage='synthetic_prepare_review_only',passed=result.wasSuccessful(),
  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
  runtime=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__),
  source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files(root)},
  reviewed_contract_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in reviewed},
  actual_preparation_checked=False,actual_factors_calculated=0,actual_returns_calculated=0,
  producer_modules_imported=False,arms=list(NAMES),features=list(EXTRA),required_checks=list(CHECKS),
  actual_api='./run_local.sh work/monthly_session_cluster148_prepare_independent.py --plan-sha <SHA256> --prepare-complete')


def verify_prefixes(arrays,meta,sessions,close,opening,expected,scores,choices):
 for n in range(len(meta['dates'])+1):
  subset=dict(meta,dates=meta['dates'][:n],symbols=meta['symbols'][:n])
  last=pd.Timestamp(meta['dates'][n-1]) if n else None
  calendar=[d for d in sessions if last is not None and pd.Timestamp(d)<=last]
  prefix=features_oracle(subset,calendar,close.loc[calendar],opening.loc[calendar])
  require(prefix['cells']==expected['cells'][:n*49],'Feature prefix isolation')
  for key in EXTRA:np.testing.assert_array_equal(prefix['arrays'][key],expected['arrays'][key][:n])
  ss,cc,_=score_oracle(arrays,meta,*(prefix['arrays'][k] for k in HANDS),n=n)
  for name in NAMES:require(cc[name]==choices[name][:n] and ss[name].tobytes()==scores[name][:n].tobytes(),'Score and holding prefix isolation')
 neutral=np.zeros((len(meta['dates']),49))
 ss,cc,_=score_oracle(arrays,meta,neutral,neutral)
 for name in NAMES[1:]:require(ss[name].tobytes()==scores[NAMES[0]].tobytes() and cc[name]==choices[NAMES[0]],'Zero/missing rank exact fallback')


"""+p[end:]
start=p.index(' dataset=verify_admission(');end=p.index(' for key in EXTRA:',start)
p=p[:start]+""" close,opening=contract.price_frames(prices,sessions,meta)
 expected=features_oracle(meta,sessions,close,opening)
 verify_audit(reader.json(directory/'FEATURE_AUDIT.json'),expected,meta)
"""+p[end:]
p=p.replace("for key in EXTRA[:3]","for key in HANDS")
p=p.replace('verify_prefixes(arrays,meta,dataset,expected,scores,choices)',
            'verify_prefixes(arrays,meta,sessions,close,opening,expected,scores,choices)')
start=p.index('  orders_checked=');end=p.index('\n\n\ndef main',start)
p=p[:start]+"""  orders_checked=366,source_cells_checked=5978,stock_cells_checked=3904,
  available_feature_cells=int(expected['arrays']['available'].sum()),
  session_event_arithmetic_verified=True,common_valid49_ranking_before_eligibility_verified=True,
  raw_prices_independently_verified=True,calendar_overlay_independently_verified=True,all_signal_prefixes_verified=True,
  max_score_error=max(float(np.max(np.abs(saved_scores[name]-scores[name]))) for name in NAMES),
  limitations=['Adjusted overnight includes dividend effects; original cached price vintage retained.',
   'Event frequency and lag1 covariance are associations, not evidence of investor identity.',
   'Only new ranks change; original27 arrays and one-episode selector remain unchanged.'])
"""+p[end:]
p=p.replace("from bisect import bisect_right\n","").replace("from decimal import Decimal\n","")
(root/'work/monthly_session_cluster148_prepare_independent.py').write_text(p)

o=common((root/'work/monthly_asset_growth147_outcome_review.py').read_text())
o=o.replace("HANDS = ('earnings_bounded', 'low_accrual_bounded', 'low_asset_growth_bounded')\nMETHOD = 'current_annual_cash_quality_v1'\nMEASUREMENT = contract.MEASUREMENT",
"""HANDS = contract.HANDS
EXTRA = contract.EXTRA
METHOD = contract.METHOD""")
o=o.replace("'protected_manifest', 'admitted_dataset', 'source_admission'","'protected_manifest'")
o=o.replace("len(group)==4","len(group)==3").replace('for i in range(3):','for i in range(2):').replace('values[3],','values[2],')
start=o.index('def validate_cells(');end=o.index('def monthly(',start)
o=o[:start]+"""validate_cells = contract.validate_cells


def verify_feature_matrices(arrays,feature,meta):
    shape=(122,49)
    want={k:np.full(shape,np.nan) for k in EXTRA[:2]}
    want.update({k:np.zeros(shape) for k in HANDS});want['available']=np.zeros(shape,bool)
    validate_cells(feature['cells'],meta)
    for cell in feature['cells']:
        i,j=cell['signal_index'],cell['pool_index'];want['available'][i,j]=cell['available']
        for k in HANDS:want[k][i,j]=cell[k]
        if cell['available']:
            for k in EXTRA[:2]:want[k][i,j]=cell[k]
    for k,v in want.items():
        require(k in arrays and arrays[k].shape==shape and arrays[k].dtype==v.dtype,'Feature matrix schema '+k)
        np.testing.assert_array_equal(arrays[k],v)
    return len(want)


def feature_checks(feature,validation,meta):
    require(type(validation.get('new_model_fits'))is int and validation['new_model_fits']==0,'No new models')
    contract.feature_checks(feature,meta)


"""+o[end:]
o=o.replace('execution_paths_verified=12','execution_paths_verified=9').replace('asset_growth_feature_matrices_verified=4','session_cluster_feature_matrices_verified=5')
start=o.index('def verify_admission(');end=o.index('def verify_diagnostic_start(',start)
o=o[:start]+o[end:]
o=o.replace('verify_proposal(reader,plan,pins);verify_admission(reader,plan,pins)','verify_proposal(reader,plan,pins)')
start=o.index("    extra=(");end=o.index("\n    require(prep.get('passed')",start)
o=o[:start]+"""    extra=('raw_prices_independently_verified','calendar_overlay_independently_verified',
           'all_signal_prefixes_verified','session_event_arithmetic_verified',
           'common_valid49_ranking_before_eligibility_verified')"""+o[end:]
start=o.index("    report['asset_growth_feature_matrices_verified']");end=o.index("    for path,sha in",start)
o=o[:start]+"""    report['session_cluster_feature_matrices_verified']=verify_feature_matrices(reader.npz(p/'DATA.npz'),feature,meta)
    for field,key in (('price_cache_sha256','prices'),('sessions_sha256','sessions')):
        require(feature.get(field)==reader.sha(plan['inputs'][key]['path']),'Feature source binding')
    report.update(schema_version=1,proposal_sha256=plan['inputs']['proposal']['sha256'],plan_sha256=plan_sha,
        cache_sha256=reader.sha(p/'CACHE.json'),orders_sha256=reader.sha(p/'PRETRAIN_ORDERS.json'),frozen_sha256=reader.sha(p/'FROZEN.json'),
        common_availability_verified=True,baseline_local67_full_orders_unchanged=True,
        common_valid49_ranking_before_eligibility_verified=True,session_event_arithmetic_verified=True,
        diagnostic_start_marker_verified=True,diagnose_started_sha256=reader.sha(p/'DIAGNOSE_STARTED.json'),audit_started_utc=started.isoformat(),
        source_hashes={str(f):reader.sha(f) for f in source_files()},first_public_version_verified=False,
        limitations=['Repeatedly reused history; not pristine out of sample.',
        'Adjusted overnight includes dividend effects; original cached price vintage retained.',
        'No protected inventory rescan in this outcome-only audit.'])
"""+o[end:]
start=o.index('    reviewed=[',o.index('def synthetic_receipt'));end=o.index('\n    proposal=',start)
o=o[:start]+"""    reviewed=[ROOT/'work/monthly_session_cluster148_runner.py',ROOT/'work/monthly_session_cluster148_factor.py',ROOT/CONCEPT]"""+o[end:]
o=o.replace("asset_growth_feature_matrices_verified","session_cluster_feature_matrices_verified")
(root/'work/monthly_session_cluster148_outcome_independent.py').write_text(o)
print('Wrote two new148 audit sources; no data loaded.')
