"""Create new activity164 lifecycle sources without numerical research."""
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'work'
def renamed(value):
    value=value.replace('run163','activity164')
    value=re.sub(r'\b(162|163|1001|1004)\b',lambda m:{'162':'163','163':'164','1001':'1004','1004':'1007'}[m[0]],value)
    value=value.replace('PRE163_','PRE164_').replace('round163','round164').replace('163rounds','164rounds')
    value=value.replace('1004trials','1007trials').replace('1001trials','1004trials')
    value=value.replace('Expected162','Expected163').replace('Prior162','Prior163')
    value=value.replace('terminal-run','activity-to-variance').replace('terminal run','activity-to-variance')
    return value

def save(name,value):
    with (WORK/name).open('x') as f:f.write(value)

host=renamed((WORK/'run163_host.py').read_text())
host=host.replace('Fixed two-sided activity-to-variance experiment.','Fixed activity-to-variance experiment with matched variance control.')
host=host.replace("  deltas,audit=kernel.build_features(px,training_meta,events,calendar)",
"""  panel={'adj':px}
  for key in ('close','volume'):
   require(px.index.isin(backend.panel[key].index).all() and px.columns.isin(backend.panel[key].columns).all(),'Missing source axes '+key)
   panel[key]=backend.panel[key].loc[px.index,px.columns].copy()
  deltas,audit=kernel.build_features(panel,training_meta,events,calendar)""")
host=host.replace('kernel.build_features(px,training_meta,events,calendar,limit=n)','kernel.build_features(panel,training_meta,events,calendar,limit=n)')
host=host.replace("  self.write('STATE_VALUES.json',dict(raw_run=[x['raw_run']for x in audit['event_history']],ranks=[x['ranks']for x in audit['event_history']],common_support=[x['common_support']for x in audit['event_history']]))",
"  self.write('STATE_VALUES.json',{key:[x[key]for x in audit['event_history']]for key in ('raw_activity','raw_control','ranks_activity','ranks_control','common_support')})")
host=host.replace('fixed ten-block activity-to-variance','fixed partial-correlation activity-to-variance')
host=host.replace('fixed signed activity-to-variance contrast','fixed activity-to-variance contrast')
save('activity164_host.py',host)
save('activity164_validate_code.py',renamed((WORK/'run163_validate_code.py').read_text()))
tests=renamed((WORK/'test_run163_host.py').read_text())
tests=tests.replace('range(161)','range(162)').replace("[{'trials':35}]","[{'trials':32}]")
tests=tests.replace('run_negative','activity_risk').replace('run_positive','variance_control')
save('test_activity164_host.py',tests)
probe=renamed((WORK/'run163_source_probe.py').read_text())
probe=probe.replace('All211 finite positive source prices are required for each security; missing observations must not alter original eligibility.',
    'All212 adjusted prices and211 paired close/volume observations must be positive finite; current eligibility is unchanged; coverage is measured only after registration.')
probe=probe.replace('Both directions have exactly opposite deltas; this is a fixed continuation-versus-exhaustion hypothesis contrast, not an identified causal effect.',
    'Activity partial correlation and matched variance persistence use the same support; only the activity arm is promotable; causal association is not established.')
probe=probe.replace("    pins = {str(p):sha(p) for p in paths.values()}","""    pins = {str(p):sha(p) for p in paths.values()}
    source_plan=ROOT/'outputs/monthly_paired_target143/PLAN.json'
    pins[str(source_plan)]=sha(source_plan)
    raw=json.loads(source_plan.read_text())['raw_source_hashes']
    assert len(raw)==64
    for name,digest in raw.items():
        assert sha(Path(name))==digest
        pins[name]=digest""")
probe=probe.replace("    assert pins=={str(p):sha(p) for p in paths.values()}","    assert all(sha(Path(p))==h for p,h in pins.items())")
save('activity164_source_probe.py',probe)

proposal=dict(stage='prospective_design_before_real_features',round=164,
 name='formation_activity_to_future_variance_conditional_on_current_variance',
 hypothesis='Among unchanged long-momentum candidates, weaker historical transmission from activity changes to next-session squared returns may indicate more persistent monthly leadership. A variance-persistence control tests whether activity adds information beyond price volatility.',
 arms=['incumbent_local67','activity_risk','variance_control'],
 history='At month-end t use210 pairs s=t-252..t-43:212 adjusted closes t-253..t-42;211 paired source closes/volumes t-253..t-43. No recent42-session observations.',
 formula='r=diff(log(adj)); x=diff(log(source_close)+log(source_volume)); z=r[:-1]^2; y=r[1:]^2. raw_activity=partial_Pearson(x,y|z), raw_control=Pearson(z,y).',
 numerics='For each v=x,y,z divide by maxabs then center. Zero maximum or centeredL2<=1e-12*scaled_uncenteredL2 is unavailable. Unit-normalize each centered vector. Residualize unit x and unit y on unit z; residualL2<=1e-12 unavailable. Activity is normalized residual dot product; control is dot(unitz,unity). Clamp to[-1,1] only for excess<=64*float64eps; larger excursions fail.',
 missing='Any missing/nonpositive/nonfinite consumed source, numerical degeneracy or invalid either correlation makes both unavailable. No fill. Both arms use identical common support; original eligibility is not modified.',
 rank='Separately ascending average percentile minus0.5 for each raw correlation, on shared finite original49 pool before eligibility; initial2016-08-17 forced neutral.',
 delta='delta_activity=-0.04*rank_activity; delta_control=-0.04*rank_control. Maximum absolute score correction0.02. No parameter grid.',
 placement='OriginalLocal67 config4 pre-score plus one delta, then exactly one original episode adjustment. Preserve original27arrays, pool, eligibility and monthly fixed units.',
 selection='One first88 decision. Only activity_risk promotable. Versus incumbent: same-close CAGR gain>=0.005, Sharpe tolerance0.05, DD tolerance0.03, early53/recent35 CAGR tolerance0.025; costednextopen CAGR noninferior with same guards. Versus variance_control: same-close and costednextopen noninferior with same guards. Activity needs>=3 changed first88 holding sets and>=1eachera. Later34 diagnostic only. Control never promoted.',
 evaluation='All3streams,9paths: sameclose0,nextopen0,nextopen5bp/side;2016-08-17..2026-09-24. OriginalEW3/EW2/cash, no leverage, fixedintramonthunits. Independent5978cells366orders123prefixes.',
 novelty_scope='Bounded local mechanism review in activity164_novelty_review.json; reject volume-at-price duplication. No exhaustive novelty claim.',
 source_limitations=[
 'Estimated historicalTop32 plus assumedtheme17, not certified all-US PIT membership or confirmed authorETFlist.',
 'Later-retrieved price vintages and repeatedly reused development history; not pristine out-of-sample evidence.',
 'Paired sourceclose*sourcevolume is an inherited daily dollar-activity proxy with heterogeneous split conventions; not observed orderflow, investor identities or true shareturnover. No corporate-action certification added.',
 'Constant unit changes cancel in log differences, but historical vendor revisions, early joins and inconsistent time-varying units can affect results.',
 'Squared returns measure total close-to-close movement, not intraday realized variance or forecastable causal risk.',
 'Some earlier volume/GARCH sources absent locally; bounded audit cannot prove exhaustive novelty.',
 'Sameclose fills idealized; costed nextopen comparison mandatory.'],
 actual_features_computed=0,actual_orders_computed=0,actual_returns_computed=0,observer_writes=False)
save('activity164_proposal.json',json.dumps(proposal,indent=2,sort_keys=True)+'\n')
print('Created164 host/probe/validator/test/proposal; no numerical research.')
