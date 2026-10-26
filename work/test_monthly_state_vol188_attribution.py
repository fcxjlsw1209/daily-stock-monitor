"""Synthetic attribution tests; never read actual188 numerical artifacts."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_state_vol188_attribution as a


def source_fixture(n=3,width=2):
    raw={'candidate':np.full((n,width),.2),'control':np.full((n,width),-.1),'available':np.ones((n,width),bool)}
    raw['available'][-1,-1]=False
    raw['candidate'][-1,-1]=raw['control'][-1,-1]=np.nan
    history=[];reasons={};refs=matched=0
    for i in range(n):
        cells=[]
        for j in range(width):
            ok=bool(raw['available'][i,j]);reason='available' if ok else 'current_volatility_unavailable'
            refs+=700;matched+=100
            reasons[reason]=reasons.get(reason,0)+1
            cells.append(dict(symbol='S'+str(j),date='row'+str(i),available=ok,current_calm=True,reference_count=700,matched_count=100,
                reason=reason,candidate=float(raw['candidate'][i,j]) if ok else None,control=float(raw['control'][i,j]) if ok else None))
        history.append(dict(date='row'+str(i),symbols=['S'+str(j) for j in range(width)],current_calm=True,cells=cells))
    f=dict(event_history=history,symbols_per_event=width,reference_observations=refs,matched_observations=matched,reason_counts=reasons,
        available_cells=int(raw['available'].sum()),unavailable_cells=int((~raw['available']).sum()),own_return_count=126,market_return_count=63,history_slots=756)
    t={}
    for arm,key in zip(a.ARMS[1:],('candidate','control')):
        t['transform_'+arm]=np.where(raw['available'],raw[key],0.)
        d=.01*t['transform_'+arm];d[0]=0.;t['delta_'+arm]=d
    return f,raw,t


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


def completion_fixture(root):
    out=root/'outputs/monthly_state_vol188';out.mkdir(parents=True)
    dates=['2016-08-17']+[str(x.date()) for x in pd.date_range('2016-08-31','2026-08-31',freq='ME')]
    ends=dates[1:]+['2026-09-24'];assert len(dates)==122
    meta=dict(dates=dates,ends=ends)
    orders={arm:[dict(date=d,end=e,symbols=['A','B','C']) for d,e in zip(dates,ends)] for arm in a.ARMS}
    changed=[1,20,60]
    for i in changed:orders[a.ARMS[1]][i]['symbols']=['A','B','D']
    f,v,t=source_fixture(122)
    for i,event in enumerate(f['event_history']):
        event['date']=dates[i]
        for cell in event['cells']:cell['date']=dates[i]
    prepared=out/'prepared';prepared.mkdir()
    np.savez(prepared/'FEATURES.npz',**v);np.savez(prepared/'TRANSFORMS_DELTAS.npz',**t)
    for name,value in [('META.json',meta),('ORDERS.json',orders),('FEATURE_AUDIT.json',f),('SOURCE_PARITY.json',{}),
        ('INDEPENDENT_PREPARE.json',dict(feature=dict(passed=True,reference_observations=f['reference_observations'],matched_observations=f['matched_observations']),delta=dict(passed=True)))]:write(prepared/name,value)
    plan=dict(input_sha256={});write(out/'PLAN.json',plan);ps=a.sha(out/'PLAN.json')
    prep=dict(status='COMPLETE',plan_sha256=ps,created_utc='2026-01-01T00:00:01+00:00',reference_observations=f['reference_observations'],matched_observations=f['matched_observations'],artifacts_sha256={p.name:a.sha(p) for p in prepared.iterdir()})
    write(prepared/'COMPLETION.json',prep)
    results=out/'results';results.mkdir();measures={};training={};indpaths={}
    bounds=dict(full122=(dates[0],ends[-1]),first88=(dates[0],ends[87]),early53=(dates[0],ends[52]),recent35=(dates[53],ends[87]),later34=(dates[88],ends[-1]))
    for ai,arm in enumerate(a.ARMS):
        measures[arm]={};training[arm]={};indpaths[arm]={}
        for path in a.PATHS:
            nav=[100000.]
            for i in range(122):nav.append(nav[-1]*(1.012+.007*math.sin(i*.71)+(ai==1)*.001))
            rows=[dict(date=d,equity=e) for d,e in zip(dates+[ends[-1]],nav)]
            slices={}
            for key,(start,end) in bounds.items():
                r=[x for x in rows if start<=x['date']<=end]
                slices[key]=a.metrics(r,100000. if key=='full122' else r[0]['equity'])
            measures[arm][path]=dict(slices=slices)
            indpaths[arm][path]=dict(slices=slices)
            if path in a.SHOW_PATHS:training[arm][path]={short:slices[full] for short,full in [('training','first88'),('early','early53'),('recent','recent35')]}
            for kind,value in [('DAILY',rows),('TRADES',[]),('HOLDINGS',[]),('DAILY_POSITIONS',[])]:write(results/(arm+'_'+path+'_'+kind+'.json'),value)
    guards={}
    for key,path,ref,gain in [('close_vs_incumbent','same_close_zero',a.ARMS[0],.005),('open_vs_incumbent','next_open_cost',a.ARMS[0],0.),('close_vs_unconditional','same_close_zero',a.ARMS[2],0.),('open_vs_unconditional','next_open_cost',a.ARMS[2],0.)]:
        guards[key]=all(a.guard(training[a.ARMS[1]][path],training[ref][path],gain).values())
    guards.update(changed_support=True,changed_support_vs_control=True)
    admitted=all(guards.values());selection=dict(first88_only=True,later34_used=False,guards=guards,admitted=admitted,primary=a.ARMS[1] if admitted else a.ARMS[0],changed_months=changed,changed_months_vs_control=changed)
    write(out/'selected/TRAIN_RESULTS.json',training)
    frozen=dict(plan_sha256=ps,prepared_sha256=a.sha(prepared/'COMPLETION.json'),orders_sha256=a.sha(prepared/'ORDERS.json'),training_sha256=a.sha(out/'selected/TRAIN_RESULTS.json'),selection=selection,created_utc='2026-01-01T00:00:02+00:00')
    write(out/'selected/FROZEN.json',frozen)
    write(results/'RESULTS.json',dict(arms=measures,selection=selection))
    write(results/'INDEPENDENT_OUTCOME.json',dict(passed=True,checked_paths=9,checked_slices=45,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths=indpaths))
    complete=dict(status='COMPLETE',plan_sha256=ps,frozen_sha256=a.sha(out/'selected/FROZEN.json'),created_utc='2026-01-01T00:00:03+00:00',artifacts_sha256={p.name:a.sha(p) for p in results.iterdir()})
    write(results/'COMPLETION.json',complete);rs=a.sha(results/'COMPLETION.json')
    closed=dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',plan_sha256=ps,result_completion_sha256=rs,created_utc='2026-01-01T00:00:04+00:00',observers_modified=False,reference_observations=f['reference_observations'],matched_observations=f['matched_observations'])
    write(out/'COMPLETION.json',closed)
    return out,ps,rs,a.sha(out/'COMPLETION.json')


class AttributionTests(unittest.TestCase):
    def test_growth_identity_and_cancelling_contributions(self):
        for factors in ([1.1,.8,1.2],[2.,.5],[1.,1.]):
            d=a.decompose(factors,.5,10.)
            self.assertAlmostEqual(d['terminal_wealth_ratio'],math.prod(factors))
            self.assertAlmostEqual(sum(d['cagr_contributions']),d['cagr_difference'])
            self.assertAlmostEqual(d['cagr_difference'],1.5*(math.prod(factors)**.1-1))
        with self.assertRaises(ValueError):a.decompose([0.],.5,10.)

    def test_concentration_signs_no_cancellation_denominator(self):
        x=a.concentration([.1,-.1,.2],['a','b','c'])
        self.assertAlmostEqual(x['absolute_log_sum'],.4)
        self.assertAlmostEqual(x['top_absolute_log_share']['1'],.5)
        self.assertAlmostEqual(x['absolute_log_hhi'],.375)
        self.assertEqual(x['worst_months'][0]['date'],'b')
        z=a.concentration([0.],['a']);self.assertEqual(z['top_absolute_log_share']['1'],0.);self.assertIsNone(z['effective_month_count'])

    def test_support_and_delta_evidence(self):
        f,v,t=source_fixture();d=a.support_diagnostics(f,v,t)
        self.assertEqual(d['available_cells'],5);self.assertEqual(d['reference_observations'],4200)
        self.assertEqual(d['max_absolute_delta'][a.ARMS[1]],.002)
        bad=copy.deepcopy(t);bad['delta_'+a.ARMS[1]][0,0]=.002
        with self.assertRaises(ValueError):a.support_diagnostics(f,v,bad)
        bad=copy.deepcopy(f);bad['event_history'][0]['current_calm']=None
        with self.assertRaises(ValueError):a.support_diagnostics(bad,v,t)
        bad=copy.deepcopy(f);bad['event_history'][0]['cells'][0]['matched_count']=62
        with self.assertRaises(ValueError):a.support_diagnostics(bad,v,t)

    def test_explicit_root_and_result_required_before_io(self):
        for args in [(None,None,None),('a'*64,'b'*64,None),('a'*64,'b'*64,'not a digest')]:
            with self.assertRaisesRegex(ValueError,'explicit valid SHA-256'):a.main(*args)

    def test_complete_synthetic_chain_report_and_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out,ps,rs,cs=completion_fixture(root)
            script=root/'script.py';script.write_text('synthetic attribution source binding')
            test=root/'test.py';test.write_text('synthetic attribution test binding')
            with patch.multiple(a,ROOT=root,OUT=out,DEST=out/'attribution',TEST_FILE=test,__file__=str(script)):
                with contextlib.redirect_stdout(io.StringIO()):a.main(ps,rs,cs)
                report=a.read(out/'attribution/REPORT.json');receipt=a.read(out/'attribution/COMPLETION.json')
                self.assertEqual(report['verification']['root_completion_sha256'],cs)
                self.assertEqual(report['verification']['metric_slices_independently_recomputed'],45)
                self.assertEqual(len(report['all122_periods']),122)
                self.assertEqual(set(report['log_concentration']),set(a.PATHS))
                self.assertEqual(receipt['result_completion_sha256'],rs)
                self.assertTrue(report['unchanged_holdings_return_effects'])
                self.assertIn(str(script),report['input_sha256'])
                with self.assertRaisesRegex(ValueError,'write-once'):a.main(ps,rs,cs)

    def test_tampered_bound_artifact_fails_before_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();out,ps,rs,cs=completion_fixture(root)
            script=root/'script.py';script.write_text('synthetic source');test=root/'test.py';test.write_text('test')
            target=out/'results'/f'{a.ARMS[1]}_same_close_zero_DAILY.json';target.write_text('[]')
            with patch.multiple(a,ROOT=root,OUT=out,DEST=out/'attribution',TEST_FILE=test,__file__=str(script)):
                with self.assertRaisesRegex(ValueError,'Input hash mismatch'):a.main(ps,rs,cs)
            self.assertFalse((out/'attribution').exists())


if __name__=='__main__':unittest.main(verbosity=2)
