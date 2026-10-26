"""Synthetic-only round187 source, eligibility and transform validation."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_eligible_core187_independent as audit


def original_fixture_inputs(panel,meta):
    px=panel['adj'];r=px.pct_change(fill_method=None);v=r.rolling(63).std(ddof=1)
    raw=px.shift(42)/px.shift(252)-1.;risk=(px.shift(42)/px.shift(294)-1.)/v.replace(0.,np.nan)
    raw=raw.replace([np.inf,-np.inf],np.nan);risk=risk.replace([np.inf,-np.inf],np.nan)
    calm=v.SPY*np.sqrt(252)<.1;good=px.notna().rolling(295).sum().eq(295)&px.gt(0)&raw.gt(0)
    cs=[];es=[];ss=[]
    for day,names in zip(meta['dates'],meta['symbols']):
        flag=bool(calm.loc[day]);ss.append(flag)
        z=(raw if flag else risk).loc[day].reindex(names)
        cs.append(z.rank(pct=True).fillna(.5).to_numpy()-.5)
        es.append(good.loc[day].reindex(names,fill_value=False).to_numpy(bool))
    state=np.asarray(ss,bool);cw=np.where(state,.5,.52)[:,None]
    original=dict(core=np.asarray(cs)*cw,core_weight=cw,state_calm10=state,eligible=np.asarray(es,bool))
    source=dict(core=original['core'].copy(),core_weight=cw.copy())
    return original,source


def fixture(width=7,count=3):
    index=pd.bdate_range('2010-01-04',periods=620);t=np.arange(len(index)-1)
    names=['S'+str(i) for i in range(width-1)]+['QQQ']
    def path(r):return 100.*np.cumprod(np.r_[1.,1.+r])
    frame=pd.DataFrame({s:path((.0007+j*.00009 if j<width-2 else -.001)+(.002+j*.0007)*np.sin(t*(.17+j*.013)))
                        for j,s in enumerate(names)},index=index)
    frame['SPY']=path(.0001+np.where(t<430,.004,.018)*np.sin(.31*t))
    dates=[str(index[i].date()) for i in (410,470,530)[:count]]
    meta=dict(dates=dates,symbols=[names[:] for _ in dates],etfs=['QQQ'])
    panel={'adj':frame};original,source=original_fixture_inputs(panel,meta)
    return panel,meta,index.strftime('%Y-%m-%d').tolist(),original,source


def route_fixture():
    rows=4;width=7
    u=np.tile(np.array([-.4,-.3,-.1,0.,.1,.3,.5]),(rows,1))
    eligible=np.tile([True,False,True,False,False,True,True],(rows,1))
    calm=np.array([True,False,True,False]);cw=np.where(calm,.5,.52)[:,None]
    old=dict(core=u*cw,core_weight=cw,state_calm10=calm,eligible=eligible)
    source=dict(core=old['core'].copy(),core_weight=cw.copy())
    values=dict(candidate=np.full_like(u,np.nan),control=np.full_like(u,np.nan),available=np.zeros_like(eligible))
    for i in range(rows):
        a,b,m,_=audit.eligible_oracle(u[i],np.ones(width,bool),eligible[i],initial=i==0)
        values['candidate'][i]=a;values['control'][i]=b;values['available'][i]=m
    return values,old,source


class OperatorTests(unittest.TestCase):
    def test_rank_and_affine_manual_unequal_spacing(self):
        u=np.array([-.4,-.3,-.1,0.,.1,.3,.5]);e=np.array([True,False,True,False,False,True,True])
        a,b,m,row=audit.eligible_oracle(u,np.ones(7,bool),e)
        expected=np.array([-.25,0.,.25,.5]);np.testing.assert_array_equal(a[e],expected)
        np.testing.assert_allclose(b[e],[-.25,0.,1./3,.5],atol=1e-16)
        self.assertFalse(row['exact_equivalent']);self.assertEqual(row['eligible_count'],4)
        self.assertTrue(np.isnan(a[~e]).all());np.testing.assert_array_equal(m,e)
        self.assertEqual(b[0],a[0]);self.assertEqual(b[-1],a[-1])

    def test_all_noop_branches_and_priority(self):
        for u,e,valid,initial,reason in (
            ([0.,0.,0.],[1,1,1],[0,1,1],True,'initial_allocation'),
            ([0.,.2,.3],[1,1,0],[0,1,1],False,'missing_eligible_core'),
            ([0.,.2,.3],[0,1,0],[1,1,1],False,'fewer_than_two_eligible'),
            ([0.,.2,.3],[1,1,1],[1,1,1],False,'all_eligible'),
            ([.1,.1,.3],[1,1,0],[1,1,1],False,'equal_eligible_core')):
            a,b,m,row=audit.eligible_oracle(np.array(u),np.array(valid,bool),np.array(e,bool),initial)
            self.assertEqual(row['reason'],reason);self.assertFalse(m.any());self.assertTrue(np.isnan(a).all());self.assertTrue(np.isnan(b).all())
            self.assertIsNone(row['affine_slope'])

    def test_missing_neutral_zero_is_not_validity(self):
        u=np.array([-.2,0.,.2,.4]);e=np.array([True,True,True,False])
        got=audit.eligible_oracle(u,np.array([True,False,True,True]),e)
        self.assertEqual(got[3]['reason'],'missing_eligible_core')
        got=audit.eligible_oracle(u,np.ones(4,bool),e)
        self.assertTrue(got[2].any())

    def test_ties_two_level_exactcopy_and_no_epsilon(self):
        u=np.array([-.2,-.2,.3,.4]);e=np.array([True,True,True,False])
        a,b,_,row=audit.eligible_oracle(u,np.ones(4,bool),e)
        np.testing.assert_array_equal(a,b);self.assertTrue(row['control_two_level_copy'])
        self.assertEqual(a[0],a[1]);self.assertEqual(a[2],.5)
        u=np.array([.1,np.nextafter(.1,.2),.2,.4]);a,b,_,row=audit.eligible_oracle(u,np.ones(4,bool),e)
        self.assertEqual(row['distinct_eligible_levels'],3);self.assertFalse(row['control_two_level_copy']);self.assertNotEqual(a[0],a[1])

    def test_pandas_midrank_identity(self):
        z=np.array([3.,np.nan,3.,1.,8.,np.inf,-2.])
        expected=pd.Series(z).replace([np.inf,-np.inf],np.nan).rank(method='average',pct=True).fillna(.5).to_numpy()-.5
        np.testing.assert_array_equal(audit.ranks(z),expected)


class SourceTests(unittest.TestCase):
    def test_full49_raw_source_parity_audit_no_mutation(self):
        args=fixture(49);before=copy.deepcopy(args);values,rec=audit.independent_features(*args)
        check=audit.verify_features(*args,values,rec)
        self.assertEqual(check['checked_cells'],147);self.assertFalse(values['available'][0].any())
        self.assertTrue(values['available'][1:].any());self.assertEqual(check['core_rank_reconstructions'],3)
        json.dumps(rec,allow_nan=False);json.dumps(check,allow_nan=False)
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj'])
        for d,b in zip(args[3:],before[3:]):
            for key in d:np.testing.assert_array_equal(d[key],b[key])

    def test_prefix_numeric_poison_and_unrequested_symbol(self):
        args=list(fixture());full,rec=audit.independent_features(*args)
        for n in range(4):
            small,sub=audit.independent_features(*args,limit=n)
            for k in full:np.testing.assert_array_equal(small[k],full[k][:n])
            self.assertEqual(sub['event_history'],rec['event_history'][:n])
        class Poison:
            def __float__(self):raise AssertionError('Future numeric read')
        args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>pd.Timestamp(args[1]['dates'][0]),:]=Poison()
        args[0]['adj']['UNREQUESTED']=Poison()
        for box in (args[3],args[4]):
            for key in ('core','core_weight'):
                box[key]=box[key].astype(object);box[key][1:]=Poison()
        args[4]['vol']=Poison();args[4]['state_highvol']=Poison();args[4]['vol_weight']=Poison()
        got,_=audit.independent_features(*args,limit=1)
        for k in full:np.testing.assert_array_equal(got[k],full[k][:1])
        args[0]['adj']=args[0]['adj'].loc[:args[1]['dates'][0]]
        got,_=audit.independent_features(*args,limit=1)
        for k in full:np.testing.assert_array_equal(got[k],full[k][:1])

    def test_calendar_gaps_no_fill_no_compress_and_precalendar_history(self):
        args=list(fixture());full,_=audit.independent_features(*args)
        args[2]=args[2][100:];got,_=audit.independent_features(*args)
        for k in full:np.testing.assert_array_equal(got[k],full[k])
        args=list(fixture());args[0]['adj']=args[0]['adj'].drop(index=args[0]['adj'].index[400])
        with self.assertRaisesRegex(ValueError,'parity'):audit.independent_features(*args)
        args=list(fixture());args[2].pop(400)
        with self.assertRaisesRegex(ValueError,'official calendar'):audit.independent_features(*args)

    def test_old_eligible_undefined_noncalm_core_whole_row_neutral(self):
        args=list(fixture());frame=args[0]['adj'];end=frame.index.get_loc(args[1]['dates'][1])
        frame.loc[frame.index[end-100]:,'S0']=float(frame['S0'].iloc[end-100])
        args[3],args[4]=original_fixture_inputs(args[0],args[1]);values,rec=audit.independent_features(*args)
        self.assertTrue(args[3]['eligible'][1,0]);self.assertFalse(args[3]['state_calm10'][1])
        self.assertEqual(rec['event_history'][1]['reason'],'missing_eligible_core');self.assertFalse(values['available'][1].any())

    def test_missing_history_has_original_eligibility_and_neutral_rank(self):
        args=list(fixture());day=args[1]['dates'][1];t=args[0]['adj'].index.get_loc(day)
        args[0]['adj'].iloc[t-100,0]=np.nan;args[3],args[4]=original_fixture_inputs(args[0],args[1])
        values,rec=audit.independent_features(*args)
        self.assertFalse(args[3]['eligible'][1,0]);self.assertFalse(values['available'][1,0]);self.assertTrue(values['available'][1,1:].any())
        # Unknown event identity must stay original neutral/ineligible, never deleted.
        args=list(fixture());args[0]['adj']=args[0]['adj'].drop(columns='S0');args[3],args[4]=original_fixture_inputs(args[0],args[1])
        values,rec=audit.independent_features(*args);self.assertFalse(values['available'][:,0].any());self.assertEqual(rec['event_history'][1]['core_rank'][0],0.)

    def test_state_core_eligibility_and_audit_mutations_rejected(self):
        args=fixture();values,rec=audit.independent_features(*args)
        for kind in ('core','eligible','calm'):
            changed=list(copy.deepcopy(args))
            if kind=='core':changed[3]['core'][1,0]+=.001;changed[4].pop('core')
            elif kind=='eligible':changed[3]['eligible'][1,0]=not changed[3]['eligible'][1,0]
            else:changed[3]['state_calm10'][1]=not changed[3]['state_calm10'][1]
            with self.assertRaises(ValueError):audit.independent_features(*changed)
        bad=copy.deepcopy(rec);bad['event_history'][1]['eligible_count']+=1
        with self.assertRaises(ValueError):audit.verify_features(*args,values,bad)
        bad=copy.deepcopy(values);j=np.flatnonzero(bad['available'][1])[0];bad['candidate'][1,j]+=.01
        with self.assertRaises(ValueError):audit.verify_features(*args,bad,rec)

    def test_invalid_price_schema_and_short_support(self):
        for bad in (0.,-1.,math.inf,'notprice'):
            args=list(fixture());args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].iloc[0,0]=bad
            with self.assertRaises(ValueError):audit.independent_features(*args)
        args=list(fixture());args[1]['dates']=[args[2][100]];args[1]['symbols']=args[1]['symbols'][:1]
        args[3],args[4]=original_fixture_inputs(args[0],args[1]);values,rec=audit.independent_features(*args)
        self.assertFalse(values['available'].any());self.assertFalse(args[3]['eligible'].any())

    def test_overflow_raw252_is_missing_before_original_eligibility(self):
        args=list(fixture());frame=args[0]['adj'];end=frame.index.get_loc(args[1]['dates'][1])
        frame['SPY']=100.  # Active calm core is precisely raw252 in this case.
        frame.iloc[end-252,0]=1e-308;frame.iloc[end-42,0]=1e308
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            args[3],args[4]=original_fixture_inputs(args[0],args[1])
            values,rec=audit.independent_features(*args)
        self.assertFalse(args[3]['eligible'][1,0]);self.assertFalse(values['available'][1,0])
        self.assertFalse(rec['event_history'][1]['core_valid'][0])
        self.assertIsNone(rec['event_history'][1]['raw_core'][0])


class DeltaTests(unittest.TestCase):
    def test_half_core_formula_prefix_broadcast_and_no_input_mutation(self):
        args=route_fixture();before=copy.deepcopy(args);full,rec=audit.independent_deltas(*args)
        check=audit.verify_deltas(*args,full,rec);self.assertTrue(check['half_core_replacement_checked'])
        v,old,_=args
        self.assertEqual(full['candidate'][1,0],.5*(old['core_weight'][1,0]*v['candidate'][1,0]-old['core'][1,0]))
        for n in range(5):
            got,_=audit.independent_deltas(*args,limit=n)
            for k in full:np.testing.assert_array_equal(got[k],full[k][:n])
        for mode in ('n','n1','nW'):
            vals,old,source=copy.deepcopy(args)
            for box in (old,source):
                if mode=='n':box['core_weight']=box['core_weight'][:,0]
                elif mode=='nW':box['core_weight']=np.repeat(box['core_weight'],7,axis=1)
            got,_=audit.independent_deltas(vals,old,source)
            for k in full:np.testing.assert_array_equal(got[k],full[k])
        for box,prior in zip(args,before):
            for k in box:np.testing.assert_array_equal(box[k],prior[k])

    def test_delta_numeric_tail_isolated(self):
        args=list(route_fixture());full,_=audit.independent_deltas(*args)
        class Poison:
            def __float__(self):raise AssertionError('Tail converted')
        for key in ('candidate','control'):
            args[0][key]=args[0][key].astype(object);args[0][key][2:]=Poison()
        for box in args[1:]:
            for key in ('core','core_weight'):
                box[key]=box[key].astype(object);box[key][2:]=Poison()
        got,_=audit.independent_deltas(*args,limit=2)
        for k in got:np.testing.assert_array_equal(got[k],full[k][:2])

    def test_nonzero_noop_wrong_fraction_and_eligibility_rejected(self):
        args=route_fixture();delta,rec=audit.independent_deltas(*args)
        for kind in ('first','ineligible','fraction','audit'):
            bad=copy.deepcopy(delta);detail=copy.deepcopy(rec)
            if kind=='first':bad['candidate'][0,0]=1e-30
            elif kind=='ineligible':bad['candidate'][1,1]=1e-30
            elif kind=='fraction':bad['candidate'][1,0]*=2.
            else:detail['core_weights'][1][0]+=.01
            with self.assertRaises(ValueError):audit.verify_deltas(*args,bad,detail)
        values,old,source=copy.deepcopy(args);values['available'][1,1]=True
        for k in ('candidate','control'):values[k][1,1]=0.
        with self.assertRaises(ValueError):audit.independent_deltas(values,old,source)

    def test_no_kernel_host_or_strategy_engine_import(self):
        text=Path(audit.__file__).read_text();tree=ast.parse(text)
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in n or 'host' in n for n in names));self.assertNotIn('__main__',text)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_monthly_upside_retention181_independent as helper
        with patch.object(helper,'audit',audit):cls.args=helper.account_fixture()

    def test_nine_paths45_slices(self):
        result=audit.audit_outcomes(*self.args)
        self.assertEqual((result['checked_paths'],result['checked_slices']),(9,45))

    def test_fill_slice_and_frozen_hash_tampering(self):
        with patch.object(audit,'OUTCOME_SHA','0'*64),self.assertRaises(ValueError):audit.audit_outcomes(*self.args)
        args=list(self.args);args[3]=copy.deepcopy(args[3]);args[3][audit.ARMS[1]]['same_close_zero']['TRADES'][0]['adjusted_units']+=1.
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args);args[4]=copy.deepcopy(args[4]);args[4][audit.ARMS[2]]['next_open_cost']['slices']['later34']['sharpe']=99.
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)


if __name__=='__main__':unittest.main()
