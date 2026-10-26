"""Synthetic tests only: original rolling, symmetric comparisons and funded deltas."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_core_peer184_independent as audit


def original_fixture_inputs(panel,meta):
    px=panel['adj'];r=px.pct_change(fill_method=None);v63=r.rolling(63).std();v126=r.rolling(126).std()
    c=(v63.SPY*np.sqrt(252)<.1);h=(v63.SPY>v63.SPY.rolling(252).median())
    raw=(px.shift(42)/px.shift(252)-1.);risk=(px.shift(42)/px.shift(294)-1.)/v63.replace(0.,np.nan)
    core,vol,calm,high=[],[],[],[]
    for day,names in zip(meta['dates'],meta['symbols']):
        calm.append(bool(c.loc[day]));high.append(bool(h.loc[day]))
        rv=(raw if calm[-1] else risk).loc[day,names].replace([np.inf,-np.inf],np.nan)
        core.append(rv.rank(pct=True).fillna(.5).to_numpy()-.5)
        vol.append(v126.loc[day,names].replace([np.inf,-np.inf],np.nan).rank(pct=True).fillna(.5).to_numpy()-.5)
    calm=np.array(calm,bool);high=np.array(high,bool);cw=np.where(calm,.5,.52)[:,None]
    core=np.array(core)*cw;vol=np.array(vol);shape=core.shape
    original=dict(core=core,core_weight=cw,state_calm10=calm,state_weak_narrow=np.arange(len(calm))%2==1,
                  extension_available=np.ones(shape,bool),forecast_available=np.ones(shape,bool),eligible=np.ones(shape,bool),
                  rank_log_excess=np.zeros(shape))
    source=dict(vol=vol,vol_weight=np.where(calm,.2,.219375*(1-np.where(high,.10,0.)/.48))[:,None],state_highvol=high,
                core=core.copy(),core_weight=cw.copy())
    return original,source


def fixture(width=5,count=3):
    index=pd.bdate_range('2010-01-04',periods=610);t=np.arange(len(index)-1)
    names=['S'+str(i) for i in range(width-1)]+['QQQ']
    def path(r): return 100.*np.cumprod(np.r_[1.,1.+r])
    frame=pd.DataFrame({s:path(.0002+j*.00002+(.003+j*.001)*np.sin(t*(.17+j*.013))) for j,s in enumerate(names)},index=index)
    frame['SPY']=path(.0001+np.where(t<430,.004,.018)*np.sin(.31*t))
    dates=[str(index[i].date()) for i in (410,470,530)[:count]]
    meta=dict(dates=dates,symbols=[names[:] for _ in dates],etfs=['QQQ'])
    panel={'adj':frame};original,source=original_fixture_inputs(panel,meta)
    return panel,meta,index.strftime('%Y-%m-%d').tolist(),original,source


class PairTests(unittest.TestCase):
    def test_manual_symmetric_pair_and_mass_control(self):
        vol=np.array([1.,3.,2.,4.]);core=np.arange(4.);u=np.array([0.,.1,.15,.5]);v=audit.ranks(vol)
        candidate,control,mask,row=audit.peer_oracle(vol,core,u,v)
        w=np.array([[0.,.5,.25,0.],[.5,0.,.75,0.],[.25,.75,0.,0.],[0.,0.,0.,0.]])
        sign=np.sign(vol[:,None]-vol[None,:]);expected=.5/4+np.sum(w*sign,axis=1)/(2*4)
        np.testing.assert_allclose(candidate,expected,atol=2e-17)
        mean=w.sum()/12
        np.testing.assert_allclose(control,.5/4+mean*(v-.5/4),atol=2e-17)
        self.assertTrue(mask.all());self.assertAlmostEqual(row['unordered_pair_weight_sum'],1.5)
        self.assertEqual(row['nonzero_unordered_pair_comparisons'],3)
        self.assertAlmostEqual(sum(candidate),sum(v));self.assertAlmostEqual(sum(control),sum(v))

    def test_exact_identity_branches_not_epsilon_snap(self):
        for vol,u,which in (([1.,2.,3.],[.1,.1,.1],'all_core_ties'),([2.,2.,2.],[-.3,0.,.3],'all_vol_ties')):
            vol=np.array(vol);v=audit.ranks(vol)
            got=audit.peer_oracle(vol,np.ones(3),np.array(u),v)
            for target in got[:2]: np.testing.assert_array_equal(target,v)
            self.assertEqual(got[3]['identity_branch'],which)
        vol=np.array([2.,2.,np.nextafter(2.,3.)]);v=audit.ranks(vol)
        row=audit.peer_oracle(vol,np.arange(3.),np.array([-.3,0.,.3]),v)[3]
        self.assertIsNone(row['identity_branch'])

    def test_missing_core_preserves_full_vol_domain(self):
        vol=np.array([1.,2.,3.,np.nan]);v=audit.ranks(vol);u=np.array([-.3,0.,.2,.5])
        for raw,reason in ((np.array([1.,np.nan,3.,4.]),'missing_core_in_vol_domain'),):
            candidate,control,valid,row=audit.peer_oracle(vol,raw,u,v)
            self.assertEqual(row['domain_count'],3);self.assertFalse(valid.any());self.assertTrue(np.isnan(candidate).all())
            self.assertEqual(row['reason'],reason);self.assertEqual(row['unordered_pair_comparisons'],0)
        candidate,control,valid,row=audit.peer_oracle(vol,np.array([1.,2.,3.,np.nan]),u,v)
        np.testing.assert_array_equal(valid,[True,True,True,False])
        vol[:3]=[np.nan,np.nan,3.]
        self.assertEqual(audit.peer_oracle(vol,np.ones(4),u,audit.ranks(vol))[3]['reason'],'insufficient_vol_domain')

    def test_bandwidth_continuity_no_bins_and_no_self_comparison(self):
        vol=np.array([1.,2.]);v=audit.ranks(vol)
        at=audit.peer_oracle(vol,np.ones(2),np.array([0.,.2]),v)[3]
        near=audit.peer_oracle(vol,np.ones(2),np.array([0.,np.nextafter(.2,0.)]),v)[3]
        self.assertEqual(at['unordered_pair_comparisons'],1);self.assertEqual(at['nonzero_unordered_pair_comparisons'],0)
        self.assertGreater(near['unordered_pair_weight_sum'],0.)
        self.assertLess(near['unordered_pair_weight_sum'],1e-14)


class SourceTests(unittest.TestCase):
    def test_original_rolling_all49_and_receipt(self):
        args=fixture(49);before=copy.deepcopy(args)
        values,record=audit.independent_features(*args)
        receipt=audit.verify_features(*args,values,record)
        self.assertEqual(values['available'].shape,(3,49));self.assertTrue(values['available'].all())
        self.assertEqual(receipt['unordered_pair_comparisons'],3*49*48//2)
        self.assertTrue(receipt['original_market_states_checked'])
        for x,y in zip(args[0].values(),before[0].values()): pd.testing.assert_frame_equal(x,y)
        for box,old in zip(args[3:],before[3:]):
            for key in box: np.testing.assert_array_equal(box[key],old[key])
        json.dumps(record,allow_nan=False);json.dumps(receipt,allow_nan=False)

    def test_prefixes_future_poison_and_truncated_price_panel(self):
        args=list(fixture());full,rec=audit.independent_features(*args)
        for n in range(4):
            got,sub=audit.independent_features(*args,limit=n)
            for key in full: np.testing.assert_array_equal(got[key],full[key][:n])
            self.assertEqual(sub['event_history'],rec['event_history'][:n])
        class Poison:
            def __float__(self): raise AssertionError('Future price converted')
        frame=args[0]['adj'].astype(object);frame.loc[frame.index>pd.Timestamp(args[1]['dates'][0]),:]=Poison()
        args[0]={'adj':frame};got,_=audit.independent_features(*args,limit=1)
        for key in full: np.testing.assert_array_equal(got[key],full[key][:1])
        args[0]['adj']['UNREQUESTED']=Poison()
        got,_=audit.independent_features(*args,limit=1)
        for key in full: np.testing.assert_array_equal(got[key],full[key][:1])
        args[0]={'adj':frame.loc[:args[1]['dates'][0]]};got,_=audit.independent_features(*args,limit=1)
        for key in full: np.testing.assert_array_equal(got[key],full[key][:1])

    def test_preserves_precalendar_history(self):
        args=list(fixture());full,record=audit.independent_features(*args)
        args[2]=args[2][100:]
        got,short=audit.independent_features(*args)
        for key in full: np.testing.assert_array_equal(got[key],full[key])
        self.assertEqual(short['event_history'][0]['session_position'],record['event_history'][0]['session_position']-100)

    def test_no_gap_compression_and_nonofficial_dates(self):
        args=list(fixture());args[0]['adj']=args[0]['adj'].drop(index=args[0]['adj'].index[400])
        with self.assertRaises(ValueError): audit.independent_features(*args)
        args=list(fixture());holiday=args[2].pop(400)
        with self.assertRaisesRegex(ValueError,'official calendar'): audit.independent_features(*args)
        args[0]['adj']=args[0]['adj'].drop(index=holiday);args[3],args[4]=original_fixture_inputs(args[0],args[1])
        self.assertTrue(audit.independent_features(*args)[0]['available'].all())

    def test_missing_core_row_fallback_and_vol_domain_exclusion(self):
        args=list(fixture());day=args[1]['dates'][0];t=args[0]['adj'].index.get_loc(day)
        horizon=252 if args[3]['state_calm10'][0] else 294
        args[0]['adj'].iloc[t-horizon,0]=np.nan;args[3],args[4]=original_fixture_inputs(args[0],args[1])
        values,record=audit.independent_features(*args)
        self.assertFalse(values['available'][0].any());self.assertEqual(record['event_history'][0]['reason'],'missing_core_in_vol_domain')
        args=list(fixture());t=args[0]['adj'].index.get_loc(args[1]['dates'][0]);args[0]['adj'].iloc[t-10,0]=np.nan
        args[3],args[4]=original_fixture_inputs(args[0],args[1]);values,record=audit.independent_features(*args)
        self.assertFalse(values['available'][0,0]);self.assertTrue(values['available'][0,1:].all())
        self.assertEqual(record['event_history'][0]['domain_count'],4)

    def test_zero_vol_valid_without_new_epsilon(self):
        args=list(fixture());args[0]['adj']['S0']=100.;args[3],args[4]=original_fixture_inputs(args[0],args[1])
        values,record=audit.independent_features(*args)
        self.assertEqual(record['event_history'][0]['raw_vol'][0],0.)
        self.assertTrue(values['available'][0,0]) # calm raw core does not divide by zero vol
        self.assertFalse(values['available'][1].any()) # noncalm original zero63 denominator has no core

    def test_raw_bad_prices_and_nonfinite_state_source_fail(self):
        for bad in (0.,-1.,math.inf,'bad'):
            args=list(fixture());args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].iloc[0,0]=bad
            with self.assertRaises(ValueError): audit.independent_features(*args)
        for where,key in ((3,'state_calm10'),(4,'state_highvol')):
            args=list(fixture());args[where][key][0]=not args[where][key][0]
            with self.assertRaises(ValueError): audit.independent_features(*args)

    def test_rank_audit_and_identity_tamper_rejected(self):
        args=fixture();values,record=audit.independent_features(*args)
        bad=copy.deepcopy(record);bad['event_history'][0]['neighbor_weight_sums'][0]+=.1
        with self.assertRaises(ValueError): audit.verify_features(*args,values,bad)
        bad=copy.deepcopy(values);bad['candidate'][1,0]+=.01
        with self.assertRaises(ValueError): audit.verify_features(*args,bad,record)
        args=list(fixture());args[4]['vol'][1,0]=np.nextafter(args[4]['vol'][1,0],math.inf)
        with self.assertRaisesRegex(ValueError,'rank parity'): audit.independent_features(*args)


def seven_states():
    calm=np.array([True,True,True,False,False,False,False]);weak=np.array([False,False,True,False,True,False,True])
    high=np.array([False,False,False,False,False,True,True]);rows=7;width=3
    old=dict(core=np.zeros((rows,width)),core_weight=np.where(calm,.5,.52)[:,None],state_calm10=calm,state_weak_narrow=weak,
             extension_available=np.zeros((rows,width),bool),forecast_available=np.ones((rows,width),bool),
             eligible=np.ones((rows,width),bool),rank_log_excess=np.zeros((rows,width)))
    old['extension_available'][1]=True
    source=dict(vol=np.tile([-.2,.1,.4],(rows,1)),vol_weight=np.where(calm,.2,.219375*(1-np.where(high,.1,0.)/.48))[:,None],state_highvol=high)
    values=dict(candidate=np.tile([.1,.2,.3],(rows,1)),control=np.tile([.2,.2,.2],(rows,1)),available=np.ones((rows,width),bool))
    return values,old,source


class DeltaTests(unittest.TestCase):
    def test_seven_effective_coefficients_and_half_replacement(self):
        args=seven_states();delta,record=audit.independent_deltas(*args)
        expected=[.19,.186,.17,.219375,.1965234375,.173671875,.1555810546875]
        np.testing.assert_allclose(np.array(record['effective_vol_weights'])[:,0],expected,atol=1e-16,rtol=0)
        self.assertTrue((delta['candidate'][0]==0.).all())
        self.assertAlmostEqual(delta['candidate'][1,0],.5*.186*(.1-(-.2)))
        receipt=audit.verify_deltas(*args,delta,record)
        self.assertTrue(receipt['half_slot_replacement_checked']);self.assertEqual(receipt['episode_applications'],0)

    def test_scope_masks_and_weak_calm_not_double_scope(self):
        for key in ('extension_available','forecast_available','eligible','rank_log_excess'):
            args=list(seven_states());args[1][key][1,0]=np.nan if key=='rank_log_excess' else False
            _,record=audit.independent_deltas(*args)
            self.assertFalse(record['scope_active'][1][0]);self.assertAlmostEqual(record['effective_vol_weights'][1][0],.19)
        args=list(seven_states());args[1]['extension_available'][2]=True
        _,record=audit.independent_deltas(*args)
        self.assertFalse(any(record['scope_active'][2]));self.assertAlmostEqual(record['effective_vol_weights'][2][0],.17)

    def test_broadcast_prefix_neutral_identity_and_no_mutation(self):
        args=seven_states();before=copy.deepcopy(args);full,rec=audit.independent_deltas(*args)
        for n in range(8):
            small,sub=audit.independent_deltas(*args,limit=n)
            for key in full: np.testing.assert_array_equal(small[key],full[key][:n])
        for box,old in zip(args,before):
            for key in box: np.testing.assert_array_equal(box[key],old[key])
        for shape in ('n','n1','nW'):
            vals,old,src=copy.deepcopy(args)
            for box,key in ((old,'core_weight'),(src,'vol_weight')):
                if shape=='n':box[key]=box[key][:,0]
                elif shape=='nW':box[key]=np.repeat(box[key],3,axis=1)
            got,_=audit.independent_deltas(vals,old,src)
            for key in full:np.testing.assert_array_equal(got[key],full[key])
        vals,old,src=copy.deepcopy(args);vals['available'][2,0]=False
        for key in ('candidate','control'):vals[key][2,0]=np.nan;vals[key][3]=src['vol'][3]
        delta,rec=audit.independent_deltas(vals,old,src)
        for key in delta:self.assertEqual(delta[key][2,0],0.);self.assertTrue((delta[key][3]==0.).all())

    def test_invalid_inputs_and_tampered_delta(self):
        args=seven_states();delta,rec=audit.independent_deltas(*args)
        for mutation in ('fraction','first','identity','audit'):
            values,old,src=copy.deepcopy(args);bad=copy.deepcopy(delta);record=copy.deepcopy(rec)
            if mutation=='fraction':bad['candidate'][1,0]*=2.
            elif mutation=='first':bad['candidate'][0,0]=1e-30
            elif mutation=='identity':
                values['candidate'][1,0]=src['vol'][1,0];bad,record=audit.independent_deltas(values,old,src);bad['candidate'][1,0]=1e-30
            else:record['effective_vol_weights'][1][0]+=.01
            with self.assertRaises(ValueError):audit.verify_deltas(values,old,src,bad,record)
        for key in ('core_weight','state_weak_narrow'):
            values,old,src=copy.deepcopy(args);old[key]=np.ones((7,2))
            with self.assertRaises(ValueError):audit.independent_deltas(values,old,src)

    def test_no_production_imports_or_entrypoint(self):
        text=Path(audit.__file__).read_text();tree=ast.parse(text)
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in name or 'host' in name for name in names));self.assertNotIn('__main__',text)


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_monthly_upside_retention181_independent as previous_tests
        with patch.object(previous_tests,'audit',audit):cls.args=previous_tests.account_fixture()

    def test_nine_paths45_slices(self):
        result=audit.audit_outcomes(*self.args)
        self.assertEqual((result['checked_paths'],result['checked_slices']),(9,45))

    def test_fill_slice_and_hash_rejection(self):
        with patch.object(audit,'OUTCOME_SHA','0'*64),self.assertRaises(ValueError):audit.audit_outcomes(*self.args)
        args=list(self.args);args[3]=copy.deepcopy(args[3]);args[3][audit.ARMS[1]]['same_close_zero']['TRADES'][0]['adjusted_units']+=1.
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args);args[4]=copy.deepcopy(args[4]);args[4][audit.ARMS[2]]['next_open_cost']['slices']['later34']['sharpe']=99.
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)


class ProductionCrossTests(unittest.TestCase):
    def test_both_fixtures_prefixes_and_missing_boundary_cases(self):
        import monthly_core_peer184_kernel as production
        from test_monthly_core_peer184_kernel import fixture as other_fixture
        for factory in (fixture,other_fixture):
            args=list(factory())
            values,record=production.build_features(*args)
            self.assertTrue(audit.verify_features(*args,values,record)['passed'])
            deltas,delta_record=production.build_deltas(values,args[3],args[4])
            self.assertTrue(audit.verify_deltas(values,args[3],args[4],deltas,delta_record)['passed'])
            for n in range(len(args[1]['dates'])+1):
                got,sub=production.build_features(*args,limit=n)
                wanted,ind=audit.independent_features(*args,limit=n)
                for key in got: audit.close(got[key],wanted[key],key)
                audit.record_check(sub,ind)
            for alteration in ('missing_core','missing_vol','all_ties','zero','early_calendar'):
                changed=list(copy.deepcopy(factory()));frame=changed[0]['adj']
                t=frame.index.get_loc(changed[1]['dates'][0]);name=changed[1]['symbols'][0][0]
                if alteration=='missing_core':frame.loc[frame.index[t-(252 if changed[3]['state_calm10'][0] else 294)],name]=np.nan
                elif alteration=='missing_vol':frame.loc[frame.index[t-10],name]=np.nan
                elif alteration=='all_ties':
                    for symbol in changed[1]['symbols'][0]:frame[symbol]=frame[name]
                elif alteration=='zero':frame[name]=100.
                else:changed[2]=changed[2][50:]
                changed[3],changed[4]=original_fixture_inputs(changed[0],changed[1])
                got,sub=production.build_features(*changed)
                self.assertTrue(audit.verify_features(*changed,got,sub)['passed'])


if __name__=='__main__':
    unittest.main()
