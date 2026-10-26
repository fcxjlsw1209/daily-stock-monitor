"""Synthetic-only source, funding identity, prefix and four-arm ledger tests."""
import ast
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import monthly_aux_reallocation186_independent as audit


def refresh(panel,meta):
    from test_monthly_core_peer184_independent import original_fixture_inputs
    original,source=original_fixture_inputs(panel,meta)
    shape=original['core'].shape
    support=.4*source['vol']+.6*np.tile(np.linspace(-.4,.4,shape[1]),(shape[0],1))
    original['baseline']=original['core']+(1-original['core_weight'])*support
    return original,source


def fixture(width=5,count=3):
    from test_monthly_core_peer184_independent import fixture as prior_fixture
    panel,meta,calendar,_,_=prior_fixture(width,count)
    original,source=refresh(panel,meta)
    return panel,meta,calendar,original,source


def delta_fixture():
    panel,meta,calendar,old,src=fixture()
    b,c,k,v=old['baseline'],old['core'],old['core_weight'],src['vol']
    return dict(vol_rank=v.copy(),unit_delta=(b-c)-(1-k)*v),old,src


class AllocationTests(unittest.TestCase):
    def test_seven_regime_vol_coefficients_remain_positive(self):
        cases=((True,False,False,False),(True,False,False,True),(True,True,False,False),
               (False,False,False,False),(False,True,False,False),
               (False,False,True,False),(False,True,True,False))
        after=[];transfers=[]
        for calm,weak,high,scope in cases:
            core_weight=.5 if calm else .52
            vol_weight=.2 if calm else .219375*(1-(.1 if high else 0.)/.48)
            effective=vol_weight*(1-(.025*calm+.05*weak+.01*scope)/(1-core_weight))
            transfer=.1*(1-core_weight-vol_weight)
            self.assertGreater(effective-transfer,0.)
            after.append(effective-transfer);transfers.append(transfer)
        self.assertAlmostEqual(min(after),.1249482421875,places=15)
        self.assertAlmostEqual(max(transfers),.0306328125,places=15)

    def test_reverse_is_exact_opposite_of_prior_direction(self):
        values,original,source=delta_fixture()
        b,c,k,v=original['baseline'],original['core'],original['core_weight'],source['vol']
        prior=(1-k)*v-(b-c)
        np.testing.assert_array_equal(values['unit_delta'],-prior)
        got,record=audit.independent_deltas(values,original,source)
        self.assertEqual(record['direction'],'vol_to_nonvol')
        receipt=audit.verify_deltas(values,original,source,got,record)
        self.assertTrue(receipt['reverse_budget_direction_checked'])
        bad=copy.deepcopy(values);bad['unit_delta']=-bad['unit_delta']
        with self.assertRaisesRegex(ValueError,'formula'):audit.independent_deltas(bad,original,source)
        wrong=copy.deepcopy(record);wrong['direction']='nonvol_to_vol'
        with self.assertRaises(ValueError):audit.verify_deltas(values,original,source,got,wrong)

    def test_scalar_funding_identity_no_core_change(self):
        for cw in (.5,.52):
            for wv in (.2,.219375,.219375*(1-.1/.48)):
                v=.3;f=-.2;core=-.13
                b=core+wv*v+(1-cw-wv)*f
                unit=audit.scalar_unit(b,core,cw,v)
                self.assertAlmostEqual(unit,(1-cw-wv)*(f-v),places=15)
                for dose in audit.LAMBDAS.values():
                    after=b+dose*unit
                    expected=core+(wv-dose*(1-cw-wv))*v+(1+dose)*(1-cw-wv)*f
                    self.assertAlmostEqual(after,expected,places=15)

    def test_three_exact_doses_initial_only_zero(self):
        args=delta_fixture();got,record=audit.independent_deltas(*args)
        for arm,dose in audit.LAMBDAS.items():
            np.testing.assert_array_equal(got[arm][1:],dose*args[0]['unit_delta'][1:])
            self.assertTrue((got[arm][0]==0.).all())
        self.assertTrue(np.any(args[0]['unit_delta'][0]!=0.))
        receipt=audit.verify_deltas(*args,got,record)
        self.assertEqual(receipt['lambdas'],audit.LAMBDAS);self.assertEqual(receipt['episode_applications'],0)

    def test_missing_neutral_is_not_new_no_correction_mask(self):
        values,old,src=copy.deepcopy(delta_fixture());src['vol'][:,0]=0.;values['vol_rank'][:,0]=0.
        old['baseline'][:,0]=old['core'][:,0]+.1
        values['unit_delta'][:,0]=old['baseline'][:,0]-old['core'][:,0]
        got,_=audit.independent_deltas(values,old,src)
        for arm,dose in audit.LAMBDAS.items():self.assertEqual(got[arm][1,0],dose*values['unit_delta'][1,0])

    def test_no_eligibility_or_model_filter(self):
        args=delta_fixture();full,_=audit.independent_deltas(*args)
        for key in ('eligible','extension_available','forecast_available'):
            changed=copy.deepcopy(args);changed[1][key][:]=False
            got,_=audit.independent_deltas(*changed)
            for arm in got:np.testing.assert_array_equal(got[arm],full[arm])

    def test_prefixes_broadcast_and_no_mutation(self):
        args=delta_fixture();before=copy.deepcopy(args);full,record=audit.independent_deltas(*args)
        for n in range(4):
            small,_=audit.independent_deltas(*args,limit=n)
            for arm in small:np.testing.assert_array_equal(small[arm],full[arm][:n])
        for shape in ('n','nW'):
            changed=copy.deepcopy(args)
            for box,key in ((changed[1],'core_weight'),(changed[2],'vol_weight'),(changed[2],'core_weight')):
                box[key]=box[key][:,0] if shape=='n' else np.repeat(box[key],5,axis=1)
            got,_=audit.independent_deltas(*changed)
            for arm in got:np.testing.assert_array_equal(got[arm],full[arm])
        for box,old in zip(args,before):
            for key in box:np.testing.assert_array_equal(box[key],old[key])

    def test_numeric_array_future_padding_is_not_converted(self):
        import monthly_aux_reallocation186_kernel as production
        class Poison:
            def __float__(self):raise AssertionError('Future numeric array converted')
        args=list(fixture());expected,record=audit.independent_features(*args,limit=1)
        for box,keys in ((args[3],('baseline','core','core_weight')),(args[4],('vol','vol_weight','core','core_weight'))):
            for key in keys:
                box[key]=box[key].astype(object);box[key][1:]=Poison()
        got,sub=audit.independent_features(*args,limit=1)
        other,_=production.build_features(*args,limit=1)
        for key in got:
            np.testing.assert_array_equal(got[key],expected[key]);np.testing.assert_array_equal(got[key],other[key])
        values={key:np.empty((3,5),object) for key in got}
        for key in values:values[key][:]=Poison();values[key][0]=got[key][0]
        actual,receipt=audit.independent_deltas(values,args[3],args[4],limit=1)
        prod,_=production.build_deltas(values,args[3],args[4],limit=1)
        for key in actual:np.testing.assert_array_equal(actual[key],prod[key])
        empty,_=audit.independent_features(*args,limit=0)
        self.assertEqual(empty['unit_delta'].shape,(0,5))

    def test_reject_nonfinite_wrong_shape_or_formula(self):
        for loc,key,value in ((0,'unit_delta',math.nan),(0,'vol_rank',math.inf),(1,'baseline',math.nan),
                              (1,'core',math.inf),(1,'core_weight',0.),(2,'vol',.6)):
            args=list(copy.deepcopy(delta_fixture()));args[loc][key][0,0]=value
            with self.assertRaises(ValueError):audit.independent_deltas(*args)
        args=list(delta_fixture());args[0]['unit_delta'][1,0]+=1e-12
        with self.assertRaisesRegex(ValueError,'formula'):audit.independent_deltas(*args)
        args=list(delta_fixture());args[1]['core_weight']=np.ones((3,2))
        with self.assertRaises(ValueError):audit.independent_deltas(*args)

    def test_reject_changed_fourth_arm_initial_and_audit(self):
        args=delta_fixture();got,record=audit.independent_deltas(*args)
        for arm in audit.LAMBDAS:
            bad=copy.deepcopy(got);bad[arm][1,0]+=1e-10
            with self.assertRaises(ValueError):audit.verify_deltas(*args,bad,record)
        bad=copy.deepcopy(got);bad[audit.ARMS[3]][0,0]=1e-30
        with self.assertRaises(ValueError):audit.verify_deltas(*args,bad,record)
        record['lambdas'][audit.ARMS[1]]=.03
        with self.assertRaises(ValueError):audit.verify_deltas(*args,got,record)


class SourceTests(unittest.TestCase):
    def test_full49_original_source_and_all_cell_audit(self):
        args=fixture(49);before=copy.deepcopy(args);values,record=audit.independent_features(*args)
        receipt=audit.verify_features(*args,values,record)
        self.assertEqual(values['vol_rank'].shape,(3,49));self.assertEqual(receipt['checked_cells'],147)
        self.assertTrue(receipt['original_scalar_vol_statistics_checked']);self.assertEqual(set(values),{'vol_rank','unit_delta'})
        pd.testing.assert_frame_equal(args[0]['adj'],before[0]['adj'])
        for a,b in zip(args[3:],before[3:]):
            for key in a:np.testing.assert_array_equal(a[key],b[key])
        json.dumps(record,allow_nan=False)

    def test_scalar_percentile_ties_missing_and_singleton(self):
        for raw,expected in (([2.,1.,2.,np.nan],[1/3.,-1/6.,1/3.,0.]),([np.nan,np.nan],[0.,0.]),([0.,np.nan],[.5,0.])):
            np.testing.assert_allclose(audit.ranks(raw),expected,atol=1e-16)

    def test_prefixes_future_and_unrequested_symbol_poison(self):
        args=list(fixture());full,rec=audit.independent_features(*args)
        for n in range(4):
            small,sub=audit.independent_features(*args,limit=n)
            for key in small:np.testing.assert_array_equal(small[key],full[key][:n])
            self.assertEqual(sub['event_history'],rec['event_history'][:n])
        class Poison:
            def __float__(self):raise AssertionError('Future or unused value converted')
        frame=args[0]['adj'].astype(object);frame.loc[frame.index>pd.Timestamp(args[1]['dates'][0])]=Poison()
        frame['UNREQUESTED']=Poison();args[0]={'adj':frame}
        small,_=audit.independent_features(*args,limit=1)
        for key in small:np.testing.assert_array_equal(small[key],full[key][:1])

    def test_missing_price_neutral_rank_and_no_core_history_requirement(self):
        args=list(fixture());day=args[1]['dates'][1];t=args[0]['adj'].index.get_loc(day)
        args[0]['adj'].iloc[t-10,0]=np.nan;args[3],args[4]=refresh(args[0],args[1])
        got,record=audit.independent_features(*args)
        self.assertEqual(got['vol_rank'][1,0],0.);self.assertGreater(record['source_vol_missing_neutral_cells'],0)
        self.assertNotEqual(got['unit_delta'][1,0],0.)
        args=list(fixture());t=args[0]['adj'].index.get_loc(args[1]['dates'][0]);args[0]['adj'].iloc[t-252,0]=np.nan
        # No new raw-core support test: stored original arrays remain authoritative.
        self.assertTrue(np.isfinite(audit.independent_features(*args)[0]['unit_delta']).all())

    def test_constant_allties_and_zero_vol_valid(self):
        args=list(fixture());names=args[1]['symbols'][0]
        for name in names:args[0]['adj'][name]=100.
        args[3],args[4]=refresh(args[0],args[1]);got,record=audit.independent_features(*args)
        self.assertEqual(record['source_vol_missing_neutral_cells'],0)
        np.testing.assert_array_equal(got['vol_rank'],np.full((3,5),3./5.-.5))

    def test_missing_official_day_not_compressed_and_precalendar_preserved(self):
        args=list(fixture());full,_=audit.independent_features(*args);args[2]=args[2][100:]
        got,_=audit.independent_features(*args)
        for key in got:np.testing.assert_array_equal(got[key],full[key])
        args=list(fixture());args[0]['adj']=args[0]['adj'].drop(index=args[0]['adj'].index[400])
        with self.assertRaises(ValueError):audit.independent_features(*args)
        args=list(fixture());args[2].pop(400)
        with self.assertRaisesRegex(ValueError,'official calendar'):audit.independent_features(*args)

    def test_bad_prices_source_rank_and_record_tamper(self):
        for bad in (0.,-1.,math.inf,'bad'):
            args=list(fixture());args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].iloc[0,0]=bad
            with self.assertRaises(ValueError):audit.independent_features(*args)
        args=list(fixture());args[4]['vol'][1,0]=np.nextafter(args[4]['vol'][1,0],math.inf)
        with self.assertRaisesRegex(ValueError,'rank parity'):audit.independent_features(*args)
        args=fixture();values,record=audit.independent_features(*args);record['event_history'][1]['unit_delta'][0]+=.01
        with self.assertRaises(ValueError):audit.verify_features(*args,values,record)

    def test_arithmetic_failure_is_not_missing_neutral(self):
        for name in ('S0','SPY'):
            args=list(fixture());frame=args[0]['adj'];frame.loc[frame.index[300],name]=1e200
            with self.assertRaisesRegex(ValueError,'calculation nonfinite'):
                audit.independent_features(*args)

    def test_no_production_import_or_entrypoint(self):
        text=Path(audit.__file__).read_text();tree=ast.parse(text)
        names=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        names += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any('kernel' in name or 'host' in name for name in names));self.assertNotIn('__main__',text)


class ProductionCrossTests(unittest.TestCase):
    def test_both_fixtures_audit_and_all_prefixes(self):
        import monthly_aux_reallocation186_kernel as production
        from test_monthly_aux_reallocation186_kernel import fixture as other_fixture
        for factory in (fixture,other_fixture):
            args=factory();values,record=production.build_features(*args)
            self.assertTrue(audit.verify_features(*args,values,record)['passed'])
            delta,drecord=production.build_deltas(values,args[3],args[4])
            self.assertTrue(audit.verify_deltas(values,args[3],args[4],delta,drecord)['passed'])
            for n in range(len(args[1]['dates'])+1):
                got,sub=production.build_features(*args,limit=n);expected,ind=audit.independent_features(*args,limit=n)
                for key in got:np.testing.assert_array_equal(got[key],expected[key])
                audit.record_check(sub,ind)
                got,sub=production.build_deltas(values,args[3],args[4],limit=n)
                expected,ind=audit.independent_deltas(values,args[3],args[4],limit=n)
                for key in got:np.testing.assert_array_equal(got[key],expected[key])
                audit.record_check(sub,ind)
        for case in ('missing','constant','all_tie','precalendar'):
            args=list(copy.deepcopy(fixture()));frame=args[0]['adj'];names=args[1]['symbols'][0]
            if case=='missing':frame.loc[frame.index[400],names[0]]=np.nan
            elif case=='constant':frame[names[0]]=100.
            elif case=='all_tie':
                for name in names:frame[name]=frame[names[0]]
            else:args[2]=args[2][50:]
            args[3],args[4]=refresh(args[0],args[1]);values,record=production.build_features(*args)
            self.assertTrue(audit.verify_features(*args,values,record)['passed'])


class AccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import test_monthly_upside_retention181_independent as previous
        with patch.object(previous,'audit',audit):cls.args=previous.account_fixture()

    def test_four_arms_twelve_paths_sixty_slices(self):
        receipt=audit.audit_outcomes(*self.args)
        self.assertEqual((receipt['checked_paths'],receipt['checked_slices']),(12,60))
        self.assertEqual(set(receipt['paths']),set(audit.ARMS))
        training=[record['slices'][split] for paths in receipt['paths'].values()
                  for path,record in paths.items() if path in ('same_close_zero','next_open_cost')
                  for split in ('first88','early53','recent35')]
        self.assertEqual(len(training),24)

    def test_fourth_arm_fill_and_slice_tamper_rejected(self):
        args=list(self.args);args[3]=copy.deepcopy(args[3]);args[3][audit.ARMS[3]]['next_open_cost']['TRADES'][0]['adjusted_units']+=1
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args);args[4]=copy.deepcopy(args[4]);args[4][audit.ARMS[3]]['next_open_cost']['slices']['later34']['cagr']+=.1
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args);args[4]=copy.deepcopy(args[4]);args[4][audit.ARMS[3]]['same_close_zero']['slices']['first88']['cagr']+=.1
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        args=list(self.args);args[2]=copy.deepcopy(args[2]);del args[2][audit.ARMS[3]]
        with self.assertRaises(ValueError):audit.audit_outcomes(*args)
        with patch.object(audit,'OUTCOME_SHA','0'*64),self.assertRaises(ValueError):audit.audit_outcomes(*self.args)


if __name__=='__main__':unittest.main()
