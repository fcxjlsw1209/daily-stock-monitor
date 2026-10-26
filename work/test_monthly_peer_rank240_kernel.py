"""Synthetic-only mid-CDF tests; source fixtures come from frozen peer239."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import monthly_peer_rank240_kernel as kernel

FIXTURE_PATH=Path(__file__).with_name('test_monthly_peer239_kernel.py')
FIXTURE_SHA256='e01dd392478a231e13f80c9005e4f3cf9877f79cfd3fc2ca3c7af5484afa20af'
if hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest()!=FIXTURE_SHA256:
    raise ValueError('Frozen synthetic source fixture changed')
_spec=importlib.util.spec_from_file_location('peer_rank240_fixtures239',FIXTURE_PATH)
_fixtures=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_fixtures)
make_report=_fixtures.make_report
physical=_fixtures.physical
warm_fixture=_fixtures.warm_fixture


def fixture():
    panel,meta,calendar,arrays,_=_fixtures.fixture()
    names=['AAA','BBB','CCC','DDD','EEE','FFF','GGG','QQQ']
    frame=pd.DataFrame(128.,index=panel['adj'].index,columns=names)
    meta['symbols']=[names.copy() for _ in meta['dates']]
    for day in meta['dates'][1:]:
        end=pd.Timestamp(calendar[calendar.index(day)-42])
        frame.loc[end,names[:7]]=[112.,128.,160.,64.,144.,192.,256.]
    sectors=pd.DataFrame('Technology',index=frame.index,columns=names)
    sectors.loc[:,names[3:7]]='Healthcare'
    panel={'adj':frame};arrays={'eligible':np.ones((len(meta['dates']),len(names)),bool)}
    arrays['eligible'][:,3]=False
    return panel,meta,calendar,arrays,make_report(panel,meta,calendar,arrays,sectors=sectors)


def run(args,limit=None):return kernel.build_features(*args,limit=limit)


def equal_arrays(case,left,right):
    case.assertEqual(tuple(left),tuple(right))
    for key in left:
        case.assertEqual(left[key].dtype,right[key].dtype)
        case.assertEqual(left[key].shape,right[key].shape)
        case.assertEqual(left[key].tobytes(),right[key].tobytes(),key)


def change_momenta(args,transforms):
    panel,meta,calendar,_,_=args
    for day in meta['dates'][1:]:
        t=calendar.index(day);start,end=pd.Timestamp(calendar[t-252]),pd.Timestamp(calendar[t-42])
        for symbol,transform in transforms.items():
            original=float(panel['adj'].at[end,symbol])/float(panel['adj'].at[start,symbol])-1.
            panel['adj'].at[end,symbol]=float(panel['adj'].at[start,symbol])*(1.+transform(original))


class KernelTests(unittest.TestCase):
    def test_exact_schema_six_counts_and_integer_audit(self):
        values,audit=run(fixture())
        self.assertEqual(tuple(values),kernel.VALUE_KEYS)
        self.assertEqual(audit['schema'],'monthly_peer_rank240_features_v1')
        self.assertEqual(audit['counts'],dict(events=4,stock_cells=28,mature_cells=21,supported_cells=21,endpoint_observations=42,new_model_fits=0))
        for event in audit['events']:
            self.assertEqual(tuple(event),kernel.EVENT_KEYS)
            for cell in event['cells']:
                self.assertEqual(tuple(cell),kernel.CELL_KEYS)
                for field in kernel.INTEGER_FIELDS:
                    self.assertIs(type(cell[field]),int if cell['column_index'] in event['supported_indices'] else type(None))
        json.dumps(audit,allow_nan=False)

    def test_three_and_four_member_groups_and_global_seven(self):
        v,a=run(fixture())
        np.testing.assert_array_equal(v['peer_target'][1,:7],[-1/3,0.,1/3,-3/8,-1/8,1/8,3/8])
        np.testing.assert_array_equal(v['absolute_target'][1,:7],[-2/7,-1/7,1/7,-3/7,0.,2/7,3/7])
        self.assertEqual([c['peer_numerator'] for c in a['events'][1]['cells']],[-2,0,2,-3,-1,1,3])
        self.assertEqual([c['peer_denominator'] for c in a['events'][1]['cells']],[6,6,6,8,8,8,8])
        self.assertEqual([c['absolute_denominator'] for c in a['events'][1]['cells']],[14]*7)

    def test_source_momentum_median_and_old_audit_preserved(self):
        args=fixture();old,oa=kernel.frozen_helper().build_features(*args);v,a=run(args)
        for key in kernel.VALUE_KEYS[:4]:self.assertEqual(v[key].tobytes(),old[key].tobytes())
        changed=deepcopy(a);changed['schema']=oa['schema']
        for ev,oev in zip(changed['events'],oa['events']):
            for cell,ocell in zip(ev['cells'],oev['cells']):
                for field in kernel.INTEGER_FIELDS:cell.pop(field)
                for key in ('peer_target','absolute_target'):cell[key]=ocell[key]
        self.assertEqual(changed,oa)

    def test_all_ties_are_exact_zero_valid_not_unavailable(self):
        args=fixture();args[0]['adj'].loc[:,:]=128.;v,a=run(args)
        self.assertTrue(v['supported'][1:,:7].all())
        self.assertTrue((v['peer_target'][1:,:7]==0.).all())
        self.assertTrue((v['absolute_target'][1:,:7]==0.).all())
        self.assertTrue(all(c['peer_numerator']==c['absolute_numerator']==0 for c in a['events'][1]['cells']))

    def test_one_constant_sector_does_not_flatten_other_sector(self):
        args=fixture();change_momenta(args,{s:lambda m:0. for s in ('AAA','BBB','CCC')});v,a=run(args)
        np.testing.assert_array_equal(v['peer_target'][1,:3],[0.,0.,0.])
        np.testing.assert_array_equal(v['peer_target'][1,3:7],[-3/8,-1/8,1/8,3/8])
        self.assertTrue(v['supported'][1,:7].all())
        self.assertTrue(all(c['absolute_numerator']==-2 for c in a['events'][1]['cells'][:3]))

    def test_exact_ties_receive_group_midrank(self):
        args=fixture();change_momenta(args,{'AAA':lambda m:0.});v,a=run(args)
        np.testing.assert_array_equal(v['peer_target'][1,:3],[-1/6,-1/6,1/3])
        self.assertEqual([c['peer_numerator'] for c in a['events'][1]['cells'][:3]],[-1,-1,2])

    def test_near_ties_are_not_snapped(self):
        args=fixture()
        for day in args[1]['dates'][1:]:
            end=pd.Timestamp(args[2][args[2].index(day)-42]);args[0]['adj'].at[end,'AAA']=128.;args[0]['adj'].at[end,'BBB']=np.nextafter(128.,np.inf)
        v,_=run(args);self.assertGreater(v['momentum'][1,1],v['momentum'][1,0])
        np.testing.assert_array_equal(v['peer_target'][1,:3],[-1/3,0.,1/3])

    def test_group_size_endpoint_effect_is_explicit(self):
        v,_=run(fixture());self.assertEqual(v['peer_target'][1,2],1/3)
        self.assertEqual(v['peer_target'][1,6],3/8);self.assertNotEqual(v['peer_target'][1,2],v['peer_target'][1,6])

    def test_sector_affine_invariance_changes_global_control(self):
        args=fixture();before,_=run(args)
        change_momenta(args,{s:lambda m:2.*m+2. for s in ('AAA','BBB','CCC')})
        after,_=run(args);np.testing.assert_array_equal(after['peer_target'],before['peer_target'])
        self.assertFalse(np.array_equal(after['absolute_target'][1,:7],before['absolute_target'][1,:7]))

    def test_sector_strict_nonlinear_monotone_invariance(self):
        args=fixture();before,_=run(args)
        change_momenta(args,{s:lambda m:(m+1.)**2-1. for s in ('DDD','EEE','FFF','GGG')})
        after,_=run(args);np.testing.assert_array_equal(after['peer_target'],before['peer_target'])

    def test_price_rescaling_by_security_invariance(self):
        args=fixture();before,ba=run(args)
        for j,s in enumerate(args[0]['adj'].columns):args[0]['adj'][s]*=2.**(j-3)
        after,aa=run(args);equal_arrays(self,after,before);self.assertEqual(aa,ba)

    def test_negative_ineligible_peers_and_E_toggle_do_not_change_features(self):
        args=fixture();before,ba=run(args)
        self.assertFalse(args[3]['eligible'][1,3]);self.assertEqual(before['momentum'][1,3],-.5)
        self.assertIn(3,ba['events'][1]['cells'][4]['peer_indices'])
        args[3]['eligible']=~args[3]['eligible'];after,aa=run(args)
        equal_arrays(self,after,before);self.assertEqual(aa,ba)

    def test_partial_and_ETF_do_not_read_numeric_sources(self):
        args=fixture();args[0]['adj']['QQQ']='poison';v,a=run(args)
        self.assertFalse(v['mature'][0].any());self.assertFalse(v['supported'][:,-1].any())
        for key in kernel.VALUE_KEYS[2:]:self.assertTrue(np.isnan(v[key][0]).all());self.assertTrue(np.isnan(v[key][:,-1]).all())
        partial=physical(args,1);partial[0]['adj']=partial[0]['adj'].iloc[:0]
        pv,pa=run(partial);equal_arrays(self,pv,{k:x[:1] for k,x in v.items()})
        self.assertEqual(pa['events'],a['events'][:1])

    def test_warmup_preserved_and_integer_audit_unavailable(self):
        args=warm_fixture();v,a=run(args);self.assertFalse(v['mature'][0,3]);self.assertFalse(v['supported'][0,3])
        cell=a['events'][0]['cells'][3];self.assertEqual(cell['history_status'],'official_pre_inception')
        self.assertTrue(all(cell[k] is None for k in kernel.INTEGER_FIELDS))
        args[0]['adj']['SPCX']='poison';vv,aa=run(args);equal_arrays(self,vv,v);self.assertEqual(aa,a)

    def test_insufficient_peer_domain_stays_unavailable(self):
        args=list(_fixtures.fixture());sectors=pd.DataFrame('Technology',index=args[0]['adj'].index,columns=args[0]['adj'].columns)
        sectors['CCC']='Healthcare';sectors['DDD']='Energy';args[4]=make_report(*args[:4],sectors=sectors)
        v,a=run(args);self.assertFalse(v['supported'].any());self.assertTrue(v['mature'][1:,:4].all())
        self.assertTrue(np.isnan(v['peer_target']).all());self.assertTrue(np.isnan(v['absolute_target']).all())
        self.assertTrue(all(c[k] is None for e in a['events'] for c in e['cells'] for k in kernel.INTEGER_FIELDS))

    def test_single_supported_sector_candidate_equals_control(self):
        args=list(_fixtures.fixture());sectors=pd.DataFrame('Technology',index=args[0]['adj'].index,columns=args[0]['adj'].columns)
        sectors['DDD']='Healthcare';args[4]=make_report(*args[:4],sectors=sectors)
        v,a=run(args);self.assertEqual(v['peer_target'].tobytes(),v['absolute_target'].tobytes())
        self.assertTrue(np.isnan(v['peer_target'][1,3]))

    def test_all_logical_physical_prefixes_and_input_immutability(self):
        args=fixture();snapshot=deepcopy(args);v,a=run(args)
        for n in range(5):
            vv,aa=run(args,n);pv,pa=run(physical(args,n));equal_arrays(self,vv,pv);self.assertEqual(aa,pa)
            equal_arrays(self,vv,{k:x[:n] for k,x in v.items()});self.assertEqual(aa['events'],a['events'][:n])
        pd.testing.assert_frame_equal(args[0]['adj'],snapshot[0]['adj']);self.assertEqual(args[1:3],snapshot[1:3])
        np.testing.assert_array_equal(args[3]['eligible'],snapshot[3]['eligible']);self.assertEqual(args[4],snapshot[4])

    def test_empty_physical_zero_width(self):
        args=list(physical(fixture(),0));args[3]['eligible']=np.empty((0,0),bool)
        v,a=run(args);self.assertTrue(all(x.shape==(0,0) for x in v.values()));self.assertEqual(a['events'],[])

    def test_future_price_metadata_report_and_E_poison_ignored(self):
        args=fixture();v,a=run(args,2);cut=pd.Timestamp(args[1]['dates'][1]);args[0]['adj']=args[0]['adj'].astype(object)
        args[0]['adj'].loc[args[0]['adj'].index>cut,:]='poison';args[1]['dates'][2:] = ['bad','bad'];args[1]['symbols'][2:] = [['bad'],['bad']]
        args[3]['eligible']=args[3]['eligible'].astype(object);args[3]['eligible'][2:,:]='bad'
        args[4]['events'][2:] = ['bad','bad']
        args[4]['records']=[r if r['query_index']<2 else {'query_index':r['query_index'],'poison':True} for r in args[4]['records']]
        vv,aa=run(args,2);equal_arrays(self,vv,v);self.assertEqual(aa,a)

    def test_latest42_prices_do_not_affect_consumed_feature(self):
        args=physical(fixture(),2);v,a=run(args);day=args[1]['dates'][-1];end=pd.Timestamp(args[2][args[2].index(day)-42])
        args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].loc[args[0]['adj'].index>end,:]='poison'
        vv,aa=run(args);equal_arrays(self,vv,v);self.assertEqual(aa,a)

    def test_missing_peer_or_bad_source_certificate_still_fails(self):
        for mutation in ('peer','admitted','duplicate','clock'):
            args=fixture();r=args[4]['records'][7]
            if mutation=='peer':r['peer_indices'].pop()
            elif mutation=='admitted':args[4]['admitted']=False
            elif mutation=='duplicate':args[4]['records'].append(deepcopy(r))
            else:r['source_dates'][0]='2000-01-01'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):run(args)

    def test_required_price_bool_string_nonpositive_or_nan_fails(self):
        for raw in (True,'128',0.,-1.,np.nan,np.inf):
            args=fixture();end=pd.Timestamp(args[2][args[2].index(args[1]['dates'][1])-42]);args[0]['adj']=args[0]['adj'].astype(object);args[0]['adj'].at[end,'AAA']=raw
            with self.subTest(raw=raw),self.assertRaises(ValueError):run(args)

    def test_frozen_dependency_pin_and_no_new_fit(self):
        self.assertEqual(set(kernel.dependency_files()),{kernel.FROZEN,kernel.AXES})
        with patch.object(kernel,'FROZEN_SHA256','bad'),self.assertRaises(ValueError):run(fixture())
        with patch.object(kernel,'AXES_SHA256','bad'),self.assertRaises(ValueError):run(fixture())
        self.assertEqual(run(fixture())[1]['counts']['new_model_fits'],0)


if __name__=='__main__':unittest.main(verbosity=2)
