"""Invented-data validation of228's detached risk-history source adapter."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import monthly_expanded_error228_adapter as model
import monthly_joint_error226_model as risk
import monthly_joint_error226_host as oldhost
import test_monthly_joint_error226_model as base


def fixture():
    train,tm,qm,old,use,pre,eligible=base.fixture()
    total=len(train['dates']);width=use.shape[1]
    train['train_eligible']=np.ones(total,bool)
    train['vol63']=np.ones(total)
    new={k:old[k].copy() for k in model.RAW_KEYS+('available','local_available')}
    new['expansion_allowed']=np.ones(len(old['available']),bool)
    labels={k:train[k].copy() for k in ('dates','ends','offsets','common_good','vol63')}
    labels.update({'target_'+k:train['target_'+k].copy() for k in model.TARGETS})
    added=np.zeros(total,bool);added[np.arange(18)*width+3]=True
    labels['added']=added.copy()
    train['common_good'][added]=False;train['train_eligible'][added]=False
    for k in model.TARGETS:train['target_'+k][added]=np.nan
    old['available'][added]=False;old['local_available'][added]=False
    for k in model.RAW_KEYS+('raw_candidate','raw_control'):old[k][added]=np.nan
    use.reshape(-1)[added]=False
    args=(train,tm,qm,old,new,labels,use)
    return args,dict(monthly=copy.deepcopy(qm),training=copy.deepcopy(tm))


def call(args,metadata):
    return model.adapt(*args,source_meta227=metadata)


def snapshot(value):
    if isinstance(value,np.ndarray):return ('array',value.dtype.str,value.shape,value.tobytes())
    if isinstance(value,dict):return {k:snapshot(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [snapshot(v) for v in value]
    return value


class AdapterTests(unittest.TestCase):
    def test_expanded_history_counts_and_only_four_labels_replaced(self):
        args,meta=fixture();out=call(args,meta);c=out['audit']['counts']
        self.assertEqual(set(out),{'train','source','historical_use','audit'})
        self.assertEqual(set(c),set(model.COUNT_FIELDS))
        self.assertEqual(c,dict(training_rows=120,query_rows=120,original_forecast_rows=102,
            expanded_forecast_rows=120,added_forecast_rows=18,original_label_rows=102,
            added_label_rows=18,extended_label_rows=120,original_history_rows=102,expanded_history_rows=120))
        for key in args[0]:
            expected=args[5][key] if key=='common_good' or key.startswith('target_') else args[0][key]
            self.assertEqual(snapshot(out['train'][key]),snapshot(expected))
        self.assertEqual(set(out['source']),{'raw_local_linear','available'})
        self.assertTrue(out['historical_use'].all())

    def test_inputs_unchanged_and_all_outputs_detached(self):
        args,meta=fixture();before=snapshot((args,meta));out=call(args,meta)
        self.assertEqual(before,snapshot((args,meta)))
        arrays=[v for d in (args[0],args[3],args[4],args[5]) for v in d.values() if isinstance(v,np.ndarray)]+[args[6]]
        for v in list(out['train'].values())+list(out['source'].values())+[out['historical_use']]:
            self.assertTrue(all(not np.shares_memory(v,x) for x in arrays))
        out['train']['train_plain'][:]=12;out['source']['raw_local_linear'][:]=13;out['historical_use'][:]=False
        self.assertEqual(before,snapshot((args,meta)))

    def test_false_original_U_not_promoted_if_old_forecast_existed(self):
        args,meta=fixture();args[6][0,0]=False;out=call(args,meta)
        self.assertFalse(out['historical_use'][0,0]);self.assertTrue(out['historical_use'][0,3])
        self.assertEqual(out['audit']['counts']['expanded_history_rows'],119)

    def test_exact_old_forecasts_reject_single_ulp(self):
        for key in model.RAW_KEYS:
            args,meta=fixture();args[4][key][0]=np.nextafter(args[4][key][0],np.inf)
            with self.assertRaises(ValueError):call(args,meta)

    def test_exact_old_forecast_signed_zero_preserved(self):
        args,meta=fixture();args[3]['raw_local_linear'][0]=0.;args[4]['raw_local_linear'][0]=-0.
        with self.assertRaises(ValueError):call(args,meta)

    def test_nonadded_nan_payload_bytes_preserved(self):
        args,meta=fixture();i=5
        for src in (args[3],args[4]):
            src['available'][i]=False
            for key in model.RAW_KEYS:src[key][i]=np.nan
        args[6].reshape(-1)[i]=False
        args[4]['raw_local_linear'].view(np.uint64)[i]=np.uint64(0x7ff8000000000001)
        with self.assertRaises(ValueError):call(args,meta)

    def test_old_availability_and_local_flags_not_removed(self):
        args,meta=fixture();args[4]['available'][0]=False
        with self.assertRaises(ValueError):call(args,meta)
        args,meta=fixture();args[4]['local_available'][0]=True
        with self.assertRaises(ValueError):call(args,meta)

    def test_new_forecasts_require_expansion_permission(self):
        args,meta=fixture();args[4]['expansion_allowed'][3]=False
        with self.assertRaises(ValueError):call(args,meta)

    def test_exact_metadata_identity_rejects_permuted_ID(self):
        args,meta=fixture();meta['monthly']['symbols'][0][0:2]=['B','A']
        with self.assertRaises(ValueError):call(args,meta)
        args,meta=fixture();meta['training']['ends'][0]='2017-08-31'
        with self.assertRaises(ValueError):call(args,meta)

    def test_date_and_offset_bytes_and_dimensions(self):
        for key in ('dates','ends'):
            args,meta=fixture();args[5][key][0]='2000-01-31'
            with self.assertRaises(ValueError):call(args,meta)
        args,meta=fixture();args[5]['offsets']=args[5]['offsets'].astype(float)
        with self.assertRaises(ValueError):call(args,meta)

    def test_added_labels_only_ineligible_and_oldgood_retained(self):
        args,meta=fixture();args[0]['train_eligible'][3]=True
        with self.assertRaises(ValueError):call(args,meta)
        args,meta=fixture();args[5]['common_good'][0]=False
        with self.assertRaises(ValueError):call(args,meta)

    def test_all_nonadded_labels_exact(self):
        for key in model.TARGETS:
            args,meta=fixture();args[5]['target_'+key][0]=np.nextafter(0.,1.)
            with self.assertRaises(ValueError):call(args,meta)

    def test_new_label_common_support_and_features_finite(self):
        args,meta=fixture();args[5]['target_log_excess'][3]=np.nan
        with self.assertRaises(ValueError):call(args,meta)
        args,meta=fixture();args[0]['train_plain'][3,0]=np.nan
        with self.assertRaises(ValueError):call(args,meta)

    def test_source_schema_boolean_axes_and_finite_support(self):
        args,meta=fixture();args[4]['extra']=np.zeros(120)
        with self.assertRaises(ValueError):call(args,meta)
        args,meta=fixture();args[4]['available']=args[4]['available'].astype(int)
        with self.assertRaises(ValueError):call(args,meta)
        args,meta=fixture();args[4]['raw_source_log'][3]=np.inf
        with self.assertRaises(ValueError):call(args,meta)

    def test_original_U_must_be_boolean_and_within_old_availability(self):
        args,meta=fixture();wrong=list(args);wrong[6]=wrong[6].astype(int)
        with self.assertRaises(ValueError):call(wrong,meta)
        args,meta=fixture();args[6][0,3]=True
        with self.assertRaises(ValueError):call(args,meta)

    def test_expanded_vol63_is_not_injected_into_risk_train(self):
        args,meta=fixture();args[5]['vol63'][:]=np.nan;out=call(args,meta)
        self.assertEqual(snapshot(out['train']['vol63']),snapshot(args[0]['vol63']))
        args,meta=fixture();args[5]['vol63']=np.ones((120,1))
        with self.assertRaises(ValueError):call(args,meta)

    def test_adapter_never_calls_risk_build_or_fits(self):
        args,meta=fixture()
        with patch.object(risk,'build',side_effect=AssertionError('must not estimate risk')):
            out=call(args,meta)
        for key in ('source_forecasts_refitted','labels_recomputed','numerical_risk_estimates_computed',
                    'features_recomputed','portfolio_returns_computed'):self.assertIs(out['audit'][key],False)
        self.assertEqual(out['audit']['train_output_role'],'risk_history_only_never_Ridge_training')

    def test_frozen_numerics_reused_and_min18_not_relaxed(self):
        args,meta=fixture();out=call(args,meta);pre=base.fixture()[5];eligible=base.fixture()[6]
        old=risk.build(args[0],args[1],args[2],args[3],args[6],pre,eligible)
        new=risk.build(out['train'],args[1],args[2],out['source'],out['historical_use'],pre,eligible)
        self.assertFalse(old['arrays']['ready'][-1]);self.assertTrue(new['arrays']['ready'][-1])
        self.assertEqual(len(new['audit']['event_history'][-1]['H']),18)
        self.assertEqual(new['audit']['min_months'],18)
        expected=risk.build(*base.fixture())
        for key in new['arrays']:np.testing.assert_array_equal(new['arrays'][key],expected['arrays'][key])
        self.assertEqual(new['audit'],expected['audit'])

    def test_every_physical_frozen_host_prefix_matches_full(self):
        args,meta=fixture();out=call(args,meta);pre=base.fixture()[5];eligible=base.fixture()[6]
        full=risk.build(out['train'],args[1],args[2],out['source'],out['historical_use'],pre,eligible)
        for n in range(len(pre)+1):
            small=oldhost.prefix_inputs(out['train'],args[1],args[2],out['source'],out['historical_use'],n)
            got=risk.build(*small,pre[:n].copy(),eligible[:n].copy())
            self.assertEqual(got['audit']['event_history'],full['audit']['event_history'][:n])
            for key in got['arrays']:np.testing.assert_array_equal(got['arrays'][key],full['arrays'][key][:n])
            if not n:self.assertEqual(len(small[0]['dates']),0)
            else:self.assertTrue(all(e<=str(int(small[2]['dates'][-1][:4])-1)+'-12-31' for e in small[1]['ends']))

    def test_no_expansion_reproduces_frozen_risk_exact(self):
        args,meta=fixture();args=list(args)
        new={k:args[3][k].copy() for k in model.RAW_KEYS+('available','local_available')}
        new['expansion_allowed']=np.zeros(120,bool);args[4]=new
        labels={k:args[0][k].copy() for k in model.LABEL_KEYS if k!='added'};labels['added']=np.zeros(120,bool);args[5]=labels
        out=call(args,meta);pre=base.fixture()[5];eligible=base.fixture()[6]
        expected=risk.build(args[0],args[1],args[2],args[3],args[6],pre,eligible)
        got=risk.build(out['train'],args[1],args[2],out['source'],out['historical_use'],pre,eligible)
        self.assertEqual(got['audit'],expected['audit'])
        for key in got['arrays']:np.testing.assert_array_equal(got['arrays'][key],expected['arrays'][key])

    def test_zero_query_adapter_prefix(self):
        args,meta=fixture();args=list(args)
        args[2]={k:[] for k in ('dates','ends','symbols')}
        args[3]={k:v[:0].copy() for k,v in args[3].items()};args[4]={k:v[:0].copy() for k,v in args[4].items()}
        args[6]=np.zeros((0,6),bool);meta['monthly']=copy.deepcopy(args[2]);out=call(args,meta)
        self.assertEqual(out['historical_use'].shape,(0,6));self.assertEqual(out['audit']['counts']['query_rows'],0)


if __name__=='__main__':unittest.main()
