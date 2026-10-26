"""Synthetic pinned-reference fixtures only; never loads saved numerical data."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import monthly_monotone_tree266_reference as r
import monthly_monotone_tree266_kernel as k
import monthly_monotone_tree266_independent as ind
import test_monthly_monotone_tree266_kernel as f

class Poison:
    def __float__(self):raise AssertionError('Future float read')
    def __iter__(self):raise AssertionError('Future body iterated')
    def __deepcopy__(self,memo):raise AssertionError('Future body copied')
    def __len__(self):raise AssertionError('Future body inspected')

def reference(args):
    b=k.legacy.fit_models(args[1],args[2],args[3]);orders,scores,pres,_=k.legacy.decisions(args[0],args[3],args[2]['monthly'],b)
    return dict(forecasts={role+'_'+key:b[role][key].copy() for role in ('control','candidate') for key in ('raw','available','rank')},
      model_audit=dict(model=b['audit'],labels=b['label_preservation'],control=b['control']['audit'],candidate=b['candidate']['audit']),
      labels=dict(training_row_ids=b['training_row_ids'].copy(),label_values=b['label_values'].copy()),scores=scores,pre_scores=pres,orders=orders,meta=deepcopy(args[2]['monthly']))

def stages(args,n=None):
    out={};n=len(args[2]['monthly']['dates']) if n is None else n
    def cal(v):
        receipt=ind.verify_calibration_models(args[1],args[2],args[3],v,n)
        strategy=k.calibration_decisions(args[0],args[3],args[2]['monthly'],v,n)
        out['calibration']=(deepcopy(v),receipt,strategy);return receipt
    def control(v):
        receipt=ind.verify_control_models(args[1],args[2],args[3],v,n)
        strategy=k.control_decisions(args[0],args[3],args[2]['monthly'],v,n)
        out['control']=(deepcopy(v),receipt,strategy);return receipt
    k.fit_models(args[1],args[2],args[3],n,calibration_validator=cal,control_validator=control)
    return out

def layout(args,b):
    return dict(n=b['n'],width=b['width'],monthly=deepcopy(args[2]['monthly']),
      original_u=(args[3]['eligible']&args[3]['forecast_available']).copy(),
      annual_cohorts=[{key:deepcopy(row[key]) for key in r.COHORT_KEYS} for row in b['calibration_ridge']['audit']['annual']])

def invoke(ref,current,lay,receipt,strategy,stage,binding=None):
    if binding is None:binding=r.bind_verified_inputs(current,lay,receipt,stage=stage)
    fn=r.compare_calibration_reference if stage=='calibration' else r.compare_control_reference
    return fn(ref,current,lay,independent_receipt=receipt,verification_binding=binding,orders=strategy[0],scores=strategy[1],pre_scores=strategy[2])

class ReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args=f.fixture();cls.ref=reference(cls.args);cls.stages=stages(cls.args)
    def case(self,stage='control'):
        b,receipt,strategy=deepcopy(self.stages[stage]);return deepcopy(self.ref),b,layout(self.args,b),receipt,strategy
    def test_full_exact_stages_and_honest_counts_no_refits(self):
        for stage in ('calibration','control'):
            ref,b,lay,receipt,strategy=self.case(stage)
            with patch.object(k,'_fit_tree',side_effect=AssertionError('No refit')),patch.object(k,'fit_ridge',side_effect=AssertionError('No solve')):out=invoke(ref,b,lay,receipt,strategy,stage)
            self.assertTrue(out['passed']);self.assertEqual(tuple(out['counts']),r.COUNT_FIELDS);self.assertEqual(out['counts']['model_fits'],0);self.assertEqual(out['counts']['returns_computed'],0)
            self.assertEqual(out['counts']['forecast_numeric_cells_compared'],48);self.assertEqual(out['counts']['availability_cells_compared'],24)
            self.assertEqual(out['counts']['ordered_events_compared'],4);self.assertEqual(out['counts']['ridge_models_compared'],2 if stage=='calibration' else 0);self.assertEqual(out['counts']['tree_models_compared'],2 if stage=='control' else 0)
            self.assertFalse(out['receipt_authentication_performed'])
    def test_all_prefixes_match_without_loading_or_refitting_helper(self):
        for n in range(5):
            ss=stages(self.args,n)
            for stage,(b,rec,strategy) in ss.items():self.assertTrue(invoke(self.ref,b,layout(self.args,b),rec,strategy,stage)['passed'])
    def test_n0_partial_no_annual_or_label_numerics(self):
        for n in (0,1):
            ss=stages(self.args,n);ref=deepcopy(self.ref)
            for role in ('control','candidate'):ref['model_audit'][role]=Poison()
            ref['labels']['training_row_ids']=Poison();ref['labels']['label_values']=Poison()
            for stage,(b,rec,strategy) in ss.items():
                lay=layout(self.args,b);lay['annual_cohorts']=Poison()
                self.assertTrue(invoke(ref,b,lay,rec,strategy,stage)['passed'])
    def test_future_forecast_labels_metadata_and_year_bodies_poison(self):
        ss=stages(self.args,2);ref=deepcopy(self.ref)
        for key,a in ref['forecasts'].items():
            if key.endswith('available'):continue
            a=a.astype(object);a[2 if key.endswith('rank') else 12:]=Poison();ref['forecasts'][key]=a
        count=len(ss['calibration'][0]['label_values']);a=ref['labels']['label_values'].astype(object);a[count:]=Poison();ref['labels']['label_values']=a
        for field in ('scores','pre_scores'):
            for arm,a in ref[field].items():a=a.astype(object);a[2:]=Poison();ref[field][arm]=a
        for arm in r.ARMS:ref['orders'][arm][2:]=[Poison(),Poison()]
        for key in ('dates','ends','symbols'):ref['meta'][key][2:]=[Poison(),Poison()]
        for role in ('control','candidate'):
            row=ref['model_audit'][role]['annual'][1]
            for key in row:
                if key!='year':row[key]=Poison()
        for stage,(b,rec,strategy) in ss.items():self.assertTrue(invoke(ref,b,layout(self.args,b),rec,strategy,stage)['passed'])
    def test_same_year_future_predictions_sliced_before_numeric_access(self):
        ss=stages(self.args,3);ref=deepcopy(self.ref)
        for role in ('control','candidate'):
            row=ref['model_audit'][role]['annual'][1]
            for key in ('raw_predictions','manual_predictions'):
                if key in row:row[key][6:]=[Poison() for _ in row[key][6:]]
        for stage,(b,rec,strategy) in ss.items():self.assertTrue(invoke(ref,b,layout(self.args,b),rec,strategy,stage)['passed'])
    def test_prefix_no_query_does_not_borrow_later_full_fit(self):
        args=f.clone(self.args);args[3]['eligible'][2]=False;f.refresh(args);ref=reference(args);ss=stages(args,3)
        for role in ('control','candidate'):
            row=ref['model_audit'][role]['annual'][1];self.assertEqual(row['status'],'fit')
            for key in ('weights','mean','scale','beta','intercept','gram','rhs','tree','raw_predictions','manual_predictions'):
                if key in row:row[key]=Poison()
        for stage,(b,rec,strategy) in ss.items():
            self.assertEqual(b['calibration_ridge']['audit']['annual'][1]['status'],'no_prediction_rows')
            out=invoke(ref,b,layout(args,b),rec,strategy,stage);self.assertEqual(out['counts']['ridge_models_compared']+out['counts']['tree_models_compared'],1)
    def test_mutation_after_independent_binding_rejected(self):
        for target in ('forecast','layout','receipt'):
            ref,b,lay,rec,strategy=self.case();binding=r.bind_verified_inputs(b,lay,rec,stage='control')
            if target=='forecast':b['unconstrained_control']['raw'][6]+=.01
            elif target=='layout':lay['original_u'][1,0]=False
            else:rec['counts']['training_rows']+=1
            with self.assertRaisesRegex(ValueError,'changed after|counts'):invoke(ref,b,lay,rec,strategy,'control',binding)
    def test_copied_or_wrong_object_binding_rejected(self):
        ref,b,lay,rec,strategy=self.case();binding=r.bind_verified_inputs(b,lay,rec,stage='control')
        for wrong in (object(),None):
            with self.assertRaisesRegex(ValueError,'issued'):r.compare_control_reference(ref,b,lay,independent_receipt=rec,verification_binding=wrong,orders=strategy[0],scores=strategy[1],pre_scores=strategy[2])
        with self.assertRaisesRegex(ValueError,'Exact verified objects'):invoke(ref,deepcopy(b),lay,rec,strategy,'control',binding)
        with self.assertRaises(AttributeError):binding.digest='changed'
    def test_generic_synthetic_or_wrong_stage_receipt_rejected(self):
        for change in ('empty','synthetic','scope','role','flag','counts'):
            ref,b,lay,rec,strategy=self.case()
            if change=='empty':rec={'passed':True}
            elif change=='synthetic':rec['synthetic_no_callback']=True
            elif change=='scope':rec['checked_events']=3
            elif change=='role':rec['full_candidate_supplied']=True
            elif change=='flag':rec['same_library_unconstrained_refit_checked']=False
            else:rec['counts']['actual_model_fits']+=1
            with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
    def test_current_status_cannot_override_verified_source_fit_state(self):
        ref,b,lay,rec,strategy=self.case();b['unconstrained_control']['audit']['annual'][0]['status']='no_prediction_rows'
        with self.assertRaisesRegex(ValueError,'prefix status'):invoke(ref,b,lay,rec,strategy,'control')
    def test_role_mismatch_rejected_before_tree_projection(self):
        ref,b,lay,rec,strategy=self.case();b['unconstrained_control']['audit']['annual'][0]['tree']['role']='monotone_candidate'
        with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
    def test_forecast_raw_rank_availability_and_dtype_tampering(self):
        for key in ('candidate_raw','candidate_rank','candidate_available'):
            ref,b,lay,rec,strategy=self.case();a=ref['forecasts'][key]
            if key.endswith('available'):a[6]=not a[6]
            elif key.endswith('rank'):a[1,0]=np.nextafter(a[1,0],np.inf)
            else:a[6]=np.nextafter(a[6],np.inf)
            with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
        ref,b,lay,rec,strategy=self.case();ref['forecasts']['candidate_raw']=ref['forecasts']['candidate_raw'].astype(np.float32)
        with self.assertRaisesRegex(ValueError,'binary64'):invoke(ref,b,lay,rec,strategy,'control')
    def test_signed_zero_in_array_is_not_equal(self):
        ref,b,lay,rec,strategy=self.case();b['unconstrained_control']['raw'][0]=-0.;ref['forecasts']['candidate_raw'][0]=0.
        with self.assertRaisesRegex(ValueError,'exact bytes'):invoke(ref,b,lay,rec,strategy,'control')
    def test_signed_zero_and_native_type_in_json_are_not_equal(self):
        for old,new in ((0.,-0.),(1,True)):
            with self.assertRaises(ValueError):r._equal(old,new,'Typed evidence')
    def test_calibration_coefficients_weights_gram_and_ids_exact(self):
        for key in ('weights','mean','scale','beta','intercept','gram','rhs','training_row_ids','query_row_ids'):
            ref,b,lay,rec,strategy=self.case('calibration');row=ref['model_audit']['control']['annual'][0]
            if key=='intercept':row[key]=np.nextafter(row[key],np.inf).item()
            elif key=='gram':row[key][0][0]=np.nextafter(row[key][0][0],np.inf).item()
            elif key.endswith('ids'):row[key][0]+=1
            else:row[key][0]=np.nextafter(row[key][0],np.inf).item()
            with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'calibration')
    def test_frozen_tree_all_fourteen_fields_are_compared(self):
        ref,b,lay,rec,strategy=self.case();old=ref['model_audit']['candidate']['annual'][0]['tree'];old['parameters']['random_state']+=1
        with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
    def test_raw_and_manual_prediction_order_tampering(self):
        for key in ('raw_predictions','manual_predictions'):
            ref,b,lay,rec,strategy=self.case();row=ref['model_audit']['candidate']['annual'][0];row[key][0]=np.nextafter(row[key][0],np.inf).item()
            with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
    def test_score_pre_and_ordered_choice_changes_rejected(self):
        for key in ('scores','pre_scores','orders'):
            ref,b,lay,rec,strategy=self.case()
            if key=='orders':ref[key]['weak_pooled_tree'][1]['symbols'].reverse()
            else:ref[key]['weak_pooled_tree'][1,0]=np.nextafter(ref[key]['weak_pooled_tree'][1,0],np.inf)
            with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
    def test_calibration_must_remain_exact_incumbent(self):
        ref,b,lay,rec,strategy=self.case('calibration');ref['scores']['incumbent_local67'][1,0]+=.01
        with self.assertRaisesRegex(ValueError,'Ridge/incumbent'):invoke(ref,b,lay,rec,strategy,'calibration')
    def test_metadata_mismatch_and_duplicate_annual_year_rejected(self):
        for mode in ('symbols','year'):
            ref,b,lay,rec,strategy=self.case()
            if mode=='symbols':ref['meta']['symbols'][1][0]='CHANGED'
            else:ref['model_audit']['candidate']['annual'].append(deepcopy(ref['model_audit']['candidate']['annual'][0]))
            with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
    def test_source_layout_must_have_explicit_whole_cohort_and_U(self):
        for mode in ('bool','ids','month'):
            ref,b,lay,rec,strategy=self.case()
            if mode=='bool':lay['original_u']=lay['original_u'].astype(int)
            elif mode=='ids':lay['annual_cohorts'][0]['training_row_ids'][0]=True
            else:lay['annual_cohorts'][0]['training_months']=99999
            with self.assertRaises(ValueError):invoke(ref,b,lay,rec,strategy,'control')
    def test_inputs_are_not_mutated(self):
        ref,b,lay,rec,strategy=self.case();before=k._fingerprint((ref,b,lay,rec,strategy));invoke(ref,b,lay,rec,strategy,'control');self.assertEqual(before,k._fingerprint((ref,b,lay,rec,strategy)))

if __name__=='__main__':unittest.main(verbosity=2)
