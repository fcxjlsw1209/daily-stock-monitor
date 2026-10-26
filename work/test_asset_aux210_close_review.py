"""Fabricated saved-JSON checks only; no actual210 artifacts or strategy imports."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import asset_aux210_close_review as r


def fixture(rows=3):
    names=list('ABCDEF'); meta=dict(symbols=[names.copy() for _ in range(rows)],etfs=list('DEF'))
    history=[]
    for i in range(rows):
        groups=[]
        for kind,ids in (('stock',[0,1,2]),('etf',[3,4,5])):
            constant=i%3==2 and kind=='stock'; two=i%3==1
            lo,hi=(.125,.125) if constant else (-.25,.25)
            qlo,qhi=(0.,0.) if constant else (-.5,.25) if two else (-.5,.5)
            delta=0. if not i or constant else .125
            difference=0. if constant or two else .125
            groups.append(dict(asset_type=kind,indices=ids,symbols=[names[j] for j in ids],size=3,
                support_min=lo,support_max=hi,candidate_min=qlo,candidate_max=qhi,control_min=qlo,control_max=qhi,
                distinct_levels=1 if constant else 2 if two else 3,constant=constant,two_levels=two,
                active_forecast_count=2,max_absolute_candidate_delta=delta,max_absolute_control_delta=delta,
                max_absolute_candidate_control_difference=difference))
        history.append(dict(query_index=i,symbols=names.copy(),groups=groups,active_forecast_count=4,
            max_absolute_candidate_delta=max(g['max_absolute_candidate_delta'] for g in groups),
            max_absolute_control_delta=max(g['max_absolute_control_delta'] for g in groups),
            max_absolute_candidate_control_difference=max(g['max_absolute_candidate_control_difference'] for g in groups)))
    audit=dict(events=rows,symbols_per_event=6,checked_cells=rows*6,rank_rows=2*rows,grouped_rank_rows=2*rows,
        constant_rows=sum(g['constant'] for x in history for g in x['groups']),
        two_level_rows=sum(g['two_levels'] for x in history for g in x['groups']),active_forecast_cells=4*rows,
        stock_cells=3*rows,etf_cells=3*rows,empty_groups=0,new_model_fits=0,predictive_model_fits=0,
        ranking_eligibility_filtered=False,new_availability_filter=False,portfolio_returns_calculated=False,event_history=history)
    return audit,meta


def receipt_fixture():
    audit,meta=fixture(122); counts,maxima=r.inspect_groups(audit,meta)
    shared=dict(**counts,original_array_count=27,new_model_fits=0,predictive_model_fits=0,
                new_predictive_model_fits=0,model_fit_accounting='Main only; no new fits')
    feature=dict(**{key:True for key in r.FEATURE_FLAGS},**counts,events=122,symbols_per_event=49,checked_cells=5978,
        computes_returns=False,numerical_tolerance=0.,support_bound_tolerance=1e-12,new_model_fits=0,predictive_model_fits=0)
    delta=dict(**{key:True for key in r.DELTA_FLAGS},max_absolute_delta=.5,
        per_arm_max_absolute_delta=maxima,observed_max_absolute_delta=max(maxima.values()),changed_cells={a:10 for a in r.ARMS[1:]})
    validation=dict(**shared,feature=feature,delta=delta,prefixes=123,feature_prefixes=list(range(123)))
    for key in ('feature_prefix_sources_truncated','feature_prefix_metadata_truncated','asset_type_metadata_checked',
                'original_cache_exact','original_orders_exact','original_arrays_unchanged',
                'original_core_metadata_and_support_preserved','single_original_episode'):validation[key]=True
    prepared=dict(**shared,new_returns_computed=False)
    return validation,prepared,dict(shared),counts,maxima


def selection_fixture():
    def stats(cagr):return dict(training=dict(cagr=cagr,sharpe=1.5,max_drawdown=-.3),early=dict(cagr=.3),recent=dict(cagr=.4))
    metrics={a:{p:stats(.5 if a!=r.ARMS[1] else .52) for p in ('same_close_zero','next_open_cost')} for a in r.ARMS}
    orders={a:[dict(date=str(i),end=str(i+1),symbols=['A','B','C']) for i in range(122)] for a in r.ARMS}
    for i in (1,2,53):orders[r.ARMS[1]][i]['symbols']=['A','B','D']
    return metrics,orders


def statistics_fixture():
    return dict(start='2020-01-01',end='2021-01-01',years=1.,final_equity=110000.,
        cagr=.1,sharpe=1.,max_drawdown=-.1,trading_sessions=252,total_return=.1)


class ReaderTests(unittest.TestCase):
    def test_no_npz_model_or_strategy_execution_in_reader(self):
        source=Path(r.__file__).read_text(); tree=ast.parse(source)
        imports=[x.module for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]
        imports += [a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]
        self.assertFalse(any(x and any(k in x for k in ('kernel','model','independent','numpy')) for x in imports))
        self.assertNotIn('np.load(',source)
        self.assertEqual((len(r.PREPARED_FILES),len(r.RESULT_FILES)),(7,38))
        with patch.object(r,'helpers',side_effect=AssertionError('No actual read')):
            for pin in (None,'bad','X'*64):
                with self.assertRaises(ValueError):r.review(pin,True)

    def test_saved_group_descriptors_and_counts(self):
        audit,meta=fixture(); counts,maxima=r.inspect_groups(audit,meta)
        self.assertEqual(counts['grouped_rank_rows'],6); self.assertEqual(counts['constant_rows'],1)
        self.assertEqual(counts['two_level_rows'],2); self.assertEqual(counts['etf_cells'],9)
        self.assertEqual(maxima,{a:.125 for a in r.ARMS[1:]})

    def test_identity_no_inference_and_membership_corruption(self):
        for mutation in ('wrong_type','wrong_index','duplicate','etfs_missing','stock_sum'):
            audit,meta=fixture()
            if mutation=='wrong_type':audit['event_history'][0]['groups'][0]['asset_type']='etf'
            elif mutation=='wrong_index':audit['event_history'][0]['groups'][1]['indices']=[0,4,5]
            elif mutation=='duplicate':meta['symbols'][0][0]='B'
            elif mutation=='etfs_missing':meta['etfs']=['D','E']
            else:audit['stock_cells']+=1
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):r.inspect_groups(audit,meta)

    def test_initial_and_constant_group_noop_required(self):
        for event,group in ((0,0),(2,0)):
            audit,meta=fixture();audit['event_history'][event]['groups'][group]['max_absolute_candidate_delta']=.01
            with self.assertRaises(ValueError):r.inspect_groups(audit,meta)

    def test_two_level_identity_and_affine_extrema(self):
        for field,value in (('max_absolute_candidate_control_difference',1e-16),('control_max',.2500001),
                            ('max_absolute_control_delta',.124)):
            audit,meta=fixture();audit['event_history'][1]['groups'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.inspect_groups(audit,meta)

    def test_bounds_counts_types_and_scope(self):
        for field,value in (('rank_rows',True),('grouped_rank_rows',7),('new_model_fits',1),
                            ('ranking_eligibility_filtered',True),('portfolio_returns_calculated',True)):
            audit,meta=fixture();audit[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.inspect_groups(audit,meta)
        for field,value in (('support_min',float('nan')),('support_max',.6),('active_forecast_count',4),
                            ('distinct_levels',4),('max_absolute_candidate_delta',.6)):
            audit,meta=fixture();audit['event_history'][1]['groups'][0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):r.inspect_groups(audit,meta)

    def test_singleton_and_empty_group_json(self):
        audit,meta=fixture(1);meta=dict(symbols=[['A']],etfs=[])
        row=audit['event_history'][0];row['symbols']=['A'];row['active_forecast_count']=0
        for kind,g in zip(('stock','etf'),row['groups']):
            present=kind=='stock';g.update(indices=[0] if present else [],symbols=['A'] if present else [],
                size=int(present),distinct_levels=int(present),constant=present,two_levels=False,active_forecast_count=0,
                support_min=.125 if present else None,support_max=.125 if present else None,
                candidate_min=0. if present else None,candidate_max=0. if present else None,
                control_min=0. if present else None,control_max=0. if present else None,
                max_absolute_candidate_delta=0.,max_absolute_control_delta=0.,max_absolute_candidate_control_difference=0.)
        row['max_absolute_candidate_control_difference']=0.
        audit.update(symbols_per_event=1,checked_cells=1,rank_rows=1,grouped_rank_rows=1,constant_rows=1,
            active_forecast_cells=0,stock_cells=1,etf_cells=0,empty_groups=1)
        counts,_=r.inspect_groups(audit,meta);self.assertEqual(counts['empty_groups'],1)
        row['groups'][1]['support_min']=0.
        with self.assertRaises(ValueError):r.inspect_groups(audit,meta)

    def test_receipt_reconciliation_and_all_required_flags(self):
        args=receipt_fixture();out=r.integration_receipts(*args);self.assertTrue(out['saved_npz_hash_verified_only'])
        for section,flags in (('feature',r.FEATURE_FLAGS),('delta',r.DELTA_FLAGS)):
            for flag in flags:
                args=receipt_fixture();args[0][section][flag]=False
                with self.subTest(flag=flag),self.assertRaises(ValueError):r.integration_receipts(*args)

    def test_receipt_prefix_count_or_fit_corruption(self):
        for mutation in ('meta','prefix','zero_fit','counts','tol','max'):
            args=list(receipt_fixture())
            if mutation=='meta':args[0]['feature_prefix_metadata_truncated']=False
            elif mutation=='prefix':args[0]['feature_prefixes']=[0,1,53,88,122]
            elif mutation=='zero_fit':args[2]['new_predictive_model_fits']=1
            elif mutation=='counts':args[1]['stock_cells']+=1
            elif mutation=='tol':args[0]['feature']['numerical_tolerance']=1e-12
            else:args[0]['delta']['observed_max_absolute_delta']=.5
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):r.integration_receipts(*args)

    def test_first88_selection_and_later_poison(self):
        h=r.helpers();metrics,orders=selection_fixture();expected=h.selection_from_saved(metrics,orders)
        self.assertTrue(expected['admitted'])
        for i in range(88,122):orders[r.ARMS[1]][i]['symbols']=['FUTURE']
        self.assertEqual(expected,h.selection_from_saved(metrics,orders))
        orders[r.ARMS[2]]=copy.deepcopy(orders[r.ARMS[1]])
        self.assertFalse(h.selection_from_saved(metrics,orders)['admitted'])

    def test_first88_gain_risk_era_and_control_cannot_promote(self):
        h=r.helpers()
        for field,value in (('cagr',.504),('sharpe',1.449),('max_drawdown',-.331)):
            metrics,orders=selection_fixture();metrics[r.ARMS[1]]['same_close_zero']['training'][field]=value
            self.assertEqual(h.selection_from_saved(metrics,orders)['primary'],r.ARMS[0])
        metrics,orders=selection_fixture();metrics[r.ARMS[1]]['next_open_cost']['early']['cagr']=.274
        self.assertFalse(h.selection_from_saved(metrics,orders)['admitted'])
        metrics,orders=selection_fixture();metrics[r.ARMS[2]]['same_close_zero']['training']['cagr']=.9
        self.assertEqual(h.selection_from_saved(metrics,orders)['primary'],r.ARMS[0])

    def test_zero_fees_and_isolated_prior_period_tolerance(self):
        result=dict(metrics=dict(**statistics_fixture(),fees_paid=0.),
                    slices={p:statistics_fixture() for p in r.BOUNDS})
        result['slices']['full122']=copy.deepcopy(result['metrics'])
        r.zero_fee_evidence(result,[dict(cost=0.)],[dict(fee=0.)])
        for target,key in ((result['metrics'],'fees_paid'),(result['slices']['full122'],'fees_paid')):
            target[key]=1e-12
            with self.assertRaises(ValueError):r.zero_fee_evidence(result,[dict(cost=0.)],[dict(fee=0.)])
            target[key]=0.
        with self.assertRaises(ValueError):r.zero_fee_evidence(result,[dict(cost=1e-12)],[dict(fee=0.)])
        with self.assertRaises(ValueError):r.zero_fee_evidence(result,[dict(cost=0.)],[dict(fee=1e-12)])
        result['slices']['early53']['fees_paid']=1e-12
        with self.assertRaises(ValueError):r.zero_fee_evidence(result,[dict(cost=0.)],[dict(fee=0.)])
        h=r.helpers();old=[dict(signal_date=str(i),execution_date=str(i),fee=1.,value=2.) for i in range(122)]
        new=copy.deepcopy(old);new[0]['fee']+=1e-12
        self.assertEqual(len(h.compare_holdings(old,new,'next_open_cost')),1)
        with self.assertRaises(ValueError):h.compare_holdings(old,new,'same_close_zero')
        new[0]['fee']+=1e-8
        with self.assertRaises(ValueError):h.compare_holdings(old,new,'next_open_cost')
        new=copy.deepcopy(old);new[0]['value']+=1e-15
        with self.assertRaises(ValueError):h.compare_holdings(old,new,'next_open_cost')

    def test_independent_statistics_cannot_be_empty_or_partial(self):
        h=r.helpers();ind=statistics_fixture();saved=dict(**ind,fees_paid=0.)
        r.complete_metric_evidence(h,ind,saved,'synthetic')
        for bad in ({},{k:v for k,v in ind.items() if k!='cagr'},dict(**ind,extra=0)):
            with self.assertRaises(ValueError):r.complete_metric_evidence(h,bad,saved,'synthetic')
        bad=dict(ind,cagr=.9)
        with self.assertRaises(ValueError):r.complete_metric_evidence(h,bad,saved,'synthetic')

    def test_registered_execution_and_cost_are_exact(self):
        for path,(execution,cost) in r.EXECUTIONS.items():
            metrics=dict(execution=execution,cost_per_side=cost)
            r.execution_evidence(metrics,path)
            for bad in (dict(metrics,execution='open' if execution=='close' else 'close'),
                        dict(metrics,cost_per_side=cost+1e-12),dict(metrics,cost_per_side=False)):
                with self.assertRaises(ValueError):r.execution_evidence(bad,path)

    def test_exact_manifests_reject_missing_extra_completion(self):
        h=r.helpers()
        with tempfile.TemporaryDirectory() as tmp:
            h.ROOT=Path(tmp).resolve();folder=h.ROOT/'saved';folder.mkdir()
            for names in (r.PREPARED_FILES,r.RESULT_FILES):
                for name in names:(folder/name).write_text('{}')
                mapping={name:r.sha(folder/name) for name in names}
                base=dict(status='COMPLETE',plan_sha256='a'*64,artifacts_sha256=mapping)
                def check(record):
                    path=folder/'COMPLETION.json';path.write_text(json.dumps(record))
                    return h.Evidence().manifest(folder,r.sha(path),'a'*64,names)
                check(base)
                for kind in ('missing','extra','completion'):
                    bad=copy.deepcopy(base)
                    if kind=='missing':del bad['artifacts_sha256'][sorted(names)[0]]
                    else:bad['artifacts_sha256']['EXTRA.json' if kind=='extra' else 'COMPLETION.json']='0'*64
                    with self.assertRaises(ValueError):check(bad)

    def test_declared_test_count_rejects_skip_duplicate_and_dynamic(self):
        good='import unittest\nclass Tests(unittest.TestCase):\n def test_a(self): pass\n def test_b(self): pass\n'
        self.assertEqual(r.declared_tests(good),2)
        for bad in (good+'def load_tests(*a): pass\n',good.replace('test_b','test_a'),
                    good.replace('class Tests','@unittest.skip("no")\nclass Tests'),good.replace('unittest.TestCase','Other')):
            with self.assertRaises(ValueError):r.declared_tests(bad)

    def test_dynamic_registered_code_and_three_test_files(self):
        h=r.helpers()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();(root/'work').mkdir()
            code_sha={};method_counts={}
            source='import unittest\nclass Tests(unittest.TestCase):\n def test_one(self): pass\n'
            for name in r.CODE_FILES:
                (root/name).write_text(source if name in r.TEST_FILES else '# frozen synthetic code\n')
                code_sha[name]=r.sha(root/name)
                if name in r.TEST_FILES:method_counts[name]=1
            tests=dict(passed=True,synthetic_only=True,tests=3,runtime=dict(python='3.9.6',numpy='2.0.2',pandas='2.3.3'),
                code_sha256={str(root/name):pin for name,pin in code_sha.items()})
            pins=dict(tests['code_sha256']);pins[str(root/r.TESTS)]='a'*64
            (root/r.COVERAGE).write_text('{}')
            coverage_pin=r.sha(root/r.COVERAGE);pins[str(root/r.COVERAGE)]=coverage_pin
            plan=dict(input_sha256=pins,runtime=tests['runtime'])
            code=dict(passed=True,unresolved_blockers=[],formal_test_receipt_sha256='a'*64,
                context_sha256=r.CONTEXT_SHA,code_sha256=code_sha,
                synthetic_test_methods=method_counts,synthetic_tests=3,input_sha256={r.COVERAGE:coverage_pin})
            for key in ('actual_features_computed','actual_model_fits_computed','actual_portfolio_returns_computed',
                        'production_files_modified','strategy_registered_by_reviewer'):code[key]=False
            h.ROOT=root
            with patch.object(r,'ROOT',root),patch.object(r,'COVERAGE_SHA',coverage_pin):
                self.assertEqual(r.test_code_binding(h.Evidence(),h,plan,tests,code)['tests'],3)
                for kind in ('count','code','file','coverage_missing','coverage_wrong'):
                    bad=copy.deepcopy(code)
                    if kind=='count':bad['synthetic_tests']=4
                    elif kind=='code':bad['code_sha256'][sorted(r.CODE_FILES)[0]]='b'*64
                    elif kind=='file':bad['synthetic_test_methods'].pop(r.TEST_FILES[0])
                    elif kind=='coverage_missing':bad['input_sha256'].pop(r.COVERAGE)
                    else:bad['input_sha256'][r.COVERAGE]='c'*64
                    with self.assertRaises(ValueError):r.test_code_binding(h.Evidence(),h,plan,tests,bad)
                bad_plan=copy.deepcopy(plan);bad_plan['input_sha256'].pop(str(root/r.COVERAGE))
                with self.assertRaises(ValueError):r.test_code_binding(h.Evidence(),h,bad_plan,tests,code)

    def test_markdown_discloses_same_author_and_saved_only_scope(self):
        metrics,orders=selection_fixture();h=r.helpers();selection=h.selection_from_saved(metrics,orders)
        summary={a:{p:{part:dict(cagr=.6,sharpe=1.5,max_drawdown=-.3) for part in r.BOUNDS} for p in r.PATHS} for a in r.ARMS}
        report=dict(selection=selection,summary=summary,failed_training_guards=[],holding_changes={})
        text=r.markdown(report);self.assertIn('未加载NPZ',text);self.assertIn('也是生产kernel作者',text)


if __name__=='__main__':
    unittest.main(verbosity=2)
