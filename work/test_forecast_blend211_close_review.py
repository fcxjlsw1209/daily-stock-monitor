"""Fabricated JSON and file pins only; never loads actual211 artifacts or numerical models."""
import ast
import calendar
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import forecast_blend211_close_review as r


def fixture(n=26):
    dates=[];ends=[]
    for i in range(n):
        y,m=2020+i//12,i%12+1;ny,nm=(y+1,1) if m==12 else (y,m+1)
        dates.append(f'{y:04d}-{m:02d}-{calendar.monthrange(y,m)[1]:02d}')
        ends.append(f'{ny:04d}-{nm:02d}-{calendar.monthrange(ny,nm)[1]:02d}')
    meta=dict(dates=dates,ends=ends,symbols=[['A','B'] for _ in dates])
    training=copy.deepcopy(meta);history=[];counts={k:0 for k in r.COUNTS};counts['cached_forecast_rows']=2*n
    for year in sorted({int(d[:4]) for d in dates}):
        cutoff=f'{year-1:04d}-12-31';start=f'{year-6:04d}-12-31'
        months=[]
        for i,(d,e) in enumerate(zip(dates,ends)):
            if not (d>start and e<=cutoff):continue
            months.append(dict(date=d,end=e,source_rows=2,invalid_label_rows=0,source_row_ids=[2*i,2*i+1],
                training_row_ids=[2*i,2*i+1],symbols=['A','B'],targets=[.25,.25],own_predictions=[1.,1.],
                peer_predictions=[0.,0.],numerator_terms=[.25,.25],denominator_terms=[1.,1.],usable_rows=2,
                accepted=True,numerator_mean=.25,denominator_mean=1.))
        k=len(months);status='learned' if k>=18 else 'cold_start'
        row=dict(year=year,cutoff=cutoff,lookback_exclusive_start=start,status=status,alpha=.25 if k>=18 else .5,
            unclipped_alpha=.25 if k>=18 else None,clipped=False,prediction_rows=2*sum(int(d[:4])==year for d in dates),
            skipped_initial_rows=0,joined_label_rows=2*k,invalid_label_rows=0,small_month_rows=0,
            calibration_rows=2*k,calibration_months=k,numerator=.25*k,denominator=float(k),
            latest_label_end=months[-1]['end'] if months else None,months=months)
        history.append(row);counts['annual_coefficient_fits' if k>=18 else 'cold_start_years']+=1
        for key in ('calibration_rows','calibration_months','joined_label_rows'):counts[key]+=row[key]
    audit=dict(method='annual_mature_asof_convex_forecast_blend',arms=list(r.ARMS),query_months=n,query_rows=2*n,
        symbols_per_event=2,min_months=18,lookback_calendar_years=5,month_equal_weights=True,
        source_forecasts_refitted=False,labels_recomputed=False,local_available_filtered=False,annual=history,counts=counts,
        new_model_fits=counts['annual_coefficient_fits'],predictive_model_fits=counts['annual_coefficient_fits'],new_ridge_fits=0,new_tree_fits=0)
    return audit,meta,training


def receipt_fixture():
    audit,meta,training=fixture();counts,_=r.model_history(audit,meta,training);n=len(meta['dates'])
    model=dict(**{k:True for k in r.MODEL_FLAGS},counts=counts,computes_returns=False,producer_modules_imported=False,
        atol=0.,query_rows=2*n,annual_cohorts=3,new_model_fits=1,predictive_model_fits=1,new_ridge_fits=0,new_tree_fits=0)
    slot=dict(**{k:True for k in r.SLOT_FLAGS},ready_rows={a:n-1 for a in r.ARMS[1:]},changed_cells={a:8 for a in r.ARMS[1:]})
    validation=dict(model=model,slot=slot,model_counts=counts,original_array_count=27,prefixes=n+1,model_prefixes=list(range(n+1)),model_prefix_evidence=[])
    for k in ('model_prefix_sources_truncated','model_prefix_metadata_truncated','model_prefix_original_arrays_truncated',
        'model_prefix_future_labels_removed','source_history_starts_at_original_zero','original_orders_exact','original_arrays_unchanged',
        'original_core_metadata_and_support_preserved','single_original_episode','no_new_score_budget'):validation[k]=True
    for j in range(n+1):
        a,_,_=fixture(j);day=meta['dates'][j-1] if j else None;cutoff=f'{int(day[:4])-1:04d}-12-31' if j else None
        ids=r.training_identities(training,cutoff) if j else []
        validation['model_prefix_evidence'].append(dict(prefix=j,query_rows=2*j,training_rows=len(ids),query_date=day,
            cutoff=cutoff,latest_label_end=max((x[2] for x in ids),default=None),source_history_starts_at_original_zero=True,counts=a['counts']))
    shared=dict(model_counts=counts,original_array_count=27,model_fit_accounting='Only learned scalar fits; no source fits')
    return validation,dict(**shared,new_returns_computed=False,source_model_refits=0),dict(shared),counts,meta,training


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
    def test_saved_only_reader_and_write_gate(self):
        source=Path(r.__file__).read_text();tree=ast.parse(source)
        imports=[x.module for x in ast.walk(tree) if isinstance(x,ast.ImportFrom)]
        imports += [a.name for x in ast.walk(tree) if isinstance(x,ast.Import) for a in x.names]
        self.assertFalse(any(x and any(k in x for k in ('kernel','model','independent','numpy')) for x in imports))
        self.assertNotIn('np.load(',source);self.assertEqual((len(r.PREPARED_FILES),len(r.RESULT_FILES)),(7,38))
        with patch.object(r,'helpers',side_effect=AssertionError('No actual reads')):
            for pin in (None,'bad','X'*64):
                with self.assertRaises(ValueError):r.review(pin,True)

    def test_annual_saved_history_and_fit_counts(self):
        audit,meta,train=fixture();counts,annual=r.model_history(audit,meta,train)
        self.assertEqual(counts['annual_coefficient_fits'],1);self.assertEqual(counts['cold_start_years'],2)
        self.assertEqual(annual[-1]['calibration_months'],23);self.assertEqual(annual[-1]['alpha'],.25)
        audit,meta,train=fixture(0);self.assertEqual(r.model_history(audit,meta,train)[0],dict.fromkeys(r.COUNTS,0))

    def test_exact_triple_identity_and_calendar_corruption(self):
        for mode in ('sourceid','trainid','symbol','end','date','cutoff','start','latest'):
            audit,meta,train=fixture();a=audit['annual'][-1];m=a['months'][0]
            if mode=='sourceid':m['source_row_ids'][0]=2
            elif mode=='trainid':m['training_row_ids'][0]=2
            elif mode=='symbol':m['symbols'][0]='C'
            elif mode=='end':m['end']='2020-03-31'
            elif mode=='date':m['date']='2020-02-29'
            elif mode=='cutoff':a['cutoff']='2022-01-31'
            elif mode=='start':a['lookback_exclusive_start']='2016-01-01'
            else:a['latest_label_end']='2022-01-31'
            with self.subTest(mode=mode),self.assertRaises(ValueError):r.model_history(audit,meta,train)

    def test_invalid_finite_vectors_counts_and_flags_fail(self):
        for mode in ('nan','inf','duplicate','vector','accepted','count','refit','local','zerofit'):
            audit,meta,train=fixture();m=audit['annual'][-1]['months'][0]
            if mode=='nan':m['targets'][0]=float('nan')
            elif mode=='inf':m['numerator_terms'][0]=float('inf')
            elif mode=='duplicate':m['source_row_ids'][1]=0
            elif mode=='vector':m['own_predictions'].pop()
            elif mode=='accepted':m['accepted']=False
            elif mode=='count':audit['counts']['calibration_rows']+=1
            elif mode=='refit':audit['source_forecasts_refitted']=True
            elif mode=='local':audit['local_available_filtered']=True
            else:audit['new_model_fits']=0
            with self.subTest(mode=mode),self.assertRaises(ValueError):r.model_history(audit,meta,train)

    def test_half_fallback_and_convex_saved_descriptors(self):
        for mode in ('coldweight','coldunclip','status','bound','clip','zerodenom'):
            audit,meta,train=fixture();a=audit['annual'][-1]
            if mode=='coldweight':audit['annual'][0]['alpha']=.4
            elif mode=='coldunclip':audit['annual'][0]['unclipped_alpha']=.5
            elif mode=='status':a['status']='cold_start'
            elif mode=='bound':a['alpha']=1.01
            elif mode=='clip':a['clipped']=True
            else:a['denominator']=0.
            with self.subTest(mode=mode),self.assertRaises(ValueError):r.model_history(audit,meta,train)
        audit,meta,train=fixture();a=audit['annual'][-1];a.update(alpha=0.,unclipped_alpha=-.1,clipped=True)
        audit['counts']['clipped_coefficient_years']=1;r.model_history(audit,meta,train)

    def test_oracle_receipts_required_flags_and_nonzero_fits(self):
        args=receipt_fixture();self.assertTrue(r.integration_receipts(*args)['saved_npz_hash_verified_only'])
        for section,flags in (('model',r.MODEL_FLAGS),('slot',r.SLOT_FLAGS)):
            for flag in flags:
                args=receipt_fixture();args[0][section][flag]=False
                with self.subTest(flag=flag),self.assertRaises(ValueError):r.integration_receipts(*args)

    def test_all_physical_prefixes_maturity_and_readiness(self):
        for mode in ('prefix','training','cutoff','latest','history','count','ready','sourcefit','tol'):
            args=list(receipt_fixture());v=args[0]
            if mode=='prefix':v['model_prefixes']=[0,1,26]
            elif mode=='training':v['model_prefix_evidence'][1]['training_rows']=2
            elif mode=='cutoff':v['model_prefix_evidence'][-1]['cutoff']='2022-12-31'
            elif mode=='latest':v['model_prefix_evidence'][-1]['latest_label_end']='2022-01-31'
            elif mode=='history':v['source_history_starts_at_original_zero']=False
            elif mode=='count':v['model']['counts']=dict(v['model']['counts'],cached_forecast_rows=2)
            elif mode=='ready':v['slot']['ready_rows'][r.ARMS[1]]=1
            elif mode=='sourcefit':args[1]['source_model_refits']=1
            else:v['model']['atol']=1e-12
            with self.subTest(mode=mode),self.assertRaises(ValueError):r.integration_receipts(*args)

    def test_first88_only_and_two_references(self):
        h=r.helpers();metrics,orders=selection_fixture();expected=r.selection_from_saved(h,metrics,orders)
        self.assertTrue(expected['admitted']);self.assertIn('close_vs_half_control',expected['guards'])
        for i in range(88,122):orders[r.ARMS[1]][i]['symbols']=['FUTURE']
        self.assertEqual(expected,r.selection_from_saved(h,metrics,orders))
        orders[r.ARMS[2]]=copy.deepcopy(orders[r.ARMS[1]])
        self.assertFalse(r.selection_from_saved(h,metrics,orders)['admitted'])
        for field,value in (('cagr',.504),('sharpe',1.449),('max_drawdown',-.331)):
            metrics,orders=selection_fixture();metrics[r.ARMS[1]]['same_close_zero']['training'][field]=value
            self.assertEqual(r.selection_from_saved(h,metrics,orders)['primary'],r.ARMS[0])

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
        metrics,orders=selection_fixture();selection=r.selection_from_saved(r.helpers(),metrics,orders)
        summary={a:{p:{part:dict(cagr=.6,sharpe=1.5,max_drawdown=-.3) for part in r.BOUNDS} for p in r.PATHS} for a in r.ARMS}
        report=dict(selection=selection,summary=summary,failed_training_guards=[])
        text=r.markdown(report);self.assertIn('未加载NPZ',text);self.assertIn('也是生产model作者',text)


if __name__=='__main__':unittest.main(verbosity=2)
