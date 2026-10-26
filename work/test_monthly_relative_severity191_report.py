import unittest,math,copy,ast
from pathlib import Path
from unittest.mock import patch
import monthly_relative_severity191_report as r

def coverage_fixture():
    audit=dict(passed=True,independent_accounts=True,independent_of_production_model_and_engine=True,
        checked_paths=9,checked_slices=45,terminal_liquidation=False,
        accounting_source_sha256=r.sha(r.ACCOUNTING),paths={})
    result=dict(arms={})
    for arm in r.ARMS:
        audit['paths'][arm]={};result['arms'][arm]={}
        for path in r.PATHS:
            audit['paths'][arm][path]=dict(passed=True,monthly_executions=122,slices={key:{} for key in r.SLICES})
            result['arms'][arm][path]=dict(slices={key:{} for key in r.SLICES})
    return audit,result

class ReportTests(unittest.TestCase):
    def test_factors_include_initial_fee_and_unchanged_next_month(self):
        rows=[dict(date='2020-01-31',equity=99.),dict(date='2020-02-28',equity=110.),dict(date='2020-03-31',equity=105.)]
        q=r.monthly_factors(rows,['2020-01-31','2020-02-28'],['2020-02-28','2020-03-31'],100.)
        self.assertAlmostEqual(math.prod(q),1.05)
        self.assertAlmostEqual(q[0],1.1)
    def test_missing_boundary_is_error(self):
        with self.assertRaises(ValueError):r.monthly_factors([dict(date='2020-01-31',equity=100.)],['2020-01-31'],['2020-02-28'])
    def test_contributions_add_for_positive_negative_and_zero(self):
        for a,b in [([1.3,.8],[1.1,1.2]),([1.1,1.2],[1.1,1.2]),([1.4,1.3],[1.1,1.2])]:
            x=r.relative_contributions(a,b,.6,10)
            self.assertAlmostEqual(sum(x['cagr_contributions_pp'])/100,x['cagr_difference'])
            self.assertAlmostEqual(x['wealth_ratio'],math.prod(a)/math.prod(b))
    def test_measure_uses_initial_capital_for_first_day_drawdown(self):
        x=r.measure([dict(date='2020-01-01',equity=95.),dict(date='2020-02-01',equity=98.),dict(date='2020-03-01',equity=96.)],100.)
        self.assertAlmostEqual(x['max_drawdown'],-.05)
        self.assertAlmostEqual(x['wealth_multiple'],.96)
    def test_zero_volatility_and_single_return_have_undefined_sharpe(self):
        for values in ([100.,100.,100.],[100.,200.,400.],[100.,95.]):
            rows=[dict(date='2020-0%d-01'%(i+1),equity=v) for i,v in enumerate(values)]
            x=r.measure(rows,100.)
            self.assertIsNone(x['sharpe']);self.assertEqual(x['wealth_multiple'],values[-1]/100.)
        self.assertAlmostEqual(r.measure([dict(date='2020-01-01',equity=95.),dict(date='2020-02-01',equity=95.)],100.)['max_drawdown'],-.05)
    def test_nullable_comparison_only_allows_matching_undefined_sharpe(self):
        r.compare_metric(None,None,'sharpe','test');r.compare_metric(.1,.1,'cagr','test')
        for a,b,key in [(None,0.,'sharpe'),(0.,None,'sharpe'),(None,None,'cagr'),(float('nan'),float('nan'),'sharpe')]:
            with self.assertRaises(ValueError):r.compare_metric(a,b,key,'test')
    def test_markdown_displays_undefined_sharpe(self):
        metrics={arm:{path:{s:dict(cagr=0.,sharpe=None,max_drawdown=0.) for s in r.SLICES} for path in r.PATHS} for arm in r.ARMS}
        text=r.markdown(dict(selection=dict(admitted=False),metrics=metrics,all122_months=[]))
        self.assertIn('0.0000% / 未定义 / 0.0000%',text)
        self.assertNotIn('None',text)
    def test_exact_artifact_sets_match_registered_host_contract(self):
        tree=ast.parse((Path(r.__file__).parent/'monthly_relative_severity191_host.py').read_text())
        declared=None
        def set_value(node):
            if isinstance(node,ast.Set):return ast.literal_eval(node)
            if isinstance(node,ast.Name) and node.id=='PREPARED_FILES' and declared is not None:return declared
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.BitOr):return set_value(node.left)|set_value(node.right)
            raise ValueError('Unexpected prepared-set declaration')
        for assignment in tree.body:
            if isinstance(assignment,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PREPARED_FILES' for t in assignment.targets):
                declared=set_value(assignment.value)
        self.assertEqual(r.PREPARED_FILES,declared)
        self.assertEqual(len(r.PREPARED_FILES),13);self.assertEqual(len(r.RESULT_FILES),38)
        self.assertEqual(r.RESULT_FILES-{'RESULTS.json','INDEPENDENT_OUTCOME.json'},
            {a+'_'+p+'_'+k+'.json' for a in r.ARMS for p in r.PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')})
    def test_manifest_missing_extra_or_hash_mismatch_is_rejected(self):
        for expected in (r.PREPARED_FILES,r.RESULT_FILES):
            full={k:'digest' for k in expected}
            with patch.object(r,'sha',return_value='digest'):
                r.bind_manifest(Path('/synthetic'),dict(artifacts_sha256=full),expected)
                for changed in ({k:v for k,v in full.items() if k!=next(iter(full))},dict(full,unexpected='digest')):
                    with self.assertRaises(ValueError):r.bind_manifest(Path('/synthetic'),dict(artifacts_sha256=changed),expected)
            with patch.object(r,'sha',return_value='changed'):
                with self.assertRaises(ValueError):r.bind_manifest(Path('/synthetic'),dict(artifacts_sha256=full),expected)
    def test_nested_outcome_coverage_not_just_totals(self):
        audit,result=coverage_fixture();r.check_outcome_coverage(audit,result)
        edits=[lambda a,z:a['paths'].pop(r.ARMS[0]),
               lambda a,z:a['paths'][r.ARMS[0]].pop(r.PATHS[0]),
               lambda a,z:a['paths'][r.ARMS[0]][r.PATHS[0]]['slices'].pop(r.SLICES[0]),
               lambda a,z:a['paths'][r.ARMS[0]][r.PATHS[0]].update(passed=False),
               lambda a,z:a['paths'][r.ARMS[0]][r.PATHS[0]].update(monthly_executions=121),
               lambda a,z:a.update(independent_accounts=False),
               lambda a,z:a.update(terminal_liquidation=True),
               lambda a,z:a.update(accounting_source_sha256='wrong'),
               lambda a,z:z['arms'][r.ARMS[0]].update(extra={}),
               lambda a,z:z['arms'][r.ARMS[0]][r.PATHS[0]]['slices'].pop(r.SLICES[-1])]
        for edit in edits:
            a,z=copy.deepcopy(audit),copy.deepcopy(result);edit(a,z)
            with self.assertRaises(ValueError):r.check_outcome_coverage(a,z)
    def test_build_refuses_unbound_required_file_before_consumption(self):
        objects={'PLAN.json':{},'COMPLETION.json':dict(status='COMPLETED_INDEPENDENTLY_VERIFIED',plan_sha256='digest',observers_modified=False,protected_files=21444,result_completion_sha256='digest'),
          'results/COMPLETION.json':dict(status='COMPLETE',plan_sha256='digest',paths=9,slices=45,artifacts_sha256={k:'digest' for k in r.RESULT_FILES}),
          'prepared/COMPLETION.json':dict(status='COMPLETE',plan_sha256='digest',artifacts_sha256={k:'digest' for k in r.PREPARED_FILES if k!='MODEL_AUDIT.json'})}
        seen=[]
        def read(path):
            name=str(path.relative_to(r.OUT));seen.append(name);return objects[name]
        with patch.object(r,'sha',return_value='digest'),patch.object(r,'read',side_effect=read):
            with self.assertRaisesRegex(ValueError,'artifact manifest'):r.build_report('digest','digest')
        self.assertNotIn('prepared/MODEL_AUDIT.json',seen)
    def test_control_parity_requires_exact_fields_and_prior_bindings(self):
        valid=dict(passed=True,control_orders_exact=True,arrays_dtype_shape_bytes_exact=True,candidate190_arrays_loaded=False,candidate190_returns_read=False,
            prior_plan_sha256='a3e57a03ca5a1d88251ea5e1a28ad1e15ad362ae55c3645fde356917b6ed0729',
            prior_prepared_completion_sha256='544009a6583c44013680984fa6744839264ceacff6b086bfec12e248d5bd76a6',
            label_keys_checked=['dates','ends','symbols','control_target','base_good','good'],
            forecast_keys_checked=['control_prediction','control_rank','common_available'])
        r.check_control_parity(valid)
        for key in valid:
            bad=copy.deepcopy(valid);bad.pop(key)
            with self.assertRaises(ValueError):r.check_control_parity(bad)
        for key,val in [('passed',False),('control_orders_exact',False),('arrays_dtype_shape_bytes_exact',False),('candidate190_arrays_loaded',True),('candidate190_returns_read',True),('prior_plan_sha256','wrong'),('label_keys_checked',['control_target']),('forecast_keys_checked',['control_rank'])]:
            bad=dict(valid);bad[key]=val
            with self.assertRaises(ValueError):r.check_control_parity(bad)

if __name__=='__main__':unittest.main()
