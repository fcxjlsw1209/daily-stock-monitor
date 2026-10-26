"""Fabricated reporting-contract tests; no real strategy or account is read."""
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import date, timedelta
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import summarize_revenue_persistence156 as s


def fixture(cagr=.67):
    dates=[(date(2016,8,17)+timedelta(days=i)).isoformat() for i in range(122)]
    ends=dates[1:]+['2026-09-24']
    meta=dict(dates=dates,ends=ends)
    results={name:{mode:dict(metrics=dict(cagr=cagr if i==0 else .95,sharpe=2.,max_drawdown=-.2))
                   for mode in s.MODES} for i,name in enumerate(s.NAMES)}
    selection=dict(primary=s.NAMES[0],admitted=False)
    orders={name:[dict(date=d,signal_date=d,end=e,symbols=['A','B'],scores=[.1,.2]) for d,e in zip(dates,ends)]
            for name in s.NAMES}
    done=dict(round=156,directory='monthly_revenue_persistence156',plan_sha256='a'*64,
        status='completed_independently_verified',independent_execution_paths=12,protected_files=21444,
        trials=4,new_model_fits=0,all_arm_results=results,primary=s.NAMES[0],selection=selection,
        primary_cagr=cagr,primary_sharpe=2.,target_reached=cagr>=.7)
    completion_ref=dict(path='outputs/monthly_revenue_persistence156/COMPLETION.json',sha256='b'*64)
    completed=[dict(directory='old'+str(i),trials=6) for i in range(154)]+[dict(directory='previous',trials=54),
        dict(directory='monthly_revenue_persistence156',trials=4,primary=s.NAMES[0],target_reached=cagr>=.7,completion=completion_ref)]
    state=dict(completed=completed,registered_queue=[],current='monthly_revenue_persistence156',
               current_status='completed_independently_verified',latest_round_receipt=completion_ref)
    audit=dict(passed=True,stage='evaluate',plan_sha256='a'*64,producer_modules_imported=False,
        execution_paths_verified=12,total_monthly_orders_verified=488,models_fit=0,
        selected_name=s.NAMES[0],primary_cagr=cagr,primary_target_reached=cagr>=.7)
    frozen=dict(selection=selection)
    return done,state,results,orders,audit,frozen,meta,'a'*64,completion_ref


class ReportingTests(unittest.TestCase):
    def test_diagnostic_winner_does_not_set_target(self):
        data=fixture()
        self.assertFalse(s.validate_result_contract(*data))
        self.assertEqual(data[2][s.NAMES[3]]['same_close']['metrics']['cagr'],.95)

    def test_exact_seventy_percent_is_target(self):
        self.assertTrue(s.validate_result_contract(*fixture(.7)))

    def test_false_completion_target_rejected(self):
        data=list(fixture());data[0]['target_reached']=True
        with self.assertRaisesRegex(ValueError,'Target flag'):s.validate_result_contract(*data)

    def test_false_audit_target_rejected(self):
        data=list(fixture());data[4]['primary_target_reached']=True
        with self.assertRaisesRegex(ValueError,'Target flag'):s.validate_result_contract(*data)

    def test_wrong_completion_reference_rejected(self):
        data=list(fixture());data[1]['latest_round_receipt']={'path':'elsewhere','sha256':'c'*64}
        with self.assertRaisesRegex(ValueError,'bind this completion'):s.validate_result_contract(*data)

    def test_truncated_order_comparison_rejected(self):
        data=list(fixture());data[3][s.NAMES[1]].pop()
        with self.assertRaisesRegex(ValueError,'Truncated order'):s.validate_result_contract(*data)

    def test_frozen_selection_mismatch_rejected(self):
        data=list(fixture());data[5]=dict(selection=dict(primary=s.NAMES[3],admitted=True))
        with self.assertRaisesRegex(ValueError,'first88 primary'):s.validate_result_contract(*data)

    def test_unselected_primary_metrics_rejected(self):
        data=list(fixture());data[0]['primary_cagr']=.95
        with self.assertRaisesRegex(ValueError,'Selected metrics'):s.validate_result_contract(*data)

    def test_nonfinite_selected_metric_rejected(self):
        data=list(fixture());data[2][s.NAMES[0]]['same_close']['metrics']['cagr']=float('nan')
        with self.assertRaisesRegex(ValueError,'Finite selected'):s.validate_result_contract(*data)

    def test_main_uses_bound_dynamic_coverage_and_preserves_input_bytes(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as folder:
            root=Path(folder);out=root/'outputs/monthly_revenue_persistence156';out.mkdir(parents=True)
            src=root/'outputs/monthly_revenue_pilot_source156';src.mkdir()
            (root/'migration').mkdir();(root/'outputs/monthly_target70_research').mkdir()
            def put(p,v):p.write_bytes(s.encode(v))
            with patch.object(s,'ROOT',root),patch.object(s,'OUT',out):
                done,state,results,orders,audit,frozen,meta,_,_=fixture()
                coverage=dict(source_coverage_floor_passed=True,qualifying_joint_cutoffs=['x']*13,
                    segments=dict(all122=dict(issuers={n:dict(usable_rows=v) for n,v in [('MSFT',60),('ABBV',40),('AAPL',0)]}),
                                  early53=dict(issuers={n:dict(usable_rows=0) for n in ['MSFT','ABBV','AAPL']})))
                put(src/'COVERAGE_GATE.json',coverage)
                put(src/'MANIFEST.json',dict(outputs={'COVERAGE_GATE.json':s.ref(src/'COVERAGE_GATE.json')}))
                put(out/'PLAN.json',dict(inputs=dict(source_manifest=s.ref(src/'MANIFEST.json'))))
                plan_sha=s.ref(out/'PLAN.json')['sha256'];done['plan_sha256']=audit['plan_sha256']=plan_sha
                for name,value in [('RESULTS.json',results),('PRETRAIN_ORDERS.json',orders),('FROZEN.json',frozen),('META.json',meta)]:put(out/name,value)
                audit['artifact_hashes']={str(out/n):s.ref(out/n)['sha256'] for n in ['RESULTS.json','PRETRAIN_ORDERS.json','FROZEN.json','META.json']}
                audit['frozen_sha256']=s.ref(out/'FROZEN.json')['sha256'];audit['orders_sha256']=s.ref(out/'PRETRAIN_ORDERS.json')['sha256']
                put(out/'INDEPENDENT_OUTCOME.json',audit);put(out/'INDEPENDENT_PREPARE.json',{'passed':True})
                put(out/'PROTECTED_AFTER.json',dict(passed=True,checked=21444,verified=21444,changed=[],missing=[],read_or_identity_errors=[]))
                done.update(independent_outcome_sha256=s.ref(out/'INDEPENDENT_OUTCOME.json')['sha256'],protected_sha256=s.ref(out/'PROTECTED_AFTER.json')['sha256'])
                put(out/'COMPLETION.json',done)
                state['latest_round_receipt']=state['completed'][-1]['completion']=s.ref(out/'COMPLETION.json')
                state_path=root/'outputs/monthly_target70_research/CONTINUATION_STATUS.json';put(state_path,state)
                put(root/'migration/RESEARCH_RESUME.json',{'unrelated':'preserved'})
                before={p:p.read_bytes() for p in [*out.iterdir(),state_path]}
                with redirect_stdout(StringIO()):s.main(plan_sha)
                summary=s.read(out/'SUMMARY.json');text=(out/'SUMMARY.md').read_text()
                self.assertFalse(summary['target_reached'])
                self.assertIn('微软60个月、艾伯维40个月',text);self.assertIn('共同覆盖13个月',text)
                self.assertIn('该选择结果达到70%目标：否',text)
                self.assertEqual(s.read(root/'migration/RESEARCH_RESUME.json')['unrelated'],'preserved')
                self.assertTrue(all(p.read_bytes()==raw for p,raw in before.items()))


if __name__=='__main__':unittest.main()
