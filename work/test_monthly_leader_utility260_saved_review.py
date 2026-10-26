"""Only fabricated JSON/ledger/certificate tests; never reads real phase outputs."""
import ast
import calendar
from copy import deepcopy
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import monthly_leader_utility260_saved_review as r

r.SCHEMAS,r.FLAGS=r.schema_flags()


def meta_fixture():
    def last(y,m):
        d=date(y,m,calendar.monthrange(y,m)[1])
        while d.weekday()>4:d-=timedelta(days=1)
        return d.isoformat()
    dates=['2016-08-17']+[last(2016+(7+j)//12,(7+j)%12+1) for j in range(121)]
    return dict(dates=dates,ends=dates[1:]+['2026-09-24'],symbols=[['S%02d'%i for i in range(49)] for _ in dates],etfs=[])


def modelcounts(n,meta):
    c=dict.fromkeys(r.SCHEMAS['MODEL_COUNT_FIELDS'],0)
    c.update(years=len(set(d[:4] for d in meta['dates'][1:n])),cold_query_events=max(0,n-1),partial_events=min(1,n))
    return c


def choicecounts(n):
    c=dict.fromkeys(r.SCHEMAS['COUNT_FIELDS'],0);c.update(events=n,symbols_per_event=49,partial_events=min(1,n),cold_events=max(0,n-1),incumbent_episode_calls=1,intercept_episode_calls=1,state_episode_calls=1);return c


def calendar_fixture():
    axis=[];d=date(2008,1,2)
    while d<=date(2026,9,24):
        if d.weekday()<5:axis.append(d.isoformat())
        d+=timedelta(days=1)
    return {'sessions':axis}


CALENDAR=calendar_fixture()
CALENDAR_POSITIONS={d:i for i,d in enumerate(CALENDAR['sessions'])}


def market_record(day,partial=False):
    pos=CALENDAR_POSITIONS[day];axis=CALENDAR['sessions']
    return dict(date=day,initial_partial=partial,spy_start=None if partial else axis[pos-252],relative_start=None if partial else axis[pos-126],price_end=None if partial else day,spy_price_count=0 if partial else 253,xly_price_count=0 if partial else 127,xlp_price_count=0 if partial else 127,values=[0.,0.,0.] if partial else [.1,.2,.3])


def market(meta,n,training=()):
    records=[market_record(d,i==0) for i,d in enumerate(meta['dates'][:n])]
    c=dict(training_events=len(training),query_events=n,partial_events=min(1,n),state_vectors=len(training)+max(0,n-1),price_observations=507*(len(training)+max(0,n-1)))
    return dict(schema='monthly_leader_utility260_market_source_v1',columns=list(r.SCHEMAS['COLUMNS']),training=[market_record(d) for d in training],queries=records,counts=c)


def receipt(kind,n,meta):
    flags=r.FLAGS[{'early_incumbent':'INCUMBENT_FLAGS','incumbent':'INCUMBENT_FLAGS','inputs':'INPUT_FLAGS','targets':'TARGET_FLAGS','source':'SOURCE_FLAGS','models':'MODEL_FLAGS','selection':'SELECTION_FLAGS'}[kind]]
    x=dict(passed=True,checked_events=n,producer_modules_imported=False,computes_returns=False,old_model_solves=0,conditional_ridge_solves=0,independent_canonical_solves=0,independent_svd_solves=0,original_episode_calls=1 if kind in ('early_incumbent','incumbent') else 3 if kind=='selection' else 0)
    x.update({k:True for k in flags})
    if kind in ('early_incumbent','incumbent','selection'):
        x.update(ordered_choices_checked=3*n if kind=='selection' else n,full_scores_supplied=True,full_pre_scores_supplied=True,numerical_tolerance=0.)
        if kind=='selection':x.update(counts=choicecounts(n),full_bundle_supplied=True,full_audit_supplied=True)
        else:x['candidate_selection_computed']=False
    else:
        x['exact_discrete']=True
        if kind in ('inputs','targets'):x['numerical_tolerance']=0.
        else:x.update(continuous_atol=2e-10,continuous_rtol=2e-10,rank_tolerance=0.)
        x[{'inputs':'full_input_audit_supplied','targets':'full_target_audit_supplied','source':'full_source_audit_supplied','models':'full_model_audit_supplied'}[kind]]=True
        if kind=='inputs':x['counts']=dict(training_rows=0,training_months=0,query_rows=n*49)
        if kind=='targets':x['counts']=dict.fromkeys(r.SCHEMAS['TARGET_COUNT_FIELDS'],0)
        if kind=='source':x.update(counts=market(meta,n)['counts'],source_audit=market(meta,n))
        if kind=='models':x.update(counts=modelcounts(n,meta),intercept_estimates=0)
    return x


def proof_fixture():
    meta=meta_fixture();annual=[]
    for year in sorted(set(int(d[:4]) for d in meta['dates'][1:])):
        annual.append(dict(year=year,cutoff=f'{year-1:04d}-12-31',lower_bound=f'{year-6:04d}-12-31',training_indices=[],query_indices=[i for i,d in enumerate(meta['dates']) if i and int(d[:4])==year],status='insufficient_months',weights=[],weight_sum=0.,mean_x=None,sd_x=None,scale_x=None,mean_d=None,target_sd=None,gram=None,rhs=None,beta=None,fit_performed=False))
    inp={k:[] for k in r.SCHEMAS['INPUT_KEYS'] if k not in ('plain','event_plain','labels')}
    inp.update(schema='monthly_leader_utility260_inputs_v1',n=122,width=49,offsets=[0],event_dates=meta['dates'],event_symbols=meta['symbols'])
    ma=dict(model=dict(schema='monthly_leader_utility260_model_audit_v1',annual=annual,counts=modelcounts(122,meta),market_columns=list(r.SCHEMAS['COLUMNS']),original_models_refitted=False),inputs=inp,target=dict(schema='monthly_leader_utility260_targets_v1',records=[],counts=dict.fromkeys(r.SCHEMAS['TARGET_COUNT_FIELDS'],0)),market=market(meta,122))
    orders={a:[dict(date=d,end=e,signal_date=d,symbols=[],scores=[]) for d,e in zip(meta['dates'],meta['ends'])] for a in r.ARMS}
    events=[dict(query_index=i,date=d,partial=i==0,active=False,eligible_indices=[],intercept_p=.5,state_p=.5,intercept_delta=[0.]*49,state_delta=[0.]*49,incumbent_choices=[],intercept_choices=[],state_choices=[]) for i,d in enumerate(meta['dates'])]
    audit=dict(schema='monthly_leader_utility260_decisions_v1',counts=choicecounts(122),events=events,incumbent_gate=receipt('incumbent',122,meta))
    def run(n):
        p={k:receipt(k,n,meta) for k in ('early_incumbent','inputs','targets','source','models','incumbent','selection')};p['model_counts']=modelcounts(n,meta);return p
    proof=run(122);physical=[dict(prefix=n,exact=True,model_recomputed=True,**run(n)) for n in (0,1,53,88)]
    physical.append(dict(prefix=122,exact=True,model_recomputed=False,reuses_main_verified_full=True,**deepcopy(proof)))
    cached=[dict(prefix=n,exact=True,incumbent=receipt('incumbent',n,meta),selection=receipt('selection',n,meta)) for n in range(123)]
    proof.update(physical_prefixes=physical,cached_prefixes=cached,all123_order_prefixes=True,original_arrays_unchanged=True,original_training_unchanged=True,saved_model_audit_sha256=r.digest_record(ma))
    proof['fit_execution_accounting']=r.execution_accounting(proof)
    return proof,ma,audit,meta,orders


def training_proof_fixture():
    p,m,d,meta,orders=proof_fixture()
    dates=['2014-12-31','2015-12-31','2020-12-31'];ends=['2015-01-30','2016-01-29','2021-01-29']
    rows=[]
    for i,(day,end) in enumerate(zip(dates,ends)):
        rows.append(dict(month_index=i,signal_date=day,label_end=end,row_indices=[2*i,2*i+1],domain_indices=[2*i,2*i+1],raw_choices=['A','B'],risk_choices=['A','B'],selected_union=['A','B'],raw_only=[],risk_only=[],k=2,valid=True,reason='identical_sets',value=0.))
    def tc(k):
        c=dict.fromkeys(r.SCHEMAS['TARGET_COUNT_FIELDS'],0);c.update(months=k,source_rows=2*k,domain_rows=2*k,identical_set_months=k,valid_months=k,selected_label_observations=2*k);return c
    inp=m['inputs'];inp.update(training_row_ids=list(range(6)),dates=[x for x in dates for _ in range(2)],ends=[x for x in ends for _ in range(2)],symbols=['A','B']*3,offsets=[0,2,4,6],eligible=[True]*6,common_good=[True]*6)
    m['target'].update(records=rows,counts=tc(3));m['market']=market(meta,122,dates)
    for a in m['model']['annual']:
        a['training_indices']=[i for i,(day,end) in enumerate(zip(dates,ends)) if a['lower_bound']<day<=a['cutoff'] and end<=a['cutoff']]
    def mc(n,k):
        c=modelcounts(n,meta);years={int(day[:4]) for day in meta['dates'][1:n]}
        c.update(training_months=k,valid_target_months=k,annual_training_month_observations=sum(len(a['training_indices']) for a in m['model']['annual'] if a['year'] in years));return c
    m['model']['counts']=mc(122,3)
    for run,n,k in [(p,122,3)]+[(x,x['prefix'],{0:0,1:0,53:2,88:3,122:3}[x['prefix']]) for x in p['physical_prefixes']]:
        run['inputs']['counts']=dict(training_rows=2*k,training_months=k,query_rows=49*n)
        run['targets']['counts']=tc(k);run['source'].update(counts=market(meta,n,dates[:k])['counts'],source_audit=market(meta,n,dates[:k]))
        run['model_counts']=mc(n,k);run['models']['counts']=mc(n,k)
    p['saved_model_audit_sha256']=r.digest_record(m);p['fit_execution_accounting']=r.execution_accounting(p)
    return p,m,d,meta,orders


def metrics_fixture():
    _,_,_,meta,orders=proof_fixture();first=date.fromisoformat(meta['dates'][0]);last=date.fromisoformat(meta['ends'][-1]);axis=[];day=first
    while day<=last:
        if day.weekday()<5:axis.append(day.isoformat())
        day+=timedelta(days=1)
    result={'arms':{a:{} for a in r.ARMS}};exports={};train={a:{} for a in r.ARMS};ind=dict(passed=True,checked_paths=9,checked_slices=45,checked_training_groups=18,independent_accounts=True,independent_of_production_model_and_engine=True,terminal_liquidation=False,paths={a:{} for a in r.ARMS})
    for arm in r.ARMS:
        for path in r.PATHS:
            data=dict(DAILY=[dict(date=d,equity=100000.) for d in axis],DAILY_POSITIONS=[dict(date=d,equity=100000.,cash=100000.,adjusted_units={},market_values={}) for d in axis],TRADES=[],HOLDINGS=[])
            for o in orders[arm]:
                execution=o['date'] if path=='same_close_zero' else axis[axis.index(o['date'])+1]
                data['HOLDINGS'].append(dict(execution_date=execution,signal_date=o['date'],symbols=[],scores=[],fee=0.))
            slices,sds=r.saved_nav_slices(data['DAILY'],meta);top=dict(slices['full122'],fees_paid=0.,dollar_turnover=0.,trade_rows=0);slices['full122']=top.copy()
            result['arms'][arm][path]=dict(metrics=top,slices=slices,accounting=dict(exported_fill_ledger_matches=True,terminal_liquidation=False))
            ind['paths'][arm][path]=dict(passed=True,monthly_executions=122,slices={s:{k:m[k] for k in r.STAT_KEYS} for s,m in slices.items()},fees_paid=0.,dollar_turnover=0.,trade_rows=0)
            exports[arm,path]=data
            if path in r.PATHS[:2]:
                train[arm][path]={short:dict(slices[long]) for short,long in [('training','first88'),('early','early53'),('recent','recent35')]}
                for g in train[arm][path].values():g.update(daily_return_sd=0.,sharpe_status='undefined_sharpe',sharpe_reason='zero_daily_return_sd')
    return result,exports,orders,meta,{'sessions':axis},train,ind


class ReaderTests(unittest.TestCase):
    def test_imports_are_standard_library_only(self):
        tree=ast.parse(Path(r.__file__).read_text());allowed={'argparse','ast','collections','datetime','hashlib','json','math','pathlib','platform','statistics'}
        imports={n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}|{a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names}
        self.assertTrue(imports<=allowed)
    def test_schema_literals_match_registered_interfaces(self):
        self.assertEqual(len(r.SCHEMAS['MODEL_COUNT_FIELDS']),10);self.assertEqual(len(r.SCHEMAS['TARGET_RECORD_KEYS']),14)
    def test_constant_parser_never_executes_source(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'x.py';p.write_text("raise RuntimeError('MUST NOT EXECUTE')\nKEY=('a','b')\n")
            self.assertEqual(r.constants(p,('KEY',))['KEY'],('a','b'))
    def test_nav_formula_nonconstant(self):
        v,sd=r.nav_metrics(['2020-01-01','2020-07-01','2021-01-01'],[100000.,80000.,120000.])
        self.assertEqual(v['max_drawdown'],-.19999999999999996);self.assertAlmostEqual(v['total_return'],.2);self.assertGreater(sd,0.)
    def test_nav_constant_has_undefined_sharpe(self):
        v,sd=r.nav_metrics(['2020-01-01','2020-01-02','2020-01-03'],[100000.]*3);self.assertIsNone(v['sharpe']);self.assertEqual(sd,0.)
    def test_bad_nav_rejected(self):
        for values in ([100000.,0.],[100000.,float('nan')],[100000.,True]):
            with self.assertRaises(ValueError):r.nav_metrics(['2020-01-01','2020-01-02'],values)
    def test_nine_paths_45_slices_18_training_recomputed(self):
        result,exports,orders,meta,cal,train,ind=metrics_fixture();out,new=r.recompute_saved_metrics(result,exports,orders,meta,cal,train)
        self.assertEqual((out['paths'],out['slices'],out['training_groups']),(9,45,18));self.assertEqual(len(new),3)
        class Fake:
            def read(self,p):
                for (arm,path),data in exports.items():
                    for kind in r.KINDS:
                        if p.name==arm+'_'+path+'_'+kind+'.json':return data[kind]
                raise AssertionError(p)
        accounting,_=r.accounting(Fake(),result,ind,train);self.assertEqual(accounting['metric_fields_compared'],810)
    def test_saved_metric_tamper_rejected(self):
        result,exports,orders,meta,cal,train,_=metrics_fixture();result['arms'][r.ARMS[2]][r.PATHS[0]]['slices']['first88']['cagr']=.001
        with self.assertRaises(ValueError):r.recompute_saved_metrics(result,exports,orders,meta,cal,train)
    def test_zero_fee_tiny_nonzero_rejected(self):
        _,exports,orders,meta,cal,_,_=metrics_fixture();data=exports[r.ARMS[0],r.PATHS[0]];data['HOLDINGS'][3]['fee']=1e-30
        with self.assertRaises(ValueError):r.exported_ledger_check(data,orders[r.ARMS[0]],cal['sessions'],r.PATHS[0],[])
    def test_intra_month_cash_change_rejected(self):
        _,exports,orders,meta,cal,_,_=metrics_fixture();data=exports[r.ARMS[0],r.PATHS[0]];data['DAILY_POSITIONS'][2]['cash']+=1.
        with self.assertRaises(ValueError):r.exported_ledger_check(data,orders[r.ARMS[0]],cal['sessions'],r.PATHS[0],[])
    def test_full_prep_evidence_and_actual_work(self):
        proof,ma,audit,meta,orders=proof_fixture();r.preparation_evidence(proof,ma,audit,meta,CALENDAR)
        summary=r.source_and_model_summary(ma,meta,CALENDAR);d=r.decision_summary(audit,meta,orders)
        self.assertEqual(summary['active_query_indices'],d['active_query_indices']);e=proof['fit_execution_accounting']
        self.assertEqual(e['fresh_model_invocations'],5);self.assertEqual(e['producer_original_episode_calls'],389);self.assertEqual(e['independent_actual_work']['original_episode_calls'],517)
        self.assertEqual(e['producer_conditional_ridge_solves'],0)
    def test_missing_cached_prefix_rejected(self):
        p,m,d,meta,_=proof_fixture();del p['cached_prefixes'][52]
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_recounted_full122_rejected(self):
        p,m,d,meta,_=proof_fixture();p['physical_prefixes'][-1]['model_recomputed']=True
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_forged_scope_flag_rejected(self):
        p,m,d,meta,_=proof_fixture();p['source'][r.FLAGS['SOURCE_FLAGS'][0]]=False
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_full_model_digest_binding(self):
        p,m,d,meta,_=proof_fixture();m['inputs']['schema']='forged'
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_missing_physical_source_payload_rejected(self):
        p,m,d,meta,_=proof_fixture();del p['physical_prefixes'][2]['source']['source_audit']
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_source_receipt_counts_must_match_payload(self):
        p,m,d,meta,_=proof_fixture();p['physical_prefixes'][2]['source']['source_audit']['counts']['price_observations']+=1
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_query_rows_exact_original49(self):
        p,m,d,meta,_=proof_fixture();p['physical_prefixes'][2]['inputs']['counts']['query_rows']-=1
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_nonempty_teacher_full_and_physical_year_cutoffs(self):
        p,m,d,meta,_=training_proof_fixture();r.source_and_model_summary(m,meta,CALENDAR);r.preparation_evidence(p,m,d,meta,CALENDAR)
        self.assertEqual([len(r.prefix_teacher_rows(m,meta,n)) for n in (0,1,2,53,88,122)],[0,0,1,2,3,3])
        self.assertEqual(p['physical_prefixes'][2]['inputs']['counts'],dict(training_rows=4,training_months=2,query_rows=49*53))
    def test_physical_training_scope_tamper_rejected(self):
        for key in ('training_rows','training_months'):
            p,m,d,meta,_=training_proof_fixture();p['physical_prefixes'][2]['inputs']['counts'][key]+=1
            with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_physical_teacher_counts_cannot_claim_query_months(self):
        p,m,d,meta,_=training_proof_fixture();p['physical_prefixes'][2]['targets']['counts']['source_rows']+=1
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_input_month_rows_bound_to_teacher_metadata(self):
        _,m,_,meta,_=training_proof_fixture();m['inputs']['ends'][2]='2016-02-01'
        with self.assertRaises(ValueError):r.source_and_model_summary(m,meta,CALENDAR)
    def test_physical_market_values_exact_saved_prefix(self):
        p,m,d,meta,_=proof_fixture();p['physical_prefixes'][2]['source']['source_audit']['queries'][5]['values'][0]+=.01
        with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_cached_and_physical_counts_recomputed_from_events(self):
        for key,index in [('cached_prefixes',53),('physical_prefixes',2)]:
            p,m,d,meta,_=proof_fixture();p[key][index]['selection']['counts']['eligible_cells']+=1
            with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_official_window_one_session_shift_rejected(self):
        for key in ('spy_start','relative_start','price_end'):
            p,m,d,meta,_=proof_fixture();record=p['physical_prefixes'][2]['source']['source_audit']['queries'][5]
            record[key]=CALENDAR['sessions'][CALENDAR_POSITIONS[record[key]]+1]
            with self.assertRaises(ValueError):r.preparation_evidence(p,m,d,meta,CALENDAR)
    def test_wrong_main_clock_not_rescued_by_same_prefix_payload(self):
        _,m,_,meta,_=proof_fixture();record=m['market']['queries'][5];record['spy_start']=CALENDAR['sessions'][CALENDAR_POSITIONS[record['spy_start']]+1]
        with self.assertRaises(ValueError):r.source_and_model_summary(m,meta,CALENDAR)
    def test_partial_has_no_fabricated_source_window(self):
        _,m,_,meta,_=proof_fixture();m['market']['queries'][0]['spy_start']='2015-01-02'
        with self.assertRaises(ValueError):r.source_and_model_summary(m,meta,CALENDAR)
    def test_original_fit_count_cannot_hide(self):
        meta=meta_fixture();proof=receipt('models',122,meta);proof['old_model_solves']=1
        with self.assertRaises(ValueError):r.receipt(proof,122,'models')
    def test_teacher_selected_union_metadata_tamper(self):
        _,m,_,meta,_=proof_fixture();m['target']['counts']['selected_label_observations']=1
        with self.assertRaises(ValueError):r.source_and_model_summary(m,meta,CALENDAR)
    def test_annual_cutoff_tamper(self):
        _,m,_,meta,_=proof_fixture();m['model']['annual'][0]['cutoff']='2016-12-31'
        with self.assertRaises(ValueError):r.source_and_model_summary(m,meta,CALENDAR)
    def test_cold_changed_order_rejected(self):
        _,_,audit,meta,orders=proof_fixture();orders[r.ARMS[2]][1]['symbols']=['S00']
        with self.assertRaises(ValueError):r.decision_summary(audit,meta,orders)
    def test_decision_descriptive_count_tamper(self):
        _,_,audit,meta,orders=proof_fixture();audit['counts']['state_changed_holdings_events']=1
        with self.assertRaises(ValueError):r.decision_summary(audit,meta,orders)
    def test_only_main_cagr_gets_declared_epsilon(self):
        g=dict(training=dict(cagr=.1,sharpe=1.,max_drawdown=-.1),early=dict(cagr=.1),recent=dict(cagr=.1));c=deepcopy(g);c['training']['cagr']=.1+.005-5e-13
        self.assertTrue(r.guard(c,g,.005));c['early']['cagr']=.1-.025-1e-14;self.assertFalse(r.guard(c,g,.005))
    def test_only_first88_no_cost_or_later_rescue(self):
        group=dict(training=dict(cagr=.1,sharpe=1.,max_drawdown=-.1,daily_return_sd=.01,sharpe_status='defined',sharpe_reason=None),early=dict(cagr=.1),recent=dict(cagr=.1))
        train={a:{p:deepcopy(group) for p in r.PATHS[:2]} for a in r.ARMS}
        changed={r.ARMS[2]+'_vs_'+ref:dict(months=[dict(index=i) for i in [1,2,53,100]]) for ref in r.ARMS[:2]}
        result=r.saved_selection(train,changed);self.assertFalse(result['admitted']);self.assertEqual(result['primary'],r.ARMS[0]);self.assertFalse(result['cost_used']);self.assertEqual(result['changed_months_by_reference'][r.ARMS[0]],[1,2,53])
        for p in r.PATHS[:2]:train[r.ARMS[2]][p]['training']['cagr']=.106
        self.assertTrue(r.saved_selection(train,changed)['admitted'])
        changed[r.ARMS[2]+'_vs_'+r.ARMS[1]]['months']=[];self.assertFalse(r.saved_selection(train,changed)['admitted'])
    def test_undefined_incumbent_sharpe_invalid(self):
        _,_,_,_,_,train,_=metrics_fixture()
        with self.assertRaises(ValueError):r.required_metrics(train[r.ARMS[0]],True)
    def test_evidence_json_no_array_decode_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();p=root/'x.json';p.write_text('{"a":1}');h=r.digest(p)
            with patch.object(r,'ROOT',root):
                e=r.Evidence();self.assertEqual(e.read(p,h),{'a':1});p.write_text('{"a":2}')
                with self.assertRaises(ValueError):e.read(p,h)
                a=root/'x.npz';a.write_bytes(b'not an array')
                with self.assertRaises(ValueError):e.read(a)
    def test_phase_exact_manifest_and_extra_file(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td).resolve();folder=root/'prepared';folder.mkdir();p=folder/'A.json';p.write_text('{}');c=folder/'COMPLETION.json';c.write_text(json.dumps(dict(status='COMPLETE',plan_sha256='a'*64,artifacts_sha256={'A.json':r.digest(p)})))
            with patch.object(r,'ROOT',root):
                r.Evidence().phase(folder,r.digest(c),'a'*64,{'A.json'});(folder/'extra.json').write_text('{}')
                with self.assertRaises(ValueError):r.Evidence().phase(folder,r.digest(c),'a'*64,{'A.json'})
    def test_terminal_success_and_artifact_binding(self):
        hashes=dict(prepare='b'*64,select='c'*64,diagnose='d'*64);p=dict(plan_sha256='a'*64,phases={})
        for key,rel in [('prepare','prepared/COMPLETION.json'),('select','selected/FROZEN.json'),('diagnose','results/COMPLETION.json')]:p['phases'][key]=dict(session_id=1,exit_code=0,chunk_id='receipt',artifact_path=str(r.OUT/rel),artifact_sha256=hashes[key])
        r.terminals_check(p,'a'*64,hashes);p['phases']['diagnose']['exit_code']=1
        with self.assertRaises(ValueError):r.terminals_check(p,'a'*64,hashes)
    def test_canonical_source_aliases_not_basename(self):
        rel={'data/%02d.raw'%i:'a'*64 for i in range(64)};absolute={str(r.ROOT/k):v for k,v in rel.items()}
        self.assertEqual(r.canonical_source_map(rel),r.canonical_source_map(absolute));bad=dict(rel);bad['data/63.raw']='b'*64;self.assertNotEqual(r.canonical_source_map(rel),r.canonical_source_map(bad))
    def test_mutable_central_never_bound(self):
        with self.assertRaises(ValueError):r.local(r.ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')
    def test_runtime_requires_all_phase_hashes(self):
        class Args:
            plan_sha256='a'*64
        with patch.object(r,'REPORT',Path('/private/tmp/no260savedreaderreport')),patch.object(r.platform,'python_version',return_value='3.9.6'):
            with self.assertRaises((AttributeError,ValueError)):r.run(Args())


if __name__=='__main__':unittest.main(verbosity=2)
