"""Read closed219 saved JSON and hashes; no NPZ decoding, fitting or replay."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_risk_consensus219'
PRIOR=ROOT/'outputs/monthly_leader_gate218'
CENTRAL=ROOT/'outputs/monthly_target70_research/CONTINUATION_STATUS.json'
ARMS=('incumbent_local67','risk_consensus_min','risk_consensus_mean_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
BOUNDS={'full122':(0,122),'first88':(0,88),'early53':(0,53),'recent35':(53,88),'later34':(88,122)}
CONTEXT=ROOT/'work/monthly_risk_consensus219_context.json'
CONTEXT_SHA='ea33e164f6a99bf438a94369d28ef356d67f14a47164ca2dddea841beee979b6'
PREP={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}

def need(ok,msg):
    if not ok: raise ValueError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(p):
    def bad(x):raise ValueError('Nonfinite JSON '+x)
    return json.loads(Path(p).read_text(),parse_constant=bad)

def close_num(a,b):
    return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-7+2e-11*abs(b)

def guard(a,b,gain):
    return a['training']['cagr']>=b['training']['cagr']+gain-1e-12 and a['training']['sharpe']>=b['training']['sharpe']-.05 and a['training']['max_drawdown']>=b['training']['max_drawdown']-.03 and all(a[k]['cagr']>=b[k]['cagr']-.025 for k in ('early','recent'))

def main(expected,test_count):
    need(sha(OUT/'COMPLETION.json')==expected,'Explicit closure SHA differs')
    need(not (OUT/'RESEARCH_REVIEW.json').exists() and not (OUT/'RESEARCH_REVIEW.md').exists(),'Write-once review')
    pins={}
    def pin(path,digest=None):
        path=Path(path);path=(path if path.is_absolute() else ROOT/path).resolve()
        need('CloudStorage' not in path.parts,'Cloud source forbidden')
        need(path.is_relative_to(ROOT) or str(path).startswith('/Library/Developer/CommandLineTools/'),'External source')
        value=sha(path);need(digest is None or value==digest,'Hash differs: '+str(path))
        need(str(path) not in pins or pins[str(path)]==value,'Conflicting hash')
        pins[str(path)]=value;return value
    plan=read(OUT/'PLAN.json'); plan_sha=pin(OUT/'PLAN.json')
    closure=read(OUT/'COMPLETION.json')
    need(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['plan_sha256']==plan_sha and closure['protected_files']==21444 and closure['observers_modified'] is False,'Incomplete closure')
    need(pin(CONTEXT)==CONTEXT_SHA==plan['contract']['prospective_context_sha256'],'Context SHA')
    for path,digest in plan['input_sha256'].items():pin(path,digest)
    phases={}
    for folder,names in [('prepared',PREP),('results',RESULT)]:
        c=read(OUT/folder/'COMPLETION.json');pin(OUT/folder/'COMPLETION.json')
        need(c['status']=='COMPLETE' and c['plan_sha256']==plan_sha and set(c['artifacts_sha256'])==names,'Incomplete phase '+folder)
        for path,digest in c['artifacts_sha256'].items():pin(OUT/folder/path,digest)
        phases[folder]=c
    need(closure['result_completion_sha256']==sha(OUT/'results/COMPLETION.json') and phases['results']['paths']==9 and phases['results']['slices']==45,'Outcome lineage')
    tests=read(ROOT/'work/monthly_risk_consensus219_TESTS.json')
    need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']==test_count and test_count>0 and tests['runtime']==plan['runtime'],'Formal synthetic receipt')
    for path,digest in tests['code_sha256'].items():
        need(plan['input_sha256'].get(path)==digest,'Tested file not registered');pin(path,digest)
    prepared=read(OUT/'prepared/INDEPENDENT_PREPARE.json');feature=read(OUT/'prepared/FEATURE_AUDIT.json')
    need(prepared['original_orders_exact'] and prepared['original_arrays_unchanged'] and prepared['original_array_count']==27 and prepared['single_original_episode'],'Original strategy preservation')
    need(prepared['prefixes']==123 and prepared['feature_prefixes']==[0,1,53,88,122] and prepared['feature_prefix_sources_truncated'] and prepared['feature_prefix_metadata_truncated'] and prepared['feature_prefix_independently_verified'],'Prefix coverage')
    need(prepared['source_admission_metadata_only'] and prepared['source_support_exact'],'Source admission scope')
    need(all(prepared['feature'][k] is True for k in ('passed','independent_scalar_rebuild','same_window_return_risk_checked','skip42_checked','arithmetic_mean_checked','sample_ddof1_checked','zero_volatility_common_exclusion_checked','common_rank_domain_checked','original_pct_ranks_checked','matched_aggregation_checked','full253_price_support_checked','official_session_windows_checked','monthend_queries_checked','exact_common_availability','audit_checked','no_future_or_outcome_inputs','descriptive_counts_checked','no_predictive_fits_checked')),'Independent feature scope')
    need(all(prepared['delta'][k] is True for k in ('passed','half_core_formula_checked','original_pct_ranks_preserved','first_partial_unchanged','calm_rows_unchanged','missing_preserves_core','core_weights_checked','no_second_rank','independent_delta_rebuild')),'Independent delta scope')
    need(prepared['feature']['numerical_tolerance']==0.,'Exact feature tolerance')
    source=read(OUT/'prepared/SOURCE_PARITY.json')
    need(source['raw_source_hashes']==plan['raw_source_hashes'] and source['price_source_binding']==plan['price_source_binding'],'Source parity binding')
    binding=plan['price_source_binding'];source_review=read(binding['path'])
    need(sha(binding['path'])==binding['sha256'] and source_review['context_sha256']==CONTEXT_SHA and source_review['npz_decoded'] is False and source_review['actual_source_coverage_recomputed'] is False,'Source review scope')
    for mapping in ('source_evidence_sha256','context_evidence_sha256'):
        for path,digest in binding[mapping].items():pin(path,digest)
    need(phases['prepared']['new_returns_computed'] is False and prepared['new_model_fits']==0,'No preparation return/fits')
    source_count=sum(feature['source_available_by_event']);valid_count=sum(feature['available_by_event'])
    need(len(feature['event_history'])==feature['events']==122 and feature['checked_cells']==5978,'Feature scope')
    for event in feature['event_history']:
        cells=event['cells'];need(len(cells)==49,'Feature cell count')
        need(sum(c['source_available'] for c in cells)==event['source_available_count'] and sum(c['available'] for c in cells)==event['available_count'],'Saved support count')
        need(all(not c['available'] or c['source_available'] for c in cells),'Available outside source')
    counts={'valid_price_observations':253*source_count,'simple_returns':252*source_count,'ranked_observations':2*valid_count,'aggregated_scores':2*valid_count}
    for k,v in counts.items():
        need(feature[k]==prepared['feature'][k]==prepared[k]==phases['prepared'][k]==v,'Count lineage '+k)
    meta=read(OUT/'prepared/META.json');orders=read(OUT/'prepared/ORDERS.json')
    need(orders[ARMS[0]]==read(ROOT/'outputs/monthly_local_linear70/local_linear_ORDERS.json'),'Incumbent orders')
    need(set(orders)==set(ARMS) and all(len(o)==122 for o in orders.values()),'Three order streams')
    for rows in orders.values():
        for i,row in enumerate(rows):
            need(row['date']==row['signal_date']==meta['dates'][i] and row['end']==meta['ends'][i],'Order period')
            need(len(row['symbols']) in (0,2,3) and set(row['symbols'])<=set(meta['symbols'][i]),'Order holdings')
    result=read(OUT/'results/RESULTS.json');ind=read(OUT/'results/INDEPENDENT_OUTCOME.json')
    frozen=read(OUT/'selected/FROZEN.json');training=read(OUT/'selected/TRAIN_RESULTS.json')
    need(frozen['plan_sha256']==plan_sha and frozen['selection']==result['selection'] and frozen['training_sha256']==sha(OUT/'selected/TRAIN_RESULTS.json') and frozen['orders_sha256']==sha(OUT/'prepared/ORDERS.json') and frozen['prepared_sha256']==sha(OUT/'prepared/COMPLETION.json'),'Frozen selection')
    need(phases['results']['frozen_sha256']==sha(OUT/'selected/FROZEN.json'),'Frozen result lineage')
    need(result['strict_pit'] is False and result['reused_sample'] is True and result['observer_promotion'] is False,'Limitations')
    need(ind['passed'] and ind['independent_accounts'] and ind['independent_of_production_model_and_engine'] and ind['checked_paths']==9 and ind['checked_slices']==45 and ind['terminal_liquidation'] is False,'Independent outcome scope')
    need(ind['accounting_source_sha256']==sha(ROOT/'work/ptr169_results_independent.py'),'Accounting source')
    prior_done=read(PRIOR/'results/COMPLETION.json');pin(PRIOR/'results/RESULTS.json',prior_done['artifacts_sha256']['RESULTS.json']);prior=read(PRIOR/'results/RESULTS.json')
    metrics={};fee_evidence={};incumbent_accounting_differences={}
    for arm in ARMS:
        metrics[arm]={};fee_evidence[arm]={}
        for path,(mode,cost) in PATHS.items():
            data=result['arms'][arm][path];m=data['metrics'];proof=ind['paths'][arm][path]
            need(m['start']=='2016-08-17' and m['end']=='2026-09-24' and m['execution']==mode and m['cost_per_side']==cost,'Execution metric identity')
            need(proof['passed'] and proof['monthly_executions']==122 and set(proof['slices'])==set(data['slices'])==set(BOUNDS),'Independent slices')
            metrics[arm][path]={}
            for label,(lo,hi) in BOUNDS.items():
                s=data['slices'][label];need(s['start']==meta['dates'][lo] and s['end']==meta['ends'][hi-1],'Slice dates')
                need(all(close_num(s[k],proof['slices'][label][k]) for k in ('cagr','sharpe','max_drawdown')),'Independent metric values')
                metrics[arm][path][label]={k:s[k] for k in ('start','end','cagr','sharpe','max_drawdown')}
            if path in training[arm]:
                for part,label in [('training','first88'),('early','early53'),('recent','recent35')]:
                    need(all(close_num(training[arm][path][part][k],data['slices'][label][k]) for k in ('cagr','sharpe','max_drawdown')),'First88 metric parity')
            if arm==ARMS[0]:
                for k in ('cagr','sharpe','max_drawdown','final_equity','total_return','trading_sessions','trade_rows'):
                    need(m[k]==prior['arms'][arm][path]['metrics'][k],'Original metric differs '+k)
                reference=prior['arms'][arm][path]['metrics']
                need(all(close_num(m[k],reference[k]) for k in ('fees_paid','dollar_turnover')),'Original accumulated accounting differs')
                incumbent_accounting_differences[path]={k:m[k]-reference[k] for k in ('fees_paid','dollar_turnover')}
            trades=read(OUT/'results'/f'{arm}_{path}_TRADES.json');holdings=read(OUT/'results'/f'{arm}_{path}_HOLDINGS.json')
            need(len(holdings)==122,'Holdings coverage')
            if cost==0.:need(m['fees_paid']==0. and all(x['cost']==0. for x in trades) and all(x['fee']==0. for x in holdings),'Nonzero fee in zero-cost path')
            fee_evidence[arm][path]={'cost_per_side':cost,'fees_paid':m['fees_paid'],'zero_fees_checked':cost==0.}
    changes={}
    for a,b in [(ARMS[1],ARMS[0]),(ARMS[2],ARMS[0]),(ARMS[1],ARMS[2])]:
        ids=[i for i in range(122) if set(orders[a][i]['symbols'])!=set(orders[b][i]['symbols'])]
        changes[a+'_vs_'+b]={'indices':ids,'counts':{s:sum(lo<=i<hi for i in ids) for s,(lo,hi) in BOUNDS.items()}}
    ids=[i for i in changes[ARMS[1]+'_vs_'+ARMS[0]]['indices'] if i<88];ids2=[i for i in changes[ARMS[1]+'_vs_'+ARMS[2]]['indices'] if i<88]
    def support(ii):return len(ii)>=3 and any(i<53 for i in ii) and any(i>=53 for i in ii)
    c,b,control=(training[a] for a in (ARMS[1],ARMS[0],ARMS[2]))
    gates=dict(close_vs_incumbent=guard(c['same_close_zero'],b['same_close_zero'],.005),open_vs_incumbent=guard(c['next_open_cost'],b['next_open_cost'],0),close_vs_mean_control=guard(c['same_close_zero'],control['same_close_zero'],0),open_vs_mean_control=guard(c['next_open_cost'],control['next_open_cost'],0),changed_support=support(ids),changed_support_vs_control=support(ids2))
    selection=dict(primary=ARMS[1] if all(gates.values()) else ARMS[0],admitted=all(gates.values()),guards=gates,changed_months=ids,changed_months_vs_control=ids2,first88_only=True,later34_used=False)
    need(selection==frozen['selection'],'Saved first88 gates differ')
    state=read(CENTRAL)
    need(len(state['completed'])==217 and sum(x['trials'] for x in state['completed'])==1168 and not state['registered_queue'],'Central counters')
    need(state['completed'][-1]['directory']==OUT.name and state['completed'][-1]['completion']['sha256']==expected and state['best_training_selected_candidate']==closure['best_training_selected_candidate'],'Central closure identity')
    for p in [OUT/'COMPLETION.json',OUT/'selected/FROZEN.json',OUT/'selected/TRAIN_RESULTS.json',Path(__file__)]:pin(p)
    report=dict(passed=True,created_utc=datetime.now(timezone.utc).isoformat(),scope='Saved JSON/hash/counter/fee audit only; no NPZ decoded or returns recalculated.',completion_sha256=expected,plan_sha256=plan_sha,input_sha256=pins,formal_synthetic_tests=test_count,source_available_cells=source_count,available_cells=valid_count,estimate_counts=counts,selection=selection,changed_holdings=changes,metrics=metrics,fee_evidence=fee_evidence,incumbent_accounting_differences=incumbent_accounting_differences,independent_paths=9,independent_slices=45,order_prefixes=123,source_prefixes=[0,1,53,88,122],protected_files_verified_by_closer=21444,strict_pit=False,reused_sample=True,observers_modified=False,best=state['best_training_selected_candidate'],completed_entries=217,trials=1168)
    lines=['# 第219轮：累计动量与风险效率共同领先','', '候选通过前88期筛选。' if selection['admitted'] else '候选未通过前88期筛选，保留原Local67。','', '同一252个收益日，跳过最近42个交易日；分别排名累计收益与日收益均值/样本波动。候选取两个排名的较低值，对照取平均值。仅非平静市况替换一半原核心；初始部分月份和calm状态保留原样。原股票池、月末三只等权、原修正及月内固定持仓保留。','', '|执行口径|原Local67年化 / Sharpe / 回撤|共同领先候选|平均排名对照|','|---|---:|---:|---:|']
    def fmt(m):return f"{m['cagr']:.2%} / {m['sharpe']:.3f} / {m['max_drawdown']:.2%}"
    for p,label in [('same_close_zero','同收盘零成本'),('next_open_zero','次开盘零成本'),('next_open_cost','次开盘每边0.05%')]:lines.append('|'+label+'|'+'|'.join(fmt(metrics[a][p]['full122']) for a in ARMS)+'|')
    lines+=['','|零成本同收盘区间|原Local67|共同领先候选|平均排名对照|','|---|---:|---:|---:|']
    for label in BOUNDS:lines.append('|'+label+'|'+'|'.join(fmt(metrics[a]['same_close_zero'][label]) for a in ARMS)+'|')
    lines+=['',f'通过{test_count}项合成测试、独立逐证券特征核对、123个订单前缀、5个物理来源截断检查、9条执行路径与45个区间核算。已核对两条零成本路径的逐笔、逐月费用为零。','', '股票池及数据版本仍有历史代理限制；后34期和全期均已反复用于研究，不是独立前向样本。此报告只复核已保存结果，没有重算收益或修改观察账户。','']
    with (OUT/'RESEARCH_REVIEW.json').open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    with (OUT/'RESEARCH_REVIEW.md').open('x') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'RESEARCH_REVIEW.json'),report_sha256=sha(OUT/'RESEARCH_REVIEW.md'),selection=selection,same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--completion-sha256',required=True);p.add_argument('--tests',type=int,required=True);a=p.parse_args();main(a.completion_sha256,a.tests)
