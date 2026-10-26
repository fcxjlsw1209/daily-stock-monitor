"""Review closed213 saved JSON and hashes; no model imports or numerical replay."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,math
R=Path(__file__).resolve().parents[1];O=R/'outputs/monthly_asset_peer213';P=R/'outputs/monthly_monotone_path212'
ARMS=('incumbent_local67','same_asset_peer_transfer','all_asset_peer_control')
PATHS={'same_close_zero':('close',0.),'next_open_zero':('open',0.),'next_open_cost':('open',.0005)}
SLICES=('full122','first88','early53','recent35','later34')
def need(x,msg):
 if not x:raise ValueError(msg)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main(expected):
 need(sha(O/'COMPLETION.json')==expected,'Explicit final closure SHA required')
 need(not (O/'RESEARCH_REVIEW.json').exists() and not (O/'RESEARCH_REVIEW.md').exists(),'Review write-once')
 closure=read(O/'COMPLETION.json');plan=read(O/'PLAN.json');prepared=read(O/'prepared/COMPLETION.json');closed=read(O/'results/COMPLETION.json')
 need(closure['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and closure['protected_files']==21444 and closure['observers_modified'] is False,'Final closure scope')
 for receipt in (closure,prepared,closed):need(receipt['plan_sha256']==sha(O/'PLAN.json'),'PLAN lineage')
 need(closure['result_completion_sha256']==sha(O/'results/COMPLETION.json'),'Result closure lineage')
 need(prepared['status']==closed['status']=='COMPLETE' and set(prepared['artifacts_sha256'])=={'FORECASTS.npz','RANKS_DELTAS.npz','MODEL_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'},'Exact7 preparation artifacts')
 files={a+'_'+p+'_'+k+'.json' for a in ARMS for p in PATHS for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}|{'RESULTS.json','INDEPENDENT_OUTCOME.json'}
 need(set(closed['artifacts_sha256'])==files and closed['paths']==9 and closed['slices']==45,'Result38 manifest')
 inputs={}
 for sub,receipt in (('prepared',prepared),('results',closed)):
  for name,digest in receipt['artifacts_sha256'].items():
   p=O/sub/name;need(sha(p)==digest,'Saved artifact changed '+str(p));inputs[str(p.relative_to(R))]=digest
 prep=read(O/'prepared/INDEPENDENT_PREPARE.json');outcome=read(O/'results/INDEPENDENT_OUTCOME.json')
 need(prep['model']['passed'] and prep['slot']['passed'] and prep['prefixes']==123 and prep['model_prefixes']==[0,1,53,88,122] and prep['model_prefix_sources_truncated'],'Feature/delta/prefix evidence')
 need(prep['original_array_count']==27 and prep['original_arrays_unchanged'] and prep['original_orders_exact'] and prep['single_original_episode'],'Original parity')
 need(outcome['passed'] and outcome['checked_paths']==9 and outcome['checked_slices']==45 and outcome['independent_of_production_model_and_engine'],'Independent outcome evidence')
 counts=prep['model']['counts']
 need(counts==prep['model_counts']==prepared['model_counts']==closure['model_counts'],'Main model count receipt parity')
 need(counts['actual_regression_solves']==counts['annual_pooled_log_refits']==prep['model']['independent_pooled_fits'],'Pooled-fit accounting')
 need(counts['same_type_supported_queries']+counts['same_type_fallback_queries']==counts['all_peer_query_evaluations']==counts['own_weight_evaluations'],'Correction query accounting')
 diagnostics=prep['transfer_diagnostics'];source_parity=read(O/'prepared/SOURCE_PARITY.json')
 need(diagnostics==source_parity['transfer_diagnostics'] and diagnostics['descriptive_only'] and diagnostics['used_for_selection'] is False,'Descriptive diagnostics scope')
 for k in ('cached_source_local_vectors_exact','availability_exact','control_raw_exact','control_rank_exact','control_delta_exact','control_orders_exact'):need(prep['control206_parity'][k] is True,'Source206 control parity '+k)
 prior_done_path=P/'results/COMPLETION.json'
 need(plan['predecessor_binding']['input_sha256'][str(prior_done_path)]==sha(prior_done_path),'Registered predecessor result manifest')
 prior_done=read(prior_done_path);prior_path=P/'results/RESULTS.json'
 need(prior_done['status']=='COMPLETE' and prior_done['artifacts_sha256']['RESULTS.json']==sha(prior_path),'Predecessor metrics hash')
 inputs[str(prior_done_path.relative_to(R))]=sha(prior_done_path);inputs[str(prior_path.relative_to(R))]=sha(prior_path)
 frozen=read(O/'selected/FROZEN.json');result=read(O/'results/RESULTS.json');prior=read(prior_path);orders=read(O/'prepared/ORDERS.json')
 need(frozen['prepared_sha256']==sha(O/'prepared/COMPLETION.json'),'Prepared-to-selection chain')
 need(closed['frozen_sha256']==sha(O/'selected/FROZEN.json') and frozen['selection']==result['selection'],'Frozen first88 selection')
 need(frozen['training_sha256']==sha(O/'selected/TRAIN_RESULTS.json') and frozen['orders_sha256']==sha(O/'prepared/ORDERS.json'),'Training/order lineage')
 need(result['selection']['first88_only'] is True and result['selection']['later34_used'] is False,'Selection temporal scope')
 need(set(result['arms'])==set(ARMS),'Three arms')
 metrics={};changed={};rawcounts={}
 for a in ARMS:
  need(set(result['arms'][a])==set(PATHS),'Three execution paths')
  metrics[a]={}
  for path,(mode,cost) in PATHS.items():
   data=result['arms'][a][path];m=data['metrics'];need(m['execution']==mode and m['cost_per_side']==cost,'Timing and costs')
   need(m['start']=='2016-08-17' and m['end']=='2026-09-24' and set(data['slices'])==set(SLICES),'Full period and slices')
   need(data['accounting']['terminal_liquidation'] is False,'No terminal liquidation')
   if cost==0:
    need(m['fees_paid']==0,'Zero cost full fee')
    for row in read(O/'results'/f'{a}_{path}_TRADES.json'):need(row['cost']==0,'Zero cost trade fee')
    for row in read(O/'results'/f'{a}_{path}_HOLDINGS.json'):need(row['fee']==0,'Zero cost monthly fee')
   if a==ARMS[0]:
    for key in ('cagr','sharpe','max_drawdown','final_equity','total_return','trading_sessions'):
     need(m[key]==prior['arms'][a][path]['metrics'][key],'Incumbent metric parity '+key)
   metrics[a][path]={s:{k:data['slices'][s][k] for k in ('cagr','sharpe','max_drawdown','start','end')} for s in SLICES}
  if a!=ARMS[0]:changed[a]=[i for i in range(122) if set(orders[a][i]['symbols'])!=set(orders[ARMS[0]][i]['symbols'])]
 for name in ('work/monthly_asset_peer213_context.json','work/monthly_asset_peer213_TESTS.json','work/asset_peer213_CODE_REVIEW.json'):
  need(plan['input_sha256'][str(R/name)]==sha(R/name),'Registered review/test/context hash')
 tests=read(R/'work/monthly_asset_peer213_TESTS.json')
 need(tests['passed'] is True and tests['synthetic_only'] is True and tests['tests']>0 and tests['tests']==read(R/'work/asset_peer213_CODE_REVIEW.json')['synthetic_tests'] and tests['runtime']==plan['runtime'],'Formal synthetic test receipt')
 state=read(R/'outputs/monthly_target70_research/CONTINUATION_STATUS.json')
 need(len(state['completed'])==211 and sum(x['trials'] for x in state['completed'])==1150 and not state['registered_queue'],'Closed counters')
 need(state['completed'][-1]['directory']==O.name and state['completed'][-1]['completion']['sha256']==expected,'Central closure')
 for p in [O/'PLAN.json',O/'COMPLETION.json',O/'prepared/COMPLETION.json',O/'results/COMPLETION.json',O/'selected/FROZEN.json',O/'selected/TRAIN_RESULTS.json',R/'work/monthly_asset_peer213_context.json',R/'work/monthly_asset_peer213_TESTS.json',R/'work/asset_peer213_CODE_REVIEW.json',Path(__file__),P/'RESEARCH_REVIEW.json']:
  inputs[str(p.relative_to(R))]=sha(p)
 r=dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,scope='Root saved-JSON/hash/cost-label review only; independent mathematical and account verification are separate registered receipts. No NPZ opened, features fitted, or returns recomputed.',input_sha256=inputs,completion_sha256=expected,selection=result['selection'],metrics=metrics,changed_holdings_vs_incumbent=changed,model_counts=counts,transfer_diagnostics=diagnostics,synthetic_tests=tests['tests'],zero_cost_trade_and_month_fees_checked=True,incumbent_metrics_match_prior=True,protected_files_verified_by_closer=21444,protected_archive_reread_in_this_review=False,completed_entries=211,trials=1150,best=state['best_training_selected_candidate'],limitations=plan['contract']['limitation'],observers_modified=False)
 with (O/'RESEARCH_REVIEW.json').open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
 lines=['# 第213轮：同类资产历史残差参考','', '候选通过前88期筛选。' if r['selection']['admitted'] else '候选未通过前88期筛选，保留原Local67。','', '原始长期动量跳过最近42个交易日；只在原小幅模型修正中限制历史参考资产类型。月末EW3、月内固定单位不变。','']
 for path,label in [('same_close_zero','同收盘零成本'),('next_open_zero','次开盘零成本'),('next_open_cost','次开盘每边0.05%成本')]:
  lines+=['## '+label,'','|区间|Local67 CAGR / Sharpe / DD|同类资产参考候选|全部资产参考控制|','|---|---:|---:|---:|']
  for s in SLICES:
   cells=[]
   for a in ARMS:
    m=metrics[a][path][s];cells.append(f"{m['cagr']:.2%} / {m['sharpe']:.3f} / {m['max_drawdown']:.2%}")
   lines+=['|'+s+'|'+'|'.join(cells)+'|']
  lines+=['']
 lines+=['历史参考支持诊断：'+json.dumps(diagnostics['by_asset_type'],ensure_ascii=False),'', '未通过门槛：'+', '.join(k for k,v in r['selection']['guards'].items() if not v),'',f"已完成{tests['tests']}项合成测试、123个订单前缀、5个实际截断模型前缀、9条执行路径及45个区间核验；保护21444件原始档案。两条零成本路径逐笔和逐月费用均为零。",'', '本文件只复核已关闭结果及哈希，不重算收益。后34个月与全期均为重复使用的研究样本，不视为独立前向业绩；股票池及数据版本的历史代理限制仍然保留。']
 with (O/'RESEARCH_REVIEW.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print(json.dumps(dict(passed=True,review_sha256=sha(O/'RESEARCH_REVIEW.json'),report_sha256=sha(O/'RESEARCH_REVIEW.md'),selection=r['selection'],same_close={a:metrics[a]['same_close_zero']['full122'] for a in ARMS}),ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--completion-sha256',required=True);main(p.parse_args().completion_sha256)
