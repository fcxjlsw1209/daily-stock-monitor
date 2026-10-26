"""Bound-export reporting only: no model/strategy/source-price imports or fits."""
import argparse
from datetime import date,datetime,timezone
import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_relative_severity191'
ARMS=('incumbent_local67','relative_severity_ridge','relative_loss_ridge')
PATHS=('same_close_zero','next_open_zero','next_open_cost')
SLICES=('full122','first88','early53','recent35','later34')
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','LABELS.npz','LABEL_AUDIT.json',
    'MODEL_AUDIT.json','RANK_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json',
    'IDENTITY_AUDIT.json','CALENDAR.json','INDEPENDENT_PREPARE.json','CONTROL_PARITY.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'} | {
    arm+'_'+path+'_'+kind+'.json' for arm in ARMS for path in PATHS
    for kind in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
ACCOUNTING=ROOT/'work/ptr169_results_independent.py'

def require(ok,msg):
    if not ok:raise ValueError(msg)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def near(x,y,msg):require(math.isfinite(x) and math.isfinite(y) and math.isclose(x,y,abs_tol=2e-10,rel_tol=2e-10),msg)

def compare_metric(x,y,key,msg):
    if x is None or y is None:
        require(key=='sharpe' and x is None and y is None,msg)
    else:near(x,y,msg)

def format_metrics(value):
    sharpe='未定义' if value['sharpe'] is None else f"{value['sharpe']:.4f}"
    return f"{value['cagr']:.4%} / {sharpe} / {value['max_drawdown']:.4%}"

def measure(rows,initial):
    days=[r['date'] for r in rows];values=[float(r['equity']) for r in rows]
    require(len(rows)>1 and len(set(days))==len(days) and days==sorted(days),'Unique ordered daily NAV')
    require(math.isfinite(initial) and initial>0 and all(math.isfinite(v) and v>0 for v in values),'Positive finite NAV')
    years=(date.fromisoformat(days[-1])-date.fromisoformat(days[0])).days/365.2425
    rr=[b/a-1 for a,b in zip(values,values[1:])];sd=statistics.stdev(rr) if len(rr)>1 else 0.
    require(years>0,'Positive period span')
    peak=initial;dd=0.
    for v in values:peak=max(peak,v);dd=min(dd,v/peak-1.)
    return dict(cagr=(values[-1]/initial)**(1/years)-1,sharpe=statistics.mean(rr)/sd*math.sqrt(252) if sd>0 else None,max_drawdown=dd,years=years,wealth_multiple=values[-1]/initial,start=days[0],end=days[-1])

def monthly_factors(rows,dates,ends,initial=100000.):
    require(len(dates)==len(ends)>0,'Period axes')
    require(all(ends[i]==dates[i+1] for i in range(len(dates)-1)),'Contiguous monthly accounting periods')
    nav={r['date']:float(r['equity']) for r in rows};require(len(nav)==len(rows),'Duplicate daily date')
    require(all(d in nav and e in nav and d<e for d,e in zip(dates,ends)),'Missing monthly boundary')
    factors=[nav[e]/(initial if i==0 else nav[d]) for i,(d,e) in enumerate(zip(dates,ends))]
    require(all(math.isfinite(v) and v>0 for v in factors),'Invalid monthly wealth factor')
    near(math.prod(factors),nav[ends[-1]]/initial,'Monthly factors do not telescope')
    return factors

def relative_contributions(candidate,control,reference_cagr,years):
    require(len(candidate)==len(control)>0 and years>0,'Matched factors required')
    require(all(math.isfinite(x) and x>0 for x in candidate+control),'Positive factors required')
    logs=[math.log(a/b) for a,b in zip(candidate,control)];total=math.fsum(logs)
    change=(1+reference_cagr)*math.expm1(total/years)
    slope=change/total if total else (1+reference_cagr)/years
    contributions=[v*slope for v in logs];near(math.fsum(contributions),change,'Additive CAGR attribution')
    return dict(log_factors=logs,cagr_contributions_pp=[100*v for v in contributions],cagr_difference=change,wealth_ratio=math.exp(total))

def bind_manifest(folder,receipt,required=None):
    mapping=receipt['artifacts_sha256']
    require(isinstance(mapping,dict) and mapping,'Artifact manifest required')
    if required is not None:require(set(mapping)==required,'Unexpected artifact manifest')
    for name,digest in mapping.items():
        require(Path(name).name==name and sha(folder/name)==digest,'Changed artifact '+name)

def check_outcome_coverage(audit,result):
    for key in ('passed','independent_accounts','independent_of_production_model_and_engine'):
        require(audit.get(key) is True,'Missing independent evidence '+key)
    require(type(audit.get('checked_paths')) is int and audit['checked_paths']==9 and
            type(audit.get('checked_slices')) is int and audit['checked_slices']==45 and
            audit.get('terminal_liquidation') is False,'Independent9path45slice result required')
    require(audit.get('accounting_source_sha256')==sha(ACCOUNTING),'Independent accounting source changed')
    require(set(audit.get('paths',{}))==set(ARMS) and set(result.get('arms',{}))==set(ARMS),'Exact independent/result arms')
    for arm in ARMS:
        require(set(audit['paths'][arm])==set(PATHS) and set(result['arms'][arm])==set(PATHS),'Exact three execution paths')
        for path in PATHS:
            record=audit['paths'][arm][path]
            require(record.get('passed') is True and type(record.get('monthly_executions')) is int and
                    record['monthly_executions']==122 and set(record.get('slices',{}))==set(SLICES),
                    'Incomplete independent path '+arm+'/'+path)
            require(set(result['arms'][arm][path].get('slices',{}))==set(SLICES),'Incomplete result slices '+arm+'/'+path)

def check_control_parity(receipt):
    require(receipt.get('passed') is True and receipt.get('control_orders_exact') is True,'Control parity failed')
    require(receipt.get('arrays_dtype_shape_bytes_exact') is True,'Exact control byte evidence missing')
    require(receipt.get('candidate190_arrays_loaded') is False and receipt.get('candidate190_returns_read') is False,'Control-only audit scope changed')
    require(receipt.get('prior_plan_sha256')=='a3e57a03ca5a1d88251ea5e1a28ad1e15ad362ae55c3645fde356917b6ed0729','Wrong prior PLAN')
    require(receipt.get('prior_prepared_completion_sha256')=='544009a6583c44013680984fa6744839264ceacff6b086bfec12e248d5bd76a6','Wrong prior preparation')
    require(receipt.get('label_keys_checked')==['dates','ends','symbols','control_target','base_good','good'],'Incomplete control labels')
    require(receipt.get('forecast_keys_checked')==['control_prediction','control_rank','common_available'],'Incomplete control forecasts')


def build_report(plan_sha,completion_sha):
    require(sha(OUT/'PLAN.json')==plan_sha and sha(OUT/'COMPLETION.json')==completion_sha,'Explicit completion and PLAN binding')
    root=read(OUT/'COMPLETION.json');plan=read(OUT/'PLAN.json')
    require(root['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and root['plan_sha256']==plan_sha and root['observers_modified'] is False,'Formal completion required')
    require(root['protected_files']==21444,'Full protected archive audit required')
    result_receipt=read(OUT/'results/COMPLETION.json');prepared=read(OUT/'prepared/COMPLETION.json')
    require(sha(OUT/'results/COMPLETION.json')==root['result_completion_sha256'],'Result closure binding')
    require(result_receipt.get('paths')==9 and result_receipt.get('slices')==45,'Incomplete result completion coverage')
    for folder,receipt,required in [('results',result_receipt,RESULT_FILES),('prepared',prepared,PREPARED_FILES)]:
        require(receipt['status']=='COMPLETE' and receipt['plan_sha256']==plan_sha,'Incomplete phase')
        bind_manifest(OUT/folder,receipt,required)
    parity=read(OUT/'prepared/CONTROL_PARITY.json');check_control_parity(parity)
    frozen=read(OUT/'selected/FROZEN.json')
    require(frozen['plan_sha256']==plan_sha and result_receipt['frozen_sha256']==sha(OUT/'selected/FROZEN.json'),'Selection was frozen')
    for key,path in [('prepared_sha256','prepared/COMPLETION.json'),('orders_sha256','prepared/ORDERS.json'),('training_sha256','selected/TRAIN_RESULTS.json')]:
        require(frozen[key]==sha(OUT/path),'Frozen input changed '+path)
    result=read(OUT/'results/RESULTS.json');audit=read(OUT/'results/INDEPENDENT_OUTCOME.json')
    require(result['selection']==frozen['selection'],'Frozen selection unchanged')
    check_outcome_coverage(audit,result)
    meta=read(OUT/'prepared/META.json');orders=read(OUT/'prepared/ORDERS.json')
    require(len(meta['dates'])==122 and meta['dates'][0]=='2016-08-17' and meta['ends'][-1]=='2026-09-24','Exact requested period')
    require(set(orders)==set(ARMS) and set(result['arms'])==set(ARMS),'Exact arms')
    intervals={'full122':(0,121),'first88':(0,87),'early53':(0,52),'recent35':(53,87),'later34':(88,121)}
    metrics={};factors={};attribution={}
    for arm in ARMS:
        metrics[arm]={};factors[arm]={}
        for path in PATHS:
            daily=read(OUT/'results'/(arm+'_'+path+'_DAILY.json'));metrics[arm][path]={}
            for part,(lo,hi) in intervals.items():
                subset=[r for r in daily if meta['dates'][lo]<=r['date']<=meta['ends'][hi]]
                initial=100000. if part=='full122' else float(subset[0]['equity'])
                value=measure(subset,initial)
                for key in ('cagr','sharpe','max_drawdown'):compare_metric(value[key],result['arms'][arm][path]['slices'][part][key],key,'Export metric '+arm+'/'+path+'/'+part+'/'+key)
                metrics[arm][path][part]=value
            factors[arm][path]=monthly_factors(daily,meta['dates'],meta['ends'])
    for path in PATHS:
        attribution[path]={}
        for name,a,b in [('severity_vs_incumbent',ARMS[1],ARMS[0]),('relative_vs_incumbent',ARMS[2],ARMS[0]),('severity_vs_relative',ARMS[1],ARMS[2])]:
            q=relative_contributions(factors[a][path],factors[b][path],metrics[b][path]['full122']['cagr'],metrics[b][path]['full122']['years'])
            near(q['cagr_difference'],metrics[a][path]['full122']['cagr']-metrics[b][path]['full122']['cagr'],'CAGR attribution to exports')
            attribution[path][name]=q
    months=[]
    for i,d in enumerate(meta['dates']):
        held={a:orders[a][i]['symbols'] for a in ARMS}
        require(all(orders[a][i]['date']==d and orders[a][i]['end']==meta['ends'][i] for a in ARMS),'Order date axes')
        months.append(dict(index=i,date=d,end=meta['ends'][i],segment='early53' if i<53 else 'recent35' if i<88 else 'later34',holdings=held,
            changed_vs_incumbent={a:set(held[a])!=set(held[ARMS[0]]) for a in ARMS[1:]},
            contribution_pp={p:{k:v['cagr_contributions_pp'][i] for k,v in pairs.items()} for p,pairs in attribution.items()}))
    return dict(round=191,created_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=plan_sha,completion_sha256=completion_sha,
        selection=frozen['selection'],control_parity=parity,metrics=metrics,all122_months=months,attribution=attribution,
        model_audit=read(OUT/'prepared/MODEL_AUDIT.json'),label_audit=read(OUT/'prepared/LABEL_AUDIT.json'),
        original_arrays=27,protected_files=21444,independent_paths=9,independent_slices=45,
        limitation=plan['contract']['limitation'],new_models_or_returns_computed=False)

def markdown(report):
    labels=dict(zip(ARMS,['Local67','相对跑输幅度 Ridge','任何相对跑输 Ridge']))
    lines=['# 第191组：月度相对跑输幅度与发生率','',
      '候选通过训练筛选。' if report['selection']['admitted'] else '候选未通过训练筛选，保留 Local67。','',
      '目标：预测下个自然月 max(0,1−相对SPY财富比)，与是否跑输SPY的对照比较。两组同样本、原10项特征、年度前五年Ridge；修正幅度最多±0.005。这是无条件下行一阶部分矩，模型输出用于排名，并非校准概率或条件尾部预期损失。','',
      '全期2016-08-17至2026-09-24；数值为年化收益 / Sharpe / 最大回撤。','']
    for path,title in [('same_close_zero','同收盘零成本'),('next_open_cost','次开盘，每边0.05%成本')]:
        lines+=['## '+title,'','|区间|Local67|相对跑输幅度|任何相对跑输|','|---|---:|---:|---:|']
        for segment in SLICES:
            vals=[report['metrics'][a][path][segment] for a in ARMS]
            lines.append('|'+segment+'|'+'|'.join(format_metrics(v) for v in vals)+'|')
        lines.append('')
    lines+=['## 训练门槛','',json.dumps(report['selection'],ensure_ascii=False),'','## 持仓集合变化','',
        '|信号日|期末|Local67|相对跑输幅度|任何相对跑输|候选相对Local67全期CAGR贡献pp|','|---|---|---|---|---|---:|']
    for r in report['all122_months']:
        if any(r['changed_vs_incumbent'].values()):
            lines.append('|'+r['date']+'|'+r['end']+'|'+'|'.join(', '.join(r['holdings'][a]) or '现金' for a in ARMS)+f"|{r['contribution_pp']['same_close_zero']['severity_vs_incumbent']:+.5f}|")
    lines+=['','已独立核验9条执行路径、45个分段指标及21444份保护文件。后34个月仅作诊断；这是反复研究的历史样本，不是未见样本外验证。原动态股票池代理和历史行情版本限制仍在。所有冻结观察账户保持原样。','']
    return '\n'.join(lines)

def main(plan_sha,completion_sha):
    dest=OUT/'attribution';require(not dest.exists(),'Report output is write-once')
    report=build_report(plan_sha,completion_sha);dest.mkdir()
    (dest/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    (dest/'REPORT.md').write_text(markdown(report))
    receipt=dict(status='COMPLETE',plan_sha256=plan_sha,root_completion_sha256=completion_sha,source_sha256=sha(Path(__file__)),artifacts_sha256={p.name:sha(p) for p in dest.iterdir()},new_model_fits=0,new_returns_computed=False)
    (dest/'COMPLETION.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='REPORT_COMPLETE',path=str(dest/'REPORT.md'),sha256=sha(dest/'COMPLETION.json'))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);p.add_argument('--completion-sha',required=True);a=p.parse_args();main(a.plan_sha,a.completion_sha)
