"""Saved-evidence descriptive monthly attribution for closed round207.

No strategy imports, raw panel reads, factor/rank reconstruction, order generation,
portfolio replay, parameter search or causal effect estimation. The author also
implemented the207 host; this is a postmortem reader, not independent execution.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import unittest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/monthly_down_volume207'
OUTPUT=ROOT/'work/downvolume207_ATTRIBUTION.json'
ARMS=('incumbent_local67','down_volume_concentration','all_volume_control')
CANDIDATE=ARMS[1]
REFERENCES=(ARMS[0],ARMS[2])
PATH='same_close_zero'
PHASE_SHA={
 'PLAN.json':'da116e41fe703df42b689e23fba8b991cf3370cc4ebbbf02bffe6811f78a8627',
 'prepared/COMPLETION.json':'eb7e781c68a1528e5ca54afb7076dfd7ee632595d55f405b58717ab363e106a9',
 'results/COMPLETION.json':'0113e7d39b91e9297a26b440abecbbab649a094f9b413bbb92df07d955a4823b',
 'COMPLETION.json':'6a316a53d7870850a9a7dd7dfd3d55fc6de696033a6c6aa818d258364428c9b5'}
PREPARED_FILES={'FEATURES.npz','TRANSFORMS_DELTAS.npz','FEATURE_AUDIT.json','ORDERS.json','META.json','SOURCE_PARITY.json','INDEPENDENT_PREPARE.json'}
RESULT_FILES={'RESULTS.json','INDEPENDENT_OUTCOME.json'}|{
    a+'_'+p+'_'+k+'.json' for a in ARMS for p in ('same_close_zero','next_open_zero','next_open_cost')
    for k in ('DAILY','TRADES','HOLDINGS','DAILY_POSITIONS')}
LOG_RECONCILIATION_ATOL=1e-12


def require(ok,message):
    if not ok:raise ValueError(message)


def positive(value,label):
    require(type(value) in (int,float) and math.isfinite(value) and value>0,label)
    return value


class Evidence:
    def __init__(self):self.pins={}
    def load(self,relative,digest):
        path=(OUT/relative).resolve()
        require(path.is_relative_to(OUT.resolve()),'Nonlocal evidence')
        data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
        require(actual==digest,'Changed evidence: '+relative)
        key=str(path.relative_to(ROOT));require(key not in self.pins or self.pins[key]==actual,'Concurrent evidence change')
        self.pins[key]=actual
        return json.loads(data)
    def verify_unchanged(self):
        for name,digest in self.pins.items():
            require(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,'Evidence changed during attribution: '+name)


def validate_axes(meta,orders,daily):
    dates,ends=meta['dates'],meta['ends'];n=len(dates)
    require(n>0 and len(ends)==n and dates==sorted(set(dates)),'Invalid monthly axes')
    require(all(a<b for a,b in zip(dates,ends)) and ends[:-1]==dates[1:],'Monthly intervals must partition the full path')
    require(set(orders)==set(ARMS) and set(daily)==set(ARMS),'Three saved arms required')
    axes=None;nav={}
    for arm in ARMS:
        require(len(orders[arm])==n,'Order count differs')
        for i,row in enumerate(orders[arm]):
            require(row['date']==dates[i] and row['end']==ends[i] and row['signal_date']==dates[i],'Order clock differs')
            require(isinstance(row['symbols'],list) and len(row['symbols'])==len(set(row['symbols'])),'Order identity duplicates')
        rows=daily[arm];clock=[r['date'] for r in rows]
        require(clock==sorted(set(clock)) and clock[0]==dates[0] and clock[-1]==ends[-1],'Daily axis does not cover full monthly partition')
        require(axes is None or axes==clock,'Daily arm clocks differ');axes=clock
        nav[arm]={r['date']:positive(r['equity'],'Invalid saved equity') for r in rows}
        require(all(day in nav[arm] for day in dates+ends),'Missing exact boundary NAV; interpolation forbidden')
    return nav


def pair_months(meta,orders,nav,reference):
    rows=[]
    for i,(start,end) in enumerate(zip(meta['dates'],meta['ends'])):
        cg=nav[CANDIDATE][end]/nav[CANDIDATE][start]
        rg=nav[reference][end]/nav[reference][start]
        ratio=positive(cg/rg,'Invalid monthly growth ratio')
        old=orders[reference][i]['symbols'];new=orders[CANDIDATE][i]['symbols']
        rows.append(dict(index=i,date=start,end=end,era='first88' if i<88 else 'later34',
            candidate_growth=cg,reference_growth=rg,candidate_return=cg-1.,reference_return=rg-1.,
            growth_ratio=ratio,log_relative_growth=math.log(ratio),changed_holding_set=set(old)!=set(new),
            candidate_symbols=new,reference_symbols=old,added=sorted(set(new)-set(old)),removed=sorted(set(old)-set(new))))
    first,last=meta['dates'][0],meta['ends'][-1]
    final_ratio=(nav[CANDIDATE][last]/nav[CANDIDATE][first])/(nav[reference][last]/nav[reference][first])
    total=math.fsum(row['log_relative_growth'] for row in rows)
    residual=total-math.log(final_ratio)
    require(abs(residual)<=LOG_RECONCILIATION_ATOL,'Monthly log relative growth fails full-NAV reconciliation')
    require(abs(math.log(math.exp(total)/final_ratio))<=LOG_RECONCILIATION_ATOL,'Exponentiated monthly sum fails reconciliation')
    summary={}
    for name,lo,hi in (('full122',0,len(rows)),('first88',0,min(88,len(rows))),('later34',min(88,len(rows)),len(rows))):
        part=rows[lo:hi];s=math.fsum(r['log_relative_growth'] for r in part)
        summary[name]=dict(months=len(part),changed_holding_months=sum(r['changed_holding_set'] for r in part),
            log_relative_growth=s,growth_ratio=math.exp(s),
            negative_months=sum(r['log_relative_growth']<0 for r in part),
            positive_months=sum(r['log_relative_growth']>0 for r in part),
            exact_zero_months=sum(r['log_relative_growth']==0 for r in part),
            tiny_nonzero_months_abs_log_le_1e12=sum(0<abs(r['log_relative_growth'])<=1e-12 for r in part),
            top5_negative_indices=[r['index'] for r in sorted((r for r in part if r['log_relative_growth']<0),key=lambda r:(r['log_relative_growth'],r['index']))[:5]])
    return dict(reference=reference,months=rows,summary=summary,reconciliation=dict(passed=True,
        sum_monthly_log_ratios=total,log_full_final_normalized_nav_ratio=math.log(final_ratio),
        full_final_normalized_nav_ratio=final_ratio,exponentiated_log_sum=math.exp(total),
        absolute_log_residual=abs(residual),absolute_log_tolerance=LOG_RECONCILIATION_ATOL))


def saved_trade_summary(meta,orders,holdings,trades):
    bydate={d:[] for d in meta['dates']};require(len(holdings)==len(meta['dates']),'Holdings count differs')
    for i,row in enumerate(holdings):
        require(row['signal_date']==row['execution_date']==meta['dates'][i],'Same-close execution clock differs')
        require(type(row['fee']) in (int,float) and row['fee']==0.,'Nonzero fee in same-close-zero path')
        require(len(row['symbols'])==len(orders[i]['symbols']),'Holdings identity width differs')
    for trade in trades:
        require(trade['date'] in bydate,'Unexpected intraperiod or terminal trade')
        require(type(trade['cost']) in (int,float) and trade['cost']==0.,'Nonzero saved trade cost')
        require(trade['side'] in ('buy','sell'),'Unknown trade side')
        bydate[trade['date']].append(trade)
    counts={d:dict(fills=len(rows),buys=sum(r['side']=='buy' for r in rows),sells=sum(r['side']=='sell' for r in rows)) for d,rows in bydate.items()}
    return dict(total_fills=len(trades),buys=sum(x['buys'] for x in counts.values()),sells=sum(x['sells'] for x in counts.values()),by_date=counts,all_recorded_fees_and_costs_zero=True)


def run(completion_sha256):
    require(completion_sha256==PHASE_SHA['COMPLETION.json'],'Explicit confirmed root completion SHA required')
    require(not OUTPUT.exists(),'Attribution output is write-once')
    ev=Evidence();phase={name:ev.load(name,digest) for name,digest in PHASE_SHA.items()}
    plan=phase['PLAN.json'];root=phase['COMPLETION.json'];prepared=phase['prepared/COMPLETION.json'];results=phase['results/COMPLETION.json']
    require(root['status']=='COMPLETED_INDEPENDENTLY_VERIFIED' and root['plan_sha256']==PHASE_SHA['PLAN.json'] and root['result_completion_sha256']==PHASE_SHA['results/COMPLETION.json'],'Root closure differs')
    require(plan['contract']['arms']==list(ARMS),'Plan arms differ')
    for receipt,names in ((prepared,PREPARED_FILES),(results,RESULT_FILES)):
        require(receipt['status']=='COMPLETE' and receipt['plan_sha256']==PHASE_SHA['PLAN.json'] and set(receipt['artifacts_sha256'])==names,'Saved phase manifest differs')
    def load_saved(folder,name):return ev.load(folder+'/'+name,phase[folder+'/COMPLETION.json']['artifacts_sha256'][name])
    meta=load_saved('prepared','META.json');orders=load_saved('prepared','ORDERS.json');audit=load_saved('prepared','FEATURE_AUDIT.json')
    require(len(meta['dates'])==122 and len(audit['event_history'])==122,'Expected122 saved months')
    daily={a:load_saved('results',a+'_'+PATH+'_DAILY.json') for a in ARMS}
    holdings={a:load_saved('results',a+'_'+PATH+'_HOLDINGS.json') for a in ARMS}
    trades={a:load_saved('results',a+'_'+PATH+'_TRADES.json') for a in ARMS}
    nav=validate_axes(meta,orders,daily)
    fills={a:saved_trade_summary(meta,orders[a],holdings[a],trades[a]) for a in ARMS}
    pairs={ref:pair_months(meta,orders,nav,ref) for ref in REFERENCES}
    for reference,pair in pairs.items():
        pair['largest_negative_months']={}
        for era in ('first88','later34'):
            rows=[]
            for i in pair['summary'][era]['top5_negative_indices']:
                row=dict(pair['months'][i]);event=audit['event_history'][i]
                require(event['date']==meta['dates'][i],'Saved feature audit date differs')
                cells={c['symbol']:c for c in event['cells']}
                row['saved_feature_evidence']={s:{key:cells[s][key] for key in ('source_available','available','down_count','candidate','control','candidate_rank','control_rank')} for s in row['added']+row['removed']}
                row['saved_display_holdings']={a:holdings[a][i]['symbols'] for a in (CANDIDATE,reference)}
                row['saved_fill_counts']={a:fills[a]['by_date'][row['date']] for a in (CANDIDATE,reference)}
                rows.append(row)
            pair['largest_negative_months'][era]=rows
    receipt=dict(schema_version=1,passed=True,created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Descriptive saved-path monthly attribution, not factor causality or a fresh backtest.',
        path=PATH,comparisons=pairs,saved_trade_counts=fills,
        boundary_convention='NAV at signal close to next signal close; last period ends2026-09-24. Contiguous intervals telescope; no interpolation.',
        interpretation=['Largest negative months are selected retrospectively for explanation, never used to tune or reselect207.',
            'Holding-set changes are observed associations. This reader does not isolate a causal factor effect or per-security PnL.',
            'Raw negative/positive month counts include tiny floating-point differences; a separate <=1e-12 absolute-log count discloses them.',
            'Only same_close_zero is used. All saved fees/costs are verified0; no slippage estimate or alternate execution comparison.',
            'CV squared, down-count and ranks shown are copied from frozen FEATURE_AUDIT; no factors or ranks recomputed.'],
        fresh_features_computed=False,orders_generated=False,portfolio_replay_performed=False,
        strategy_modules_imported=False,raw_panel_loaded=False,parameters_optimized=False,
        production_or_central_modified=False,observer_accessed=False,
        author_scope='Same author as207host; saved-evidence postmortem only.',
        input_sha256=ev.pins,reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    ev.verify_unchanged()
    with OUTPUT.open('x') as stream:json.dump(receipt,stream,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    digest=hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    print(json.dumps(dict(passed=True,path=str(OUTPUT),sha256=digest,
        summaries={ref:value['summary'] for ref,value in pairs.items()},
        largest_negative={ref:{era:[dict(date=r['date'],ratio=r['growth_ratio'],added=r['added'],removed=r['removed']) for r in rows] for era,rows in value['largest_negative_months'].items()} for ref,value in pairs.items()}),indent=2))


class SyntheticTests(unittest.TestCase):
    def fixture(self):
        meta=dict(dates=['2020-01-01','2020-02-01'],ends=['2020-02-01','2020-03-01'])
        orders={a:[dict(date=d,end=e,signal_date=d,symbols=['A']) for d,e in zip(meta['dates'],meta['ends'])] for a in ARMS}
        daily={a:[dict(date=d,equity=v) for d,v in zip(meta['dates']+[meta['ends'][-1]],[100.,110.,121.])] for a in ARMS}
        daily[CANDIDATE][1]['equity']=120.;daily[CANDIDATE][2]['equity']=108.
        orders[CANDIDATE][1]['symbols']=['B']
        return meta,orders,daily
    def test_telescoping_and_changed_sets(self):
        m,o,d=self.fixture();p=pair_months(m,o,validate_axes(m,o,d),ARMS[0])
        self.assertAlmostEqual(p['reconciliation']['full_final_normalized_nav_ratio'],108/121)
        self.assertEqual(p['summary']['first88']['changed_holding_months'],1)
        self.assertEqual(p['summary']['first88']['top5_negative_indices'],[1])
    def test_gap_rejected(self):
        m,o,d=self.fixture();m['ends'][0]='2020-01-31'
        with self.assertRaises(ValueError):validate_axes(m,o,d)
    def test_duplicate_or_nonpositive_daily_rejected(self):
        for kind in ('duplicate','zero'):
            m,o,d=self.fixture()
            if kind=='duplicate':d[CANDIDATE].append(d[CANDIDATE][-1])
            else:d[CANDIDATE][1]['equity']=0.
            with self.assertRaises(ValueError):validate_axes(m,o,d)
    def test_no_interpolation(self):
        m,o,d=self.fixture()
        for a in ARMS:d[a].pop(1)
        with self.assertRaises(ValueError):validate_axes(m,o,d)
    def test_sameclose_fees_and_offclock_trades_rejected(self):
        m,o,d=self.fixture();holds=[dict(signal_date=x,execution_date=x,symbols=['A'],fee=0.) for x in m['dates']]
        trade=dict(date=m['dates'][0],cost=0.,side='buy')
        self.assertEqual(saved_trade_summary(m,o[ARMS[0]],holds,[trade])['total_fills'],1)
        trade['cost']=.01
        with self.assertRaises(ValueError):saved_trade_summary(m,o[ARMS[0]],holds,[trade])
        trade.update(cost=0.,date='2020-01-15')
        with self.assertRaises(ValueError):saved_trade_summary(m,o[ARMS[0]],holds,[trade])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--selftest',action='store_true');parser.add_argument('--completion-sha256');args=parser.parse_args()
    if args.selftest:
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
        require(unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful(),'Synthetic attribution checks failed')
    else:run(args.completion_sha256)
