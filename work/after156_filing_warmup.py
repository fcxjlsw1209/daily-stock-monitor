"""Read only original date/symbol axes; no price values or new factors."""
import argparse
from bisect import bisect_left
from collections import Counter
from datetime import datetime, timezone
import io
import json
import numpy as np
from after156_event_scope_local_review import Sources, require

PINS = {
    'outputs/monthly_local_linear70/META.json': '3b46968f486e88fa170ac01f120dd5ef74ebd5a56243c6991c5a4fea506076f4',
    'outputs/monthly_2016_extension/PRICES.npz': 'c877cbe443964d53972114bc654c473608362a6bb31577fb1de5e00cbd3c1485',
    'outputs/monthly_paired_target143/CALENDAR.json': '8ee4da69fba84db1edee37a7e0f5af5d3ab16c6257e5d37cddbd531f01404622',
}


def axis_check(symbol, signal, dates, symbols, calendar):
    if symbol not in symbols or 'SPY' not in symbols:
        return dict(axis_ready=False, reason='missing_symbol_or_SPY')
    t = bisect_left(calendar, signal)
    require(t < len(calendar) and calendar[t] == signal, 'Signal not exchange session')
    if t < 504:
        return dict(axis_ready=False, reason='calendar_left_truncated')
    needed = calendar[t-504:t-42+1]
    have = set(dates)
    missing = [x for x in needed if x not in have]
    return dict(axis_ready=not missing, reason='axis_ready_only' if not missing else 'missing_date_axis',
                required_price_start=needed[0], required_price_end=needed[-1],
                required_prices=len(needed), missing_dates=missing,
                positive_finite_prices_checked=False)


def self_test():
    from datetime import date, timedelta
    dates=[(date(2000,1,1)+timedelta(days=i)).isoformat() for i in range(700)]
    a=axis_check('A',dates[600],dates,{'A','SPY'},dates)
    require(a['axis_ready'] and a['required_prices']==463 and a['required_price_start']==dates[96]
            and a['required_price_end']==dates[558], 'Inclusive window bounds')
    require(not axis_check('A',dates[600],dates,{'A'},dates)['axis_ready'], 'Missing benchmark')
    require(not axis_check('B',dates[600],dates,{'A','SPY'},dates)['axis_ready'], 'Missing issuer')
    require(not axis_check('A',dates[600],dates[97:],{'A','SPY'},dates)['axis_ready'], 'Left warmup')
    require(not axis_check('A',dates[100],dates,{'A','SPY'},dates)['axis_ready'], 'Left calendar')
    require(axis_check('A',dates[600],dates[:559],{'A','SPY'},dates)['axis_ready'], 'No post-skip data consumed')
    return 6


def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    checks=self_test();src=Sources()
    meta=src.read('outputs/monthly_local_linear70/META.json',PINS['outputs/monthly_local_linear70/META.json'])
    cal=src.read('outputs/monthly_paired_target143/CALENDAR.json',PINS['outputs/monthly_paired_target143/CALENDAR.json'])['sessions']
    raw=src.read_bytes('outputs/monthly_2016_extension/PRICES.npz',PINS['outputs/monthly_2016_extension/PRICES.npz'])
    with np.load(io.BytesIO(raw), allow_pickle=False) as panel:
        # NPZ lazy access: only these two axis arrays are decompressed/interpreted.
        require(set(panel.files)=={'dates','symbols','close','opening'},'Original NPZ schema')
        dates=panel['dates'].astype(str).tolist();symbols=panel['symbols'].astype(str).tolist()
    require(dates==sorted(set(dates)) and len(symbols)==len(set(symbols)), 'Unique axes')
    require(cal==sorted(set(cal)) and set(dates)<=set(cal), 'Exchange calendar alignment')
    require(len(meta['dates'])==122 and all(len(s)==49 for s in meta['symbols']), 'Original grid')
    rows=[]
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        for j,s in enumerate(names):
            if s in meta['etfs']: continue
            rows.append(dict(signal_index=i,candidate_index=j,symbol=s,signal_date=day,
                             **axis_check(s,day,dates,symbols,cal)))
    src.read_bytes('work/after156_filing_warmup.py');src.read_bytes('work/after156_event_scope_local_review.py')
    src.recheck()
    summary={}
    for key,low,hi in [('early53',0,53),('recent35',53,88),('later34',88,122),('all122',0,122)]:
        group=[r for r in rows if low<=r['signal_index']<hi]
        summary[key]=dict(slots=len(group),axis_ready=sum(r['axis_ready'] for r in group),
                          reasons=dict(Counter(r['reason'] for r in group)))
    result=dict(status='AXIS_WARMUP_ONLY_NO_PRICE_OR_SOURCE_ADMISSION',created_utc=datetime.now(timezone.utc).isoformat(),
        panel_start=dates[0],panel_end=dates[-1],summary=summary,rows=rows,synthetic_checks=checks,
        caveat='Axis presence does not prove any individual stock has positive finite observations throughout its window.',
        price_array_values_read=0,factor_calculations=0,returns=0,new_trials=0,input_hashes=src.hashes)
    if args.write:
        with src.path('work/after156_filing_warmup.json').open('x') as f:
            json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ['status','panel_start','panel_end','summary','synthetic_checks']}))


if __name__=='__main__': main()
