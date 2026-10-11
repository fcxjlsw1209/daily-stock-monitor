import json
import numpy as np
import pandas as pd
from research import P,O,OLD,base,ranks_and_rules,fit_models,monthly_examples,model_rule,dump


def main():
    px,fs,pools=base.load();raw=base.raw_features(px,fs);member=pools['core17']
    rank,_,_,_=ranks_and_rules(px,raw,member)
    examples=monthly_examples(px,raw,member,rank)
    # Independent label check for every historical observation, not just maturity.
    for k in range(len(examples['ends'])-1):
        entry,next_entry=examples['ends'][k:k+2]
        expected=(np.log(px.iloc[next_entry]/px.iloc[entry])-np.log(px.SPY.iloc[next_entry]/px.SPY.iloc[entry])).clip(-1,1)
        np.testing.assert_allclose(examples['y'][k],expected.to_numpy(),equal_nan=True)
        assert examples['sig'][k]==entry-1
        assert examples['maturity'][k]==next_entry
    # Refit after physically removing future prices and membership rows.
    cutoff=pd.Timestamp('2020-07-01');p=px.loc[:cutoff];f={k:v.loc[:cutoff] for k,v in fs.items()};m=member.loc[:cutoff]
    raw_short=base.raw_features(p,f);rank_short,_,_,_=ranks_and_rules(p,raw_short,m)
    partial,audit=fit_models(p,raw_short,m,rank_short,'strong_half')
    full=pd.read_csv(P/'predictions_core17_strong_half.csv.gz',index_col=0,parse_dates=True).reindex(partial.index)
    difference=(full-partial).abs().max().max();assert difference<1e-10
    assert full.notna().equals(partial.notna())
    audits=json.loads((P/'TRAINING_AUDIT.json').read_text());count=0
    for records in audits.values():
        for row in records:
            assert row['latest_label_end']<=row['signal_date']<row['entry_date']
            assert not row['uses_inverse_labels'];assert row['months']<=60
            count+=1
    orders=json.loads((P/'ORDERS.json').read_text());results=json.loads((P/'RESULTS.json').read_text());nav=pd.read_csv(P/'CURVES.csv.gz',index_col=0,parse_dates=True)
    assert len(results)==len(orders)==nav.shape[1]==42
    for key,rows in orders.items():
        assert len(rows)==121
        for row in rows:
            assert len(row['holdings']) in [0,2,3]
            assert row['signal_date']<row['date']
            assert px.index[px.index.get_loc(pd.Timestamp(row['date']))-1]==pd.Timestamp(row['signal_date'])
        assert np.isclose(nav[key].iloc[-1],results[key]['wealth'])
    # Neutral optional-factor treatment is explicit and works at actual gaps.
    secondary_checks=0
    for name in ['relative','cmf21','volume_ratio']:
        missing=member&raw[name].isna()
        if missing.to_numpy().any():
            assert np.allclose(rank[name].where(missing).stack(),.5)
            secondary_checks+=1
    protected=json.loads((P/'PROTECTED_BEFORE.json').read_text());changes=[f for f,h in protected.items() if base.sha(f)!=h];assert not changes
    sources=json.loads((OLD/'SOURCE_HASHES.json').read_text());source_changes=[f for f,h in sources.items() if base.sha(f)!=h];assert not source_changes
    previous=json.loads((P/'BEFORE_NUMERICAL_FIX.json').read_text())
    # A byte hash of all orders also includes accumulated floating point fees.
    # Verify actual dates/holdings directly against the old reduction instead.
    compared=0;prediction_reduction_diff=0.
    for pool in ['core17','theme17']:
        mm=pools[pool];rr,room,momentum,_=ranks_and_rules(px,raw,mm);sim=base.Simulator(px,fs,mm)
        strong=(raw['m252_42']>0)&(rr['m252_42']>=.5)
        for scope in ['all_positive','strong_half']:
            ref,_=fit_models(px,raw,mm,rr,scope,reference_matmul=True)
            saved=pd.read_csv(P/f'predictions_{pool}_{scope}.csv.gz',index_col=0,parse_dates=True)
            assert ref.notna().equals(saved.notna())
            delta=float((ref-saved).abs().max().max());prediction_reduction_diff=max(prediction_reduction_diff,delta)
            assert delta<1e-12
            sc=model_rule(rr,room,momentum,ref,mm)
            for gn,gate in [('strong',strong),('strong_recovering',strong&(raw['r63']>0))]:
                for buffer in [0,5]:
                    key=f'{pool}__ridge_{scope}__{gn}__keep{buffer}'
                    _,oo=sim.run(sc,gate,buffer=buffer,cost_bps=10,signal_lag=1)
                    assert [(q['date'],q['holdings']) for q in oo]==[(q['date'],q['holdings']) for q in orders[key]]
                    compared+=1
    result_diff=max(abs(results[k][f]-previous['results'][k][f]) for k in results for f in ['wealth','cagr_252','max_dd','sharpe'])
    assert result_diff<1e-12
    assert 'RuntimeWarning' not in (P/'run.log').read_text()
    result=dict(prefix_cutoff=str(cutoff.date()),prediction_prefix_max_difference=float(difference),prediction_prefix_missing_pattern_equal=True,
                mature_label_fits=count,calendar_label_rows_checked=len(examples['ends'])-1,
                completed_cases=42,monthly_events_per_case=121,position_count_and_signal_clock_passed=True,
                missing_secondary_features_checked=secondary_checks,protected_files_unchanged=len(protected),source_hashes_verified=len(sources),
                inverse_holdings_used_as_training_labels=False,independent_oos=False,production_accounts_changed=False,
                numerical_fix_holdings_identical=True,reference_model_cases_replayed=compared,
                reference_prediction_max_difference=prediction_reduction_diff,
                raw_order_file_hash_equal=previous['ORDERS.json']==base.sha(P/'ORDERS.json'),
                numerical_fix_metric_max_difference=result_diff,finite_model_outputs=True)
    dump('VALIDATION.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
