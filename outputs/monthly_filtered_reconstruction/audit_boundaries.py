from filtered_study import *

def main():
    plan,packs=prepare()
    sys.path.insert(0,str(OLD));from data_inputs import load_inputs
    sys.path.insert(0,str(O/'monthly_ledger_reconstruction'));from rank_audit import cap_panel,ordered
    px,raw,_,_,sources=load_inputs();caps,eligible,issuers=cap_panel(px,raw,sources)
    ranks={d:{display(s):j+1 for j,s in enumerate(ordered(pd.Timestamp(d),caps,eligible,issuers))} for d in packs['theme17__percentile']['dates']}
    inputs=['FITS.json','COMPACT_LEADERS.json','CORRECTION_LEADERS.json','RETENTION_LEADERS.json']
    if (P/'SPARSE_FITS.json').exists():inputs+=['SPARSE_FITS.json']
    fits=[r for f in inputs for r in read(P/f) if 'evaluations' in r]
    out=[]
    for sl in ['A','B']:
        for label,choices in [('compact',[r for r in fits if r['kind'] in ['compact_grid','compact_small_correction']]),('retention',[r for r in fits if r['kind']=='recursive_retention']),('flexible',[r for r in fits if r['kind']=='maximum_months' and r.get('max_features') is None]),('sparse',[r for r in fits if r.get('max_features')==8])]:
            choices=[r for r in choices if r['slice']==sl]
            if not choices:continue
            best=max(choices,key=lambda r:(r['evaluations'][sl]['exact_periods'],r['evaluations'][sl]['shared_positions']))
            details=[]
            for row in best['evaluations'][sl]['details']:
                q=ranks[row['date']];selected=row['selected'];target=row['target'];wrong=[s for s in selected if s not in target]
                details.append(dict(row,selected_cap_ranks={s:q.get(s) for s in selected},target_cap_ranks={s:q.get(s) for s in target},
                    new_boundary=[s for s in selected if q.get(s) is not None and 28<=q[s]<=32],wrong_boundary=[s for s in wrong if q.get(s) is not None and 28<=q[s]<=32],new_ARKK_SOXX=sorted(set(selected)&{'ARKK','SOXX'})))
            mismatch=[r for r in details if not r['exact']]
            clean=[r for r in details if not r['new_boundary']]
            out.append(dict(slice=sl,model=label,key=best['key'],kind=best['kind'],coefficients=best['coefficients'],weights=best.get('weights'),
                       evaluations={k:{kk:v for kk,v in z.items() if kk!='details'} for k,z in best['evaluations'].items()},
                       boundary_periods=sum(bool(r['new_boundary']) for r in details),mismatch_periods=len(mismatch),mismatch_with_boundary=sum(bool(r['new_boundary']) for r in mismatch),mismatch_with_wrong_boundary=sum(bool(r['wrong_boundary']) for r in mismatch),
                       clean_after_new_boundary_diagnostic=dict(periods=len(clean),exact=sum(r['exact'] for r in clean)),new_ARKK_SOXX_periods=sum(bool(r['new_ARKK_SOXX']) for r in details),details=details))
    dump('BOUNDARY_RESULTS.json',out);dump('CAP_AUDIT_SOURCES.json',sources)
    for r in out:print(r['slice'],r['model'],'periods',r['evaluations'][r['slice']]['periods'],'exact',r['evaluations'][r['slice']]['exact_periods'],'border',r['boundary_periods'],'mismatch',r['mismatch_periods'],'wrongborder',r['mismatch_with_wrong_boundary'],'clean',r['clean_after_new_boundary_diagnostic'],flush=True)
if __name__=='__main__':main()
