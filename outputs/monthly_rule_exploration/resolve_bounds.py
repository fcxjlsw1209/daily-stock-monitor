from joint_explore import *
from monotone_core import transformed,CORE_NAMES

def main():
    env=prepare();results=[]
    for label,kind,family in [('original','quadratic','raw_magnitude'),('monotone','threshold','universe_percentile')]:
        rows=[];cs=[]
        for start,end in JOINT_WINDOWS:
            audit=read(P/f'JOINT_{start}_AMBIGUITY.json')
            for a in audit:
                row=env['rows'][a['start']];rows.append(transformed(row) if label=='monotone' else row);cs.append(a['candidates'])
        r=mixed_fit(rows,cs,family,kind,seconds=120);r['core_hypothesis']=label
        if label=='monotone' and r.get('fit'):r['fit']['feature_names']=CORE_NAMES+list(r['fit']['feature_names'][6:])
        r['maximum_margin_upper_bound']=-r['mip_dual_bound'] if 'mip_dual_bound' in r else None
        r['strict_positive_margin_excluded_by_bound']=r['maximum_margin_upper_bound'] is not None and r['maximum_margin_upper_bound'] < -1e-6
        results.append(r);dump('BOUND_RESOLUTION.json',results)
        print('RESOLVE',label,r['status'],r.get('fit',{}).get('margin'),r['maximum_margin_upper_bound'],r['elapsed'],flush=True)
    for f,h in read(P/'PROTECTED_SOURCES.json').items():assert sha(f)==h,f
if __name__=='__main__':main()
