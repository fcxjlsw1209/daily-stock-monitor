"""Past feature descriptions for agreement months; not observed training labels."""
import json
import numpy as np
from research import P, load, raw_features, dump


def main():
    px,fs,pools=load();raw=raw_features(px,fs);member=pools['core17']
    ranks={k:v.where(member).rank(axis=1,pct=True) for k,v in raw.items()}
    ambiguity=json.loads((P/'HOLDING_AMBIGUITY.json').read_text());rows=[]
    for row in ambiguity['rows']:
        if not row['same_complete_combination']:
            continue
        day=row['rebalance_date'];sc=raw['m252_42'].loc[day].where(member.loc[day]).dropna().sort_values(ascending=False)
        ordinal={s:i+1 for i,s in enumerate(sc.index)}
        for s in row['intersection']:
            features={k:float(ranks[k].at[day,s]) for k in ['risk','vol','r63','r21','relative','cmf21','near252']}
            features={k:v if np.isfinite(v) else None for k,v in features.items()}
            rows.append(dict(month=row['month'],signal_date=day,ticker=s,r12_2_rank=ordinal[s],
                             r12_2=float(raw['m252_42'].at[day,s]),feature_percentiles=features))
    result=dict(note='Agreement months were selected with the full future chart. Descriptive clues only, not verified holdings or independent labels. Original core17 ranks. Missing factors stay null.',
                months=sum(r['same_complete_combination'] for r in ambiguity['rows']),observations=len(rows),
                top3=sum(r['r12_2_rank']<=3 for r in rows),top5=sum(r['r12_2_rank']<=5 for r in rows),
                median_r12_2_rank=float(np.median([r['r12_2_rank'] for r in rows])),rows=rows)
    dump(P/'CLUE_FEATURES.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__':
    main()
