from reconstruct import *

def main():
    px,raw,members,periods,sources=load_inputs();f=feature_set(px,raw)
    check=[];state_full=market_states(px)
    for cut in ['2020-12-31','2021-08-31','2022-05-31','2023-06-30','2024-03-28','2025-05-30','2026-03-31','2026-08-31']:
        limited=feature_set(px.loc[:cut],{k:z.loc[:cut] for k,z in raw.items()})
        for name in FEATURES:
            a=f[name].loc[cut];b=limited[name].iloc[-1]
            assert np.allclose(a,b,equal_nan=True,atol=1e-12,rtol=1e-12),(cut,name)
            check.append(dict(cut=cut,feature=name,cells=len(a)))
        state_short=market_states(px.loc[:cut])
        for name in state_full:
            assert state_full.at[pd.Timestamp(cut),name]==state_short.iloc[-1][name]
            check.append(dict(cut=cut,feature='market_'+name,cells=1))
    annual=read(P/'ANNUAL_FROZEN_TRIALS.json');folds=[]
    for model in annual:
        for fold in model['years']:
            cutoff=min(r['date'] for r in fold['evaluation']['details'])
            assert fold['fit']['last_label_end']<=cutoff
            assert fold['fit']['last_label_start']<cutoff
            folds.append(dict(pool=model['pool'],family=model['family'],model=model['model'],year=fold['year'],cutoff=cutoff,last_label_end=fold['fit']['last_label_end']))
    for filename in ['SOURCE_HASHES.json','CAP_SOURCE_HASHES.json','DEFINITION_SOURCES.json']:
        for path,h in read(P/filename).items():assert sha(path)==h,path
    ml_audits=read(P/'MONTHLY_ML_TRAINING_AUDIT.json')
    for a in ml_audits: assert a['train_end']<=a['fit_cutoff']
    dump('VALIDATION.json',dict(monthly_return_model_training_cutoffs=len(ml_audits),prefix_checks=len(check),prefix_cells=sum(z['cells'] for z in check),year_frozen_training_cutoffs=len(folds),checks=check,folds=folds,source_hashes_unchanged=True,
                               note='No claim of recovering true author pool or strategy; invalid target-universe rows remain in match denominators.'))
    print('Validated',len(check),'feature prefixes;',len(folds),'annual training cutoffs; source hashes unchanged.')
if __name__=='__main__':main()
