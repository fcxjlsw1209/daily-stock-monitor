"""Recompute winner scores, component effects, budget and conservation audits."""
from pathlib import Path
import itertools, json, sys, unittest
import numpy as np
import research as r


def main():
    data=r.load_eval();seeds=r.seeds(data);pred=np.load(r.P/'PREDICTIONS.npz')['ranks'];leaders=r.read(r.P/'LEADERS.json');best=leaders['A'];seed=seeds[best['trial']['seed']];ab=r.read(r.P/'ABLATIONS.json');rows=best['details']
    # Preserve source records and both predetermined masks; never further exclude months.
    changes=[]
    for old,new in zip(ab[0]['details'],rows):
        if set(old['selected'])!=set(new['selected']):changes.append(dict(date=new['date'],target=new['target'],base=old['selected'],new=new['selected'],before_shared=old['shared'],after_shared=new['shared'],before_exact=old['exact'],after_exact=new['exact'],included_A=new['included_A'],included_B=new['included_B']))
    strata={}
    for group in ['all','A','B']:
        kept=[x for x,yes in zip(rows,data['masks'][group]) if yes]
        strata[group]={state:dict(periods=sum(x['weak']==weak for x in kept),exact=sum(x['exact'] and x['weak']==weak for x in kept),shared=sum(x['shared'] for x in kept if x['weak']==weak)) for state,weak in [('weak',True),('strong',False)]}
    # Inspect near-neighbor weights without choosing a new winner or changing grid.
    nearby=[]
    for recent,shape in itertools.product([.045,.05,.055],repeat=2):
        w=[0,recent,0,shape];sc=r.compose(seed,*pred[best['bundle']['model_ids']],data['shapes'][:,:,best['trial']['shape']],data['weak'],w)
        nearby.append(dict(recent_weight=recent,shape_weight=shape,metrics={k:v.tolist() for k,v in r.metrics(data,sc).items()}))
    r.dump('DISAGREEMENT_AUDIT.json',dict(changes=changes,strata=strata,nearby_weight_sensitivity=nearby,
        remaining_A=[x for x in rows if x['included_A'] and not x['exact']],remaining_B=[x for x in rows if x['included_B'] and not x['exact']]))
    assert best['weights']==[0,.05,0,.05] and best['shape']=='chase_penalty'
    recipe=seed['rule']['recipe'];features=data['packs'][seed['rule']['key']]['meta']['feature_names'];wc=seed['core_weight'];g=seed['rule']['gamma'];factor=1-.1/(1-wc)
    core=features[recipe[0]];weights={core:wc,features[recipe[1]]:recipe[4]*(1-g/(1-wc))*factor,features[recipe[2]]:recipe[5]*(1-g/(1-wc))*factor,'high_market_vol__ma200_distance':g*factor,'recent_weighted_Ridge_rank':.05,'negative_rebound21_atr_rank':.025,'negative_positive_gap5_vol_rank':.025}
    assert abs(sum(weights.values())-1)<1e-10
    r.dump('BEST_RULE.json',dict(status='RETROSPECTIVE_RECONSTRUCTION_CANDIDATE_NOT_AUTHOR_REPLICATION',weights=weights,ridge=best['bundle'],unused=['ordinary_Ridge','weak_GBDT'],normalization='All base/shape factors: percentile among current49 -0.5. Ridge output: percentile among eligible candidates -0.5. Gates inherited unchanged.',decision='Recompute on monthend; choose highest3 eligible, equal dollars. Two eligible: equal half; fewer2: cash. No intra-month changes. Same-close scores are descriptive, not proven executable at same close.'))
    loader=unittest.TestLoader();tests=loader.discover(str(r.P),pattern='test_research.py');result=unittest.TextTestRunner(verbosity=2).run(tests)
    assert result.wasSuccessful()
    protected=r.read(r.P/'PROTECTED_BEFORE.json');sources=r.read(r.P/'SOURCE_HASHES.json')
    assert all(r.sha(f)==h for f,h in {**protected,**sources}.items())
    validation=dict(tests_run=result.testsRun,success=result.wasSuccessful(),protected_files_unchanged=len(protected),source_files_unchanged=len(sources),fit_count=len(r.read(r.P/'TRAINING_AUDIT.json')),all_labels_matured=all(x['train_end']<=x['cutoff'] for x in r.read(r.P/'TRAINING_AUDIT.json')),missing_feature_counts={'industry_relative60':55,'sector_corr63':58,'sector_link_stability':78},numerical=r.read(r.P/'NUMERICAL_VALIDATION.json'))
    r.dump('VALIDATION.json',validation)
    r.dump('CODE_HASHES.json',{str(f):r.sha(f) for f in r.P.glob('*.py')})
    r.dump('STATUS.json',dict(status='EXPERIMENT_COMPLETE_NOT_FULLY_REPLICATED',periods=69,primary=best['metrics'],trial_count=r.read(r.P/'SEARCH_STATUS.json')['trials'],validation_success=True,remaining_A=54-best['metrics']['A']['exact'],remaining_B=39-best['metrics']['B']['exact']))
    r.report()
    out=['','\u5177\u4f53\u7ed3\u8bba\uff1a','- \u8fd1\u671f\u52a0\u6743Ridge\uff0b\u8ffd\u6da8\u964d\u5206\u7684\u5c0f\u5e45\u7ec4\u5408\u5728A\u4e2d\u65b0\u4fee\u6b633\u671f\uff0c\u6ca1\u6709\u4e22\u5931\u539f17\u671f\u5b8c\u5168\u5339\u914d\uff1a']
    for x in changes:
        if x['included_A'] and x['after_exact'] and not x['before_exact']:out.append(f"  {x['date']}: {'/'.join(x['base'])} \u2192 {'/'.join(x['new'])}\uff1b\u4f5c\u8005 {'/'.join(x['target'])}")
    out+=['- \u5f53\u524d\u6700\u4f18\u6ca1\u6709\u542f\u7528\u666e\u901aRidge\u6216\u5f31\u5e02GBDT\u3002\u5f3a\u5236\u56db\u4e2a\u6a21\u5757\u5168\u542f\u7528\uff0c\u5728A\u4e2d\u6700\u9ad818/54\uff0c\u4e0d\u80fd\u56e0\u67b6\u6784\u66f4\u590d\u6742\u5c31\u58f0\u79f0\u66f4\u50cf\u539f\u4f5c\u8005\u3002','- \u5f53\u524d\u7b80\u6d01\u5019\u9009\u5728\u5168\u90e869\u671f\u547d\u4e2d138/207\u4e2a\u4f4d\u7f6e\uff0c24\u671f\u4e09\u53ea\u5168\u5bf9\uff1bA\u547d\u4e2d110/162\u4e2a\u4f4d\u7f6e\uff0c20\u671f\u5168\u5bf9\uff1bB\u547d\u4e2d84/117\u4e2a\u4f4d\u7f6e\uff0c18\u671f\u5168\u5bf9\u3002\u4f4d\u7f6e\u547d\u4e2d\u7387\u4e0d\u7b49\u4e8e\u6574\u671f\u5339\u914d\u7387\u3002','- A\u4ecd34\u671f\u4e0d\u5b8c\u5168\u4e00\u81f4\uff0cB\u4ecd21\u671f\u3002\u5df2\u77e5\u7684\u5e02\u573a\u72b6\u6001/\u91cf\u4ef7/ML\u8865\u5145\u4e0d\u8db3\u4ee5\u89e3\u91ca\u8fd9\u4e9b\u5dee\u5f02\u3002','', '\u53ef\u8bfb\u8bc4\u5206\u6743\u91cd\uff08\u5747\u4e3a\u4e2d\u5fc3\u5316\u767e\u5206\u4f4d\uff09\uff1a',json.dumps(weights,ensure_ascii=False,indent=2),'',f"\u9a8c\u8bc1\uff1a{result.testsRun}\u9879\u6d4b\u8bd5\u901a\u8fc7\uff1b{len(protected)}\u4e2a\u53d7\u4fdd\u62a4\u6587\u4ef6\u548c{len(sources)}\u4e2a\u6e90\u6587\u4ef6\u54c8\u5e0c\u672a\u53d8\u3002",'- \u4e3b\u673a\u539f\u77e9\u9635\u4e58\u6cd5\u51fa\u73b0\u6570\u503c\u8b66\u544a\uff1b\u5168\u90e8162\u4e2aRidge\u6539\u7528\u72ec\u7acb\u663e\u5f0f\u52a0\u6743\u6b63\u89c4\u65b9\u7a0b\u91cd\u7b97\uff0c\u9884\u6d4b\u6700\u5927\u5dee\u5c0f\u4e8e6e-13\uff0c\u6392\u5e8f\u5b8c\u5168\u4e00\u81f4\u3002\u521d\u6b21\u8f93\u51fa\u5df2\u5f52\u6863\uff0c\u6ca1\u6709\u9759\u9ed8\u5ffd\u7565\u8b66\u544a\u3002','- \u5c11\u6570\u5386\u53f2\u884c\u4e1a\u7279\u5f81\u7f3a\u5931\u6309\u4e2d\u6027\u5206\u5904\u7406\uff1b69\u671f\u7d2f\u8ba1\u7f3a\u5931\uff1a\u884c\u4e1a\u76f8\u5bf9\u5f3a\u5ea655\u4e2a\u5355\u5143\u3001\u884c\u4e1a\u76f8\u5173\u602758\u4e2a\u3001\u884c\u4e1a\u7a33\u5b9a\u602778\u4e2a\uff1b\u89c1MISSING_FEATURES.json\u3002','- BEST_RULE.json\u662f\u7814\u7a76\u5019\u9009\uff0c\u6ca1\u6709\u52a0\u5165\u6216\u4fee\u6539\u4efb\u4f55\u89c2\u5bdf\u8d26\u6237\u3002']
    with (r.P/'REPORT.txt').open('a') as f:f.write('\n'.join(out)+'\n')
    print('VALIDATED',validation['tests_run'],'tests; exact A/B/all',best['metrics']['A']['exact'],best['metrics']['B']['exact'],best['metrics']['all']['exact'])
    import performance
    performance.main()
    r.dump('CODE_HASHES.json',{str(f):r.sha(f) for f in r.P.glob('*.py')})

if __name__=='__main__':main()
