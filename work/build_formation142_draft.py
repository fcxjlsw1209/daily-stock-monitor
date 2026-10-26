from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
p=json.loads((root/'work/monthly_formation142_proposal_draft.json').read_text())
p['status']='FINAL_PROSPECTIVE_SPECIFICATION_BEFORE_REGISTRATION'
p['round']=142
p['cache_contract']={'extra_arrays':{'formation_rank':'float64[122,12,49], unused rows NaN','formation_count':'int64[122],10 or12;0 if insufficient history','formation_ready':'bool[122]','formation_valid':'bool[122,49] original finite-core mask'},'scores':'For new active arms record original pre-episode single-stock scores; the choice is a joint objective, so these are not a standalone rank explaining selection. Fallback rows preserve incumbent final scores.','audit':'months122, each signal_index,date,calm,horizon,ready,reason,blocks,original_finite_mask,selectors'}
path=root/'work/monthly_formation142_proposal.json'
with path.open('x') as f:json.dump(p,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
proposal_sha=hashlib.sha256(path.read_bytes()).hexdigest()
s=(root/'work/monthly_price_delay140_runner.py').read_text().replace('price_delay140','formation142').replace('round140','round142').replace('round140','round142').replace("==140","==142").replace("== 140","== 142")
s=s.replace('Registered old-window market-price-delay experiment.','Registered shared formation-history portfolio experiment.')
s=s.replace('import monthly_formation142_factor as factor','import monthly_formation142_features as factor\nimport monthly_formation142_selector as selector')
s=s.replace("('incumbent_local67','contemporaneous_nonfit','lagged_price_delay')","('incumbent_local67','formation_common_delete_mean','formation_common_delete_worst')")
s=s.replace('EXTRA=factor.EXTRA',"EXTRA=('formation_rank','formation_count','formation_ready','formation_valid')")
s=s.replace('dafaf048f8139b3b1a486b37873115d5b551cb8741ce325b25aae9f87089319c',proposal_sha)
start=s.index('def choose(');end=s.index('\n\nclass Runner',start)
s=s[:start]+'''def build_features(panel, meta, old, sessions):
    prices=panel['adj'];axis=prices.index
    require(axis.is_unique and axis.is_monotonic_increasing and axis.tz is None,
            'Canonical unique historical price axis')
    calendar=pd.DatetimeIndex(sessions)
    require(calendar.is_unique and calendar.is_monotonic_increasing,'Official session order')
    n=len(meta['dates']);extra=dict(formation_rank=np.full((n,12,49),np.nan),
        formation_count=np.zeros(n,dtype=np.int64),formation_ready=np.zeros(n,dtype=bool),
        formation_valid=np.zeros((n,49),dtype=bool))
    rows=[]
    for i,(day,names) in enumerate(zip(meta['dates'],meta['symbols'])):
        t=axis.get_loc(pd.Timestamp(day));require(isinstance(t,(int,np.integer)) and t>=294,'Original core history')
        # Reconstruct the unperturbed original core from source prices rather
        # than inferring finite masks from rank zero (which is ambiguous).
        p=prices.loc[:day,names]
        require(p.index[-295:].equals(calendar[(calendar>=p.index[-295])&(calendar<=pd.Timestamp(day))]),'Missing official formation sessions')
        raw=p.pct_change(fill_method=None)
        sigma=raw.rolling(63).std().iloc[-1].replace(0,np.nan)
        spy=prices.loc[:day,'SPY'].pct_change(fill_method=None).rolling(63).std().iloc[-1]*np.sqrt(252)
        calm=bool(old['state_calm10'][i]);require(calm==bool(spy<.1),'Original calm state parity')
        cw=.5 if calm else .52
        expected_cw=np.asarray(old['core_weight'][i])
        require(np.all(expected_cw==cw),'Original core weight parity')
        numerator=p.iloc[-43]/p.iloc[-253 if calm else -295]-1
        rawcore=numerator if calm else numerator/sigma
        require(not np.isinf(rawcore.to_numpy()).any(),'Infinite original raw core')
        mask=np.isfinite(rawcore.to_numpy())
        rank=rawcore.rank(method='average',pct=True).fillna(.5).to_numpy()-.5
        require(np.allclose(cw*rank,old['core'][i],rtol=0,atol=1e-12),'Original weighted core parity')
        computed=factor.formation_scenarios(p.to_numpy(),len(p)-1,calm=calm)
        require(np.array_equal(computed['original_finite_mask'],mask),'Original raw finite mask parity')
        require(np.allclose(computed['core_ranks'],rank,rtol=0,atol=1e-12),'No-delete core rank parity')
        extra['formation_valid'][i]=mask
        extra['formation_count'][i]=len(computed['blocks'])
        extra['formation_ready'][i]=computed['ready']
        if computed['ready']:
            extra['formation_rank'][i,:len(computed['blocks'])]=computed['scenario_core_ranks']
        rows.append(dict(signal_index=i,date=day,calm=calm,horizon=computed['horizon'],
            ready=computed['ready'],reason=computed['reason'],blocks=computed['blocks'],
            original_finite_mask=mask.tolist(),selectors={}))
    return dict(arrays=extra,audit=dict(months=rows,original_months=n,original_width=49,
        new_model_fits=0,monthly_label_models_fit=0,returns_or_nav_computed=False,
        original_arrays_unchanged=True,raw_core_mask_and_rank_verified=True))


def choices_all(arrays,meta,n,backend,policy):
    require(type(n) is int and 0<=n<=len(meta['dates']),'Valid monthly prefix')
    original,original_scores=backend.baseline(arrays,meta,n,policy)
    pre=backend.pre(arrays,n)[:n]
    choices={NAMES[0]:original};scores={NAMES[0]:original_scores};audits={}
    for name,arm in zip(NAMES[1:],('mean_scenario','worst_scenario')):
        selected=[];saved=pre.copy();details=[]
        for i,names in enumerate(meta['symbols'][:n]):
            count=int(arrays['formation_count'][i]);ready=bool(arrays['formation_ready'][i])
            require(0<=count<=12,'Scenario count')
            scenarios=arrays['formation_rank'][i,:count] if ready else None
            cw=float(np.asarray(arrays['core_weight'][i]).reshape(-1)[0])
            item=selector.select_portfolio(pre[i],original[i],arrays['core'][i],cw,
                scenarios,arrays['eligible'][i],names,arm=arm,history_available=ready)
            selected.append(item['selected_indices']);details.append(item)
            if item['fallback']:saved[i]=original_scores[i]
        choices[name]=selected;scores[name]=saved;audits[name]=details
    return choices,scores,audits


def panels(a,m,backend,policy):
    unchanged={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()}
    choices,scores,audits=choices_all(a,m,MONTHS,backend,policy)
    orders={name:backend.make_orders(choices[name],scores[name],m) for name in NAMES}
    # Every possible signal-prefix must produce the same chosen orders/scores.
    for n in range(MONTHS+1):
        c,s,_=choices_all(a,m,n,backend,policy)
        for name in NAMES:
            require(c[name]==choices[name][:n] and s[name].tobytes()==scores[name][:n].tobytes(),'Prefix mismatch')
    missing=dict(a,formation_ready=np.zeros(MONTHS,dtype=bool))
    c,s,_=choices_all(missing,m,MONTHS,backend,policy)
    for name in NAMES[1:]:
        require(c[name]==choices[NAMES[0]] and s[name].tobytes()==scores[NAMES[0]].tobytes(),'Missing exact fallback')
    require(unchanged=={k:(v.dtype.str,v.shape,v.tobytes()) for k,v in a.items()},'Input mutation')
    return orders,scores,audits
''' + s[end:]
s=s.replace("result=factor.build_features(backend.panel,m,sessions['sessions'])","result=build_features(backend.panel,m,a,sessions['sessions'])")
s=s.replace("orders,scores=panels(a,m,backend,read(self.inputs['policy']))","orders,scores,audits=panels(a,m,backend,read(self.inputs['policy']))\n        for i,row in enumerate(result['audit']['months']):\n            row['selectors']={name:audits[name][i] for name in NAMES[1:]}")
start=s.index("        cells=read(self.out/'FEATURE_AUDIT.json')['cells']");end=s.index("        self.write('RESULTS.json'",start)
s=s[:start]+'''        months=read(self.out/'FEATURE_AUDIT.json')['months']
        require(len(months)==MONTHS and len(periods)==MONTHS,'All122 month rows required')
        for i,row in enumerate(periods):
            require(months[i]['signal_index']==i and months[i]['date']==row['date'],'Feature/month axes')
            row['formation_scenario']=months[i]
        comparisons={label:backend.old130.attribution(periods,results,new=new,old=old) for label,new,old in
            (('worst_vs_incumbent',NAMES[2],NAMES[0]),('mean_vs_incumbent',NAMES[1],NAMES[0]),('worst_vs_mean',NAMES[2],NAMES[1]))}
        by_date={row['date']:row['formation_scenario'] for row in periods}
        for comparison in comparisons.values():
            for change in comparison['changes']:change['formation_scenario']=by_date[change['date']]
''' + s[end:]
s=s.replace("paths=[Path(__file__),Path(factor.__file__),root/'work/test_monthly_formation142_factor.py',", "paths=[Path(__file__),Path(factor.__file__),Path(selector.__file__),\n        root/'work/test_monthly_formation142_features.py',root/'work/test_monthly_formation142_selector.py',")
with (root/'work/monthly_formation142_runner.py').open('x') as f:f.write(s)
print(json.dumps(dict(proposal_sha256=proposal_sha,runner_written=True)))
