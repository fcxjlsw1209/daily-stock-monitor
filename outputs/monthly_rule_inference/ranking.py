"""Listwise choice likelihood, with the same prespecified factor budgets."""
import numpy as np
import pandas as pd
from infer import P, O, read, dump, sha, load, CORE, SIDE, ALPHA, simplex, l1ball, score, evaluate, serial


def listwise_fit(rows, mass):
    X = np.vstack([r['X'] for r in rows])
    target = np.concatenate([r['y'].astype(float)/r['y'].sum() for r in rows])
    lengths = np.array([len(r['symbols']) for r in rows])
    group = np.repeat(np.arange(len(rows)), lengths)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    tau = .1

    def loss_grad(w):
        logits = np.einsum('ni,i->n', X, w, optimize=False)/tau
        maximum = np.maximum.reduceat(logits, starts)
        exp = np.exp(logits-maximum[group])
        total = np.bincount(group, weights=exp)
        probability = exp/total[group]
        logz = maximum+np.log(total)
        loss = float((logz.sum()-np.sum(target*logits))/len(rows) + ALPHA*np.sum(w*w)/2)
        grad = np.einsum('ni,n->i', X, (probability-target)/len(rows)/tau, optimize=False) + ALPHA*w
        return loss, grad

    def project(w):
        return np.r_[simplex(w[:len(CORE)], mass), l1ball(w[len(CORE):], 1-mass)]

    w = np.r_[np.full(len(CORE), mass/len(CORE)), np.zeros(len(SIDE))]
    a, t, L = w.copy(), 1., 10.
    for iteration in range(1, 20001):
        value, grad = loss_grad(a)
        for _ in range(30):
            candidate = project(a-grad/L)
            newloss, _ = loss_grad(candidate)
            change = candidate-a
            if newloss <= value + grad@change + L*np.sum(change*change)/2 + 1e-13:
                break
            L *= 2
        else:
            raise AssertionError('Line search did not converge')
        if np.linalg.norm(candidate-w) < 1e-10:
            _, check_grad = loss_grad(candidate)
            if np.linalg.norm(project(candidate-check_grad)-candidate) < 1e-7:
                w = candidate; break
            # A small accelerated step is not proof of constrained convergence.
            w = candidate; a = w.copy(); t = 1.
            continue
        nt = (1+np.sqrt(1+4*t*t))/2
        a = candidate + (t-1)/nt*(candidate-w)
        w, t, L = candidate, nt, max(1., L*.95)
    value, grad = loss_grad(w)
    residual = float(np.linalg.norm(project(w-grad)-w))
    assert residual < 1e-6, residual
    assert abs(w[:len(CORE)].sum()-mass)<1e-10 and np.all(w[:len(CORE)]>=0)
    assert np.abs(w[len(CORE):]).sum() <= 1-mass+1e-10
    return dict(name=f'listwise_core{int(mass*100)}_side{int(round((1-mass)*100))}', w=w,
                iterations=iteration, loss=value, stationarity_residual=residual)


def main():
    features = pd.DataFrame(read(P/'FEATURE_DIAGNOSTICS.json'))
    labels = read(O/'01a0b6b1-511d-7401-b4f7-4dd2aaa895a7/monthly_inverse_holdings/records.json')['monthly']
    rows = []
    for label in labels:
        z = features[features.date==label['rebalance_date']].sort_values('symbol')
        rows.append(dict(date=pd.Timestamp(label['rebalance_date']), end=pd.Timestamp(label['through_date']),
                         symbols=z.symbol.tolist(), X=z[CORE+SIDE].to_numpy()-.5,
                         y=z.selected.to_numpy(dtype=bool)))
    px, _, _ = load()
    results, models, curves = {}, {}, {}
    for mass in [.5,.6,.8]:
        model = listwise_fit(rows, mass); name = model['name']
        summary, comparisons, nav = evaluate(rows, [score(model, r) for r in rows], px)
        dump(name+'_FULL_EXPLANATION.json', comparisons)
        models[name] = {**serial(model), 'loss':model['loss'], 'stationarity_residual':model['stationarity_residual']}
        curves[name] = nav
        testrows, testscores, folds = [], [], []
        for year in range(2020,2027):
            train = [r for r in rows if r['date'].year<year]
            test = [r for r in rows if r['date'].year==year]
            assert train[-1]['date']<test[0]['date']
            fm = listwise_fit(train,mass)
            pred = [score(fm,r) for r in test]
            fs, _, _ = evaluate(test,pred,px)
            folds.append(dict(test_year=year,train_through=str(train[-1]['date'].date()),
                **serial(fm),**fs,stationarity_residual=fm['stationarity_residual']))
            testrows.extend(test); testscores.extend(pred)
        hold, hc, hnav = evaluate(testrows,testscores,px)
        dump(name+'_YEAR_HOLDOUT.json',hc)
        curves[name+'__year_holdout'] = hnav
        results[name] = dict(full_fit=summary, expanding_year_holdout=hold, folds=folds,
                            eligible_author_hypothesis=True,genuine_investment_out_of_sample=False)
        dump('RANKING_RESULTS.json',results)
        print(name, summary, hold, flush=True)
    dump('RANKING_MODELS.json',models)
    pd.DataFrame(curves).to_csv(P/'RANKING_NAV.csv.gz')
    for file, expected in read(P/'SOURCE_HASHES.json').items(): assert sha(file)==expected,file
    dump('RANKING_VALIDATION.json',dict(status='passed', fits=24, labels_changed=False,
        projected_stationarity_checks_passed=True, ranking_temperature=.1, protected_sources_unchanged=True))
    print('DONE',flush=True)


if __name__=='__main__': main()
