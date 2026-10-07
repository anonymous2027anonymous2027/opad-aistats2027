"""bvs MSE, absolute bias and variance with six estimators.

Run through: python reproduce.py tradeoff --model bvs
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from om.models.var_select import BayesianVarSelectModel, fetch_mince_nutrition_data, standardize_add_one_column
from om.samplers.mh import FixedFirstElementArraySampler
from om.samplers.greedy import GreedyExplore
from om.tools.ratio_ht import RatioHT, ratio_weights

METHODS = ['MCMC', 'MCMC RB/WR', 'MCMC OPAD', 'MCMC OPAD+', 'NWSS', 'Clyde-Ghosh HT']


def chain(model, ids, seed, seconds=None, minimum=0, length=None):
    """Retain full transition records; time kernel calls, including initialization.

    Recording IDs, diagnostic storage and disk I/O are excluded. The existing
    sampler is called directly, with its own independently seeded random stream.
    """
    start = time.perf_counter()
    np.random.seed(seed)
    sampler = FixedFirstElementArraySampler(model)
    state = model.generate_init_state()
    initial = ids[state.tobytes()]
    elapsed = time.perf_counter() - start
    initialization_seconds = elapsed
    visited, proposed, acceptance, times = [], [], [], []
    while (len(visited) < length if length is not None else
           (elapsed < seconds or len(visited) < minimum)):
        start = time.perf_counter()
        state, proposals, alphas = sampler.next_sample_proposals_acceptances(state)
        elapsed += time.perf_counter() - start
        visited.append(ids[state.tobytes()])
        proposed.append(ids[proposals[0].tobytes()])
        acceptance.append(alphas[0])
        times.append(elapsed)
    return dict(initial=np.array(initial), states=np.array(visited, dtype=np.uint16),
                proposals=np.array(proposed, dtype=np.uint16), alpha=np.array(acceptance),
                seconds=np.array(times), seed=np.array(seed),
                initialization_seconds=np.array(initialization_seconds))


def kl(prob, logp):
    mask = prob > 0
    return float(np.sum(prob[mask] * (np.log(prob[mask])-logp[mask])))


def baselines(primary, u, f, logp, eval_grid):
    """Replay the same primary trajectory for four estimators with separate clocks.

    Each clock is primary kernel time PLUS that estimator's cumulative update
    and expectation-computation time. Work on other estimators is never charged.
    All expectations use cumulative prefixes; recording is not thinning.
    """
    n, d = len(primary['states']), len(u)
    checkpoints = np.unique(np.r_[eval_grid, np.arange(1024, n+1, 1024), n])
    checkpoints = checkpoints[checkpoints <= n]
    setup = time.perf_counter()
    counts = np.zeros(d)
    mc_extra = time.perf_counter()-setup
    setup = time.perf_counter()
    rb_weights = np.zeros(d)
    rb_extra = time.perf_counter()-setup
    setup = time.perf_counter()
    seen = np.zeros(d, dtype=bool)
    opad_extra = time.perf_counter()-setup
    setup = time.perf_counter()
    seen_plus = np.zeros(d, dtype=bool)
    plus_extra = time.perf_counter()-setup
    out = []
    previous_ids = np.r_[primary['initial'], primary['states'][:-1]]
    start_index = 0
    for end in checkpoints:
        s = primary['states'][start_index:end]
        a = primary['alpha'][start_index:end]
        proposed = primary['proposals'][start_index:end]
        previous = previous_ids[start_index:end]
        start = time.perf_counter()
        counts += np.bincount(s, minlength=d)
        mc_prob = counts/end
        mc_mean = mc_prob @ f
        mc_extra += time.perf_counter()-start
        start = time.perf_counter()
        rb_weights += (np.bincount(previous, weights=1-a, minlength=d)
                       + np.bincount(proposed, weights=a, minlength=d))
        rb_prob = rb_weights/end
        rb_mean = rb_prob @ f
        rb_extra += time.perf_counter()-start
        start = time.perf_counter()
        seen[s] = True
        opad_prob = u*seen
        opad_prob /= opad_prob.sum()
        opad_mean = opad_prob @ f
        opad_extra += time.perf_counter()-start
        start = time.perf_counter()
        seen_plus[s] = True
        seen_plus[proposed] = True
        plus_prob = u*seen_plus
        plus_prob /= plus_prob.sum()
        plus_mean = plus_prob @ f
        plus_extra += time.perf_counter()-start
        for name, mean, prob, extra in zip(METHODS[:4],
                [mc_mean, rb_mean, opad_mean, plus_mean],
                [mc_prob, rb_prob, opad_prob, plus_prob],
                [mc_extra, rb_extra, opad_extra, plus_extra]):
            out.append(dict(method=name, evaluations=int(end), seconds=primary['seconds'][end-1]+extra,
                            estimate=mean, kl=kl(prob, logp), status='valid'))
        start_index = end
    return out


def ht_records(primary, auxiliary, u, f, logp, r_grid):
    start = time.perf_counter()
    estimator = RatioHT(u, f)
    extra = time.perf_counter()-start
    max_r = min(len(primary['states'])//8, len(auxiliary['states']))
    checkpoints = np.unique(np.r_[r_grid, np.arange(128, max_r+1, 128), max_r])
    checkpoints = checkpoints[checkpoints <= max_r]
    previous_r = 0
    rows, diagnostics = [], []
    detailed = {k: [] for k in ['r', 'state_id', 'p_hat', 'pi_hat', 'oracle_pi', 'weight', 'oracle_weight']}
    for r in checkpoints:
        start = time.perf_counter()
        estimator.update(primary['states'][8*previous_r+7:8*r:8], auxiliary['states'][previous_r:r])
        result = estimator.estimate()
        extra += time.perf_counter()-start
        total_seconds = primary['seconds'][8*r-1] + auxiliary['seconds'][r-1] + extra
        ids = result['ids']
        oracle_w, oracle_pi = ratio_weights(u[ids], np.exp(logp[ids]), int(r))
        oracle_mean = oracle_w @ f[ids]
        oracle_kl = float(np.sum(oracle_w*(np.log(oracle_w)-logp[ids])))
        ht_kl = (float(np.sum(result['weights']*(np.log(result['weights'])-logp[ids])))
                 if result['status'] == 'valid' else np.nan)
        rows.append(dict(method=METHODS[5], evaluations=int(9*r), seconds=total_seconds,
                         estimate=result['estimate'], kl=ht_kl, status=result['status']))
        # Oracle is diagnostic only: use the SAME budget/time coordinates as HT.
        rows.append(dict(method='Oracle-C HT', evaluations=int(9*r), seconds=total_seconds,
                         estimate=oracle_mean, kl=oracle_kl, status='valid'))
        diagnostics.append(dict(r=int(r), evaluations=int(9*r), seconds=total_seconds,
            primary_evaluations=int(8*r), auxiliary_evaluations=int(r),
            primary_seconds=float(primary['seconds'][8*r-1]),
            auxiliary_seconds=float(auxiliary['seconds'][r-1]), ht_seconds=extra,
            C_hat=result['C_hat'], C_exact=1/u.sum(),
            C_ratio=result['C_hat']*u.sum(), overlap_count=result['overlap_count'],
            S_size=len(ids), A_size=result['auxiliary_size'], status=result['status'],
            p_hat_max=float(result['p_hat'].max()), pi_hat_min=float(result['pi_hat'].min()),
            pi_hat_max=float(result['pi_hat'].max()),
            oracle_estimate=oracle_mean, estimate=result['estimate']))
        if r in r_grid:
            detailed['r'].append(np.full(len(ids), r, dtype=np.int32))
            detailed['state_id'].append(ids.astype(np.uint16))
            detailed['p_hat'].append(result['p_hat'])
            detailed['pi_hat'].append(result['pi_hat'])
            detailed['oracle_pi'].append(oracle_pi)
            detailed['weight'].append(result['weights'] if result['weights'] is not None else np.full(len(ids), np.nan))
            detailed['oracle_weight'].append(oracle_w)
        previous_r = r
    return rows, diagnostics, {k: np.concatenate(v) for k, v in detailed.items()}


def nwss_records(model, ids, u, f, logp, seed, budget, seconds, eval_grid):
    start = time.perf_counter()
    np.random.seed(seed)
    random.seed(seed)
    search = GreedyExplore(model, possible_values=(0, 1), fixed_first_value=1)
    search.initialize('NWSS', model.generate_init_state())
    elapsed = time.perf_counter()-start
    rows = []
    step, last_record = 0, 0.
    while search.candid_to_predicted_score and (step < budget or elapsed < seconds):
        start = time.perf_counter()
        search.evolve()
        elapsed += time.perf_counter()-start
        step += 1
        if step in eval_grid or elapsed-last_record >= .03 or not search.candid_to_predicted_score:
            start = time.perf_counter()
            support = np.fromiter((ids[k] for k in search.inner_opad.state_to_weight), dtype=np.int64)
            weights = u[support]
            mean = weights @ f[support]/weights.sum()
            elapsed += time.perf_counter()-start
            rows.append(dict(method='NWSS', evaluations=step, seconds=elapsed, estimate=mean,
                kl=float(np.log(u.sum())-np.log(weights.sum())), status='valid',
                support_size=len(support)))
            last_record = elapsed
    return rows, dict(exhausted=not search.candid_to_predicted_score,
                      actual_evaluations=step, actual_seconds=elapsed,
                      final_support_size=len(search.inner_opad.state_to_weight))


def run(args):
    folder = Path(args.output_dir)/datetime.now().strftime('BVS_HT_%Y-%m-%d_%H-%M-%S_%f')
    folder.mkdir(parents=True, exist_ok=False)
    data = ROOT/'om/models/lifespan-merged.csv'
    meta = dict(status='running', completed_chains=0, chains=args.chains,
        evaluations=args.evaluations, seconds=args.seconds, burn_in_primary=0, burn_in_auxiliary=0,
        thinning=8, allocation=[8,1], primary_seeds=list(range(args.chains)),
        auxiliary_seeds=list(range(100000,100000+args.chains)),
        target=dict(a=3., b=1., pi=.5, linear_algebra='numpy.linalg.pinv'),
        data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        source='Clyde and Ghosh (2012), Biometrika 99(4):981-988, Sections 3-5',
        evaluation_policy='One proposal/transition per evaluation, as in existing BVS plots; no burn-in. Initial states are not samples.',
        timing_policy='Summed sequential kernel runtime plus method-specific updates and expectation calculation, including random initialization. Excludes shared target-table construction, truth/oracle/KL diagnostics and disk I/O for every method.',
        shared_setup='The unchanged BVS model precomputes its unnormalized target table. This common setup is outside the sampling benchmark, as in existing experiments; exact C is unavailable to the main estimators.',
        invalid_policy='No clipping or fallback. Invalid checkpoints have NaN estimates; aggregate curves require all replicates valid.',
        time_sampling='Last completed checkpoint at or before the displayed time; never interpolation from future estimates.',
        variance='Sample variance ddof=1; MSE is mean squared error; absolute bias is absolute mean signed error',
        nwss={}, implementation_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in [Path(__file__), ROOT/'om/tools/ratio_ht.py',ROOT/'om/models/var_select.py',ROOT/'om/samplers/mh.py',ROOT/'om/samplers/greedy.py']})
    def save():
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save()
    print(f'Results: {folder}', flush=True)
    print('Building the unchanged BVS target once...', flush=True)
    x, y, names = fetch_mince_nutrition_data(str(data))
    x, _ = standardize_add_one_column(x,names)
    from om.experiments.common import make_model
    model = make_model('bvs')
    states = np.array(model.generate_all_states())
    ids = {state.tobytes(): i for i,state in enumerate(states)}
    neglog = np.array([model.calc_neg_log_unnormalized_prob(s) for s in states])
    u, f = np.exp(-neglog), states.sum(axis=1)
    logp = -neglog-np.log(u.sum())
    truth = float(u @ f/u.sum())
    np.testing.assert_allclose(truth,3.967976168799272,atol=1e-8,rtol=0)
    meta.update(ground_truth=truth, C_exact=1/u.sum(), Z_exact=float(u.sum()), states=len(states))
    meta['target']['c'] = len(x)
    np.savez_compressed(folder/'target_diagnostics.npz',states=states,u=u,f=f,log_p=logp)
    r_grid = np.unique(np.r_[np.arange(10,args.evaluations//9+1,10), args.evaluations//9])
    eval_grid = np.unique(np.r_[9*r_grid,args.evaluations])
    np.savetxt(folder/'evaluation_grid.csv',eval_grid,fmt='%d',header='evaluations',comments='')
    save()
    for rep in range(args.chains):
        repdir=folder/f'replicate_{rep:02d}';repdir.mkdir()
        print(f'Replicate {rep+1}/{args.chains}: primary chain...',flush=True)
        primary=chain(model,ids,rep,seconds=args.seconds,minimum=args.evaluations)
        np.savez_compressed(repdir/'primary_trajectory.npz',**primary)
        print(f'Replicate {rep+1}/{args.chains}: auxiliary and estimators...',flush=True)
        auxiliary=chain(model,ids,100000+rep,length=len(primary['states'])//8)
        np.savez_compressed(repdir/'auxiliary_trajectory.npz',**auxiliary)
        records=baselines(primary,u,f,logp,eval_grid)
        ht,diag,detail=ht_records(primary,auxiliary,u,f,logp,r_grid)
        records.extend(ht)
        nwss,info=nwss_records(model,ids,u,f,logp,rep,args.evaluations,args.seconds,set(eval_grid))
        records.extend(nwss)
        pd.DataFrame(records).to_csv(repdir/'estimates.csv',index=False)
        pd.DataFrame(diag).to_csv(repdir/'ht_diagnostics.csv',index=False)
        np.savez_compressed(repdir/'ht_inclusion_probabilities.npz',**detail)
        meta['nwss'][str(rep)]=info
        meta['completed_chains']=rep+1;save()
        bad=pd.DataFrame(diag).query("status != 'valid'")
        print(f'Replicate {rep+1}/{args.chains} saved; {len(primary["states"])} primary transitions; '
              f'{len(bad)} invalid HT checkpoints; NWSS support={info["final_support_size"]}.',flush=True)
    meta['status']='complete';save()
    return folder


def summarize(folder):
    meta=json.loads((folder/'metadata.json').read_text())
    reps=[pd.read_csv(folder/f'replicate_{i:02d}/estimates.csv') for i in range(meta['completed_chains'])]
    evaluation_grid=pd.read_csv(folder/'evaluation_grid.csv')['evaluations'].to_numpy()
    time_grid=np.linspace(meta['seconds']/100,meta['seconds'],100)
    rows=[]
    for axis,grid in [('evaluations',evaluation_grid),('seconds',time_grid)]:
        for method in METHODS+['Oracle-C HT']:
            estimates=np.full((len(grid),len(reps)),np.nan)
            kls=estimates.copy();used=estimates.copy()
            for rep,frame in enumerate(reps):
                v=frame[frame.method==method].sort_values(axis)
                x=v[axis].to_numpy()
                index=np.searchsorted(x,grid,side='right')-1
                valid=index>=0
                # HT ends at its actual budget checkpoint: do not label 19998 as 20000.
                if axis=='evaluations' and method in ['Clyde-Ghosh HT','Oracle-C HT']:
                    valid &= grid<=9*(meta['evaluations']//9)
                estimates[valid,rep]=v.estimate.to_numpy()[index[valid]]
                kls[valid,rep]=v.kl.to_numpy()[index[valid]]
                used[valid,rep]=x[index[valid]]
            errors=estimates-meta['ground_truth']
            for j,b in enumerate(grid):
                good=np.isfinite(errors[j]);all_valid=good.all()
                rows.append(dict(axis=axis,budget=b,method=method,valid_replicates=int(good.sum()),
                    mse=float(np.mean(errors[j]**2)) if all_valid else np.nan,
                    abs_bias=float(abs(np.mean(errors[j]))) if all_valid else np.nan,
                    variance=float(np.var(errors[j],ddof=1)) if all_valid and len(reps)>1 else np.nan,
                    kl=float(np.mean(kls[j])) if all_valid else np.nan,
                    mean_actual_checkpoint=float(np.nanmean(used[j])) if np.isfinite(used[j]).any() else np.nan))
            np.savez_compressed(folder/f'{method.replace("/","_").replace(" ","_")}_{axis}_aligned.npz',
                                grid=grid,estimates=estimates,kl=kls,actual_checkpoint=used)
    summary=pd.DataFrame(rows);summary.to_csv(folder/'summary.csv',index=False)
    diagnostics=pd.concat([pd.read_csv(folder/f'replicate_{i:02d}/ht_diagnostics.csv').assign(replicate=i)
                           for i in range(len(reps))],ignore_index=True)
    diagnostics.to_csv(folder/'ht_diagnostics.csv',index=False)
    return summary,diagnostics,meta


def plot(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from om.experiments.plot_bvs_ht_paper_style import main as plot_comparison
    # Reuse saved statistics; a fresh experiment needs alignment only once.
    if (folder/'summary.csv').exists():
        summary=pd.read_csv(folder/'summary.csv')
        diagnostics=pd.read_csv(folder/'ht_diagnostics.csv')
        meta=json.loads((folder/'metadata.json').read_text())
    else:
        summary,diagnostics,meta=summarize(folder)
    plot_comparison(folder)
    colors=['blue','orange','red','black','#00BFC4','#7B2CBF']
    styles=['-',':','-','-','-','--']
    fig,axes=plt.subplots(1,3,figsize=(15,4.4))
    for rep in range(meta['completed_chains']):
        v=diagnostics[(diagnostics.replicate==rep)&(diagnostics.evaluations<=meta['evaluations'])]
        axes[0].plot(v.evaluations,v.C_ratio,color='#7B2CBF',alpha=.3,lw=1)
    axes[0].axhline(1,color='black',ls='--')
    axes[0].set(ylabel=r'$\widehat C/C$',xlabel='Total target evaluations')
    for method,color,style in [('Clyde-Ghosh HT','#7B2CBF','--'),('Oracle-C HT','gray','-'),('MCMC OPAD','red','-')]:
        v=summary[(summary.axis=='evaluations')&(summary.method==method)]
        axes[1].plot(v.budget,v.mse,color=color,ls=style,lw=2.5,label=method)
        stem=method.replace('/','_').replace(' ','_')
        with np.load(folder/f'{stem}_evaluations_aligned.npz') as data:
            errors=data['estimates']-meta['ground_truth']
            valid=np.isfinite(errors).all(axis=1)
            low=np.full(len(errors),np.nan);high=low.copy()
            low[valid],high[valid]=np.quantile(errors[valid]**2,[.05,.95],axis=1)
            axes[1].fill_between(data['grid'],low,high,color=color,alpha=.1,linewidth=0)
    axes[1].set(yscale='log',ylabel='MSE',xlabel='Total target evaluations')
    axes[1].legend(fontsize=11)
    for method,color,style in zip(METHODS,colors,styles):
        v=summary[(summary.axis=='evaluations')&(summary.method==method)]
        axes[2].plot(v.budget,v.kl,color=color,ls=style,lw=2.5,label=method)
    axes[2].set(yscale='log',ylim=(.00001,10),ylabel='KL',xlabel='Total target evaluations')
    for ax in axes:
        ax.grid(alpha=.2);ax.set_xlim(0,meta['evaluations'])
    fig.text(.5,.01,'MSE shading: empirical 5th--95th percentiles of squared errors across independent replicates.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.035,1,1])
    fig.savefig(folder/'bvs_ht_diagnostics_paper_style.pdf')
    fig.savefig(folder/'bvs_ht_diagnostics_paper_style.png',dpi=180)
    plt.close(fig)
    print('Saved plots to',folder,flush=True)
