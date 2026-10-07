"""From-scratch KL comparisons for all four submission panels."""
import numpy as np
import pandas as pd
from om.tools.opad import OPADDistribution
from om.tools.ratio_ht import RatioHT
from om.experiments.common import (make_model, reference, sampler_and_initial,
    make_search, new_run, save_metadata, plot_style, save_figure, KL_METHODS, COLORS, STYLES)


def distribution_kl(q, logp):
    keep = q > 0
    result = float(q[keep] @ (np.log(q[keep])-logp[keep]))
    if result < -1e-8:
        raise ArithmeticError('Negative KL beyond numerical tolerance.')
    return max(0.,result)


def support_kl(keys, ids, logp):
    from scipy.special import logsumexp
    indices = np.fromiter((ids[k] for k in keys),dtype=int)
    return max(0.,float(-logsumexp(logp[indices])))


def trajectory(name, seed, length, ids):
    sampler,state = sampler_and_initial(name,seed)
    initial = ids[state.tobytes()]
    states = np.empty(length,dtype=np.int32)
    proposals = np.empty(length,dtype=np.int32)
    alpha = np.empty(length)
    for i in range(length):
        state,ps,aa = sampler.next_sample_proposals_acceptances(state)
        states[i],proposals[i],alpha[i] = ids[state.tobytes()],ids[ps[0].tobytes()],aa[0]
    return dict(initial=initial,states=states,proposals=proposals,alpha=alpha)


def mc_records(primary, logu, logp, grid, include_initial=False):
    d = len(logu); n0 = 0
    counts=np.zeros(d); rb=np.zeros(d); seen=np.zeros(d,bool); plus=seen.copy()
    if include_initial: plus[primary['initial']]=True
    previous=np.r_[primary['initial'],primary['states'][:-1]]
    rows=[]
    for n in grid:
        s=primary['states'][n0:n];p=primary['proposals'][n0:n]
        a=primary['alpha'][n0:n];old=previous[n0:n]
        counts+=np.bincount(s,minlength=d)
        rb+=np.bincount(old,weights=1-a,minlength=d)+np.bincount(p,weights=a,minlength=d)
        seen[s]=True;plus[s]=True;plus[p]=True
        from scipy.special import logsumexp
        values=[distribution_kl(counts/n,logp),distribution_kl(rb/n,logp),
                max(0.,float(-logsumexp(logp[seen]))),max(0.,float(-logsumexp(logp[plus])))]
        for method,value in zip(['MCMC','MCMC RB/WR','MCMC OPAD','MCMC OPAD+'],values):
            rows.append(dict(method=method,evaluations=int(n),kl=value,status='valid',particles=0))
        n0=n
    return rows


def ht_records(primary, auxiliary, logu, logp, budget, records):
    u=np.exp(logu-np.max(logu))
    ht=RatioHT(u,np.zeros(len(u)))
    rmax=budget//9
    grid=np.unique(np.linspace(1,rmax,min(records,rmax),dtype=int))
    rows=[];diagnostics=[];last=0
    for r in grid:
        ht.update(primary['states'][8*last+7:8*r:8],auxiliary['states'][last:r])
        out=ht.estimate();value=np.nan
        if out['status']=='valid':
            value=distribution_kl(out['weights'],logp[out['ids']])
        rows.append(dict(method='Clyde-Ghosh HT',evaluations=int(9*r),kl=value,status=out['status'],particles=0))
        diagnostics.append(dict(r=int(r),evaluations=int(9*r),primary_evaluations=int(8*r),auxiliary_evaluations=int(r),
            status=out['status'],C_hat_scaled=out['C_hat'],overlap_count=out['overlap_count']))
        last=r
    return rows,diagnostics


def dpvi_records(name, seed, particles, budget, ids, logp):
    from om.samplers.dvpi import DiscreteParticleVariationalInference, FixedFirstDiscreteParticleVariationalInference
    from om.samplers.structure_mcmc import DAGParticleVariationalInference
    model=make_model(name);np.random.seed(seed)
    if name=='bsler':
        rng=np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[1])
        dpvi=DAGParticleVariationalInference(model,particles,rng)
    elif name=='bvs':
        dpvi=FixedFirstDiscreteParticleVariationalInference(model,particles)
    else:
        dpvi=DiscreteParticleVariationalInference(model,particles)
    plus=OPADDistribution(model)
    for state in dpvi.particles_scores.generate_all_states():plus.add_array(state)
    rows=[]
    # Run the full requested budget, including coordinate sweeps after convergence.
    for n in range(particles,budget+1,particles):
        candidates=dpvi.evolve_all_states_in_one_dim()
        for state in candidates.generate_all_states():plus.add_array(state)
        for method,dist in [('DPVI',dpvi.particles_scores),('DPVI OPAD+',plus)]:
            rows.append(dict(method=method,evaluations=n,kl=support_kl(dist.state_to_weight,ids,logp),
                             status='valid',particles=particles))
    return rows


def run(name, args):
    folder=new_run(args.output_dir,'kl_'+name)
    particles=args.particles or ([10,100] if name=='ising15' else [100])
    meta=dict(kind='kl',model=name,chains=args.chains,evaluations=args.evaluations,particles=particles,
        status='running',completed_chains=0,burn_in=0,primary_seeds=list(range(args.chains)),
        auxiliary_seeds=list(range(100000,100000+args.chains)),ht_allocation=[8,1],
        unit='proposal attempts' if name=='bsler' else 'target evaluations')
    save_metadata(folder,meta)
    print(f'KL {name}: preparing target',flush=True)
    _,ids,logu,logp,_=reference(name)
    if any(k>len(ids) for k in particles):
        raise ValueError('DPVI particle count exceeds the number of states.')
    grid=np.unique(np.linspace(1,args.evaluations,args.records,dtype=int))
    for rep in range(args.chains):
        print(f'KL {name}: replicate {rep+1}/{args.chains}',flush=True)
        primary=trajectory(name,rep,args.evaluations,ids)
        auxiliary=trajectory(name,100000+rep,args.evaluations//9,ids)
        rows=mc_records(primary,logu,logp,grid,include_initial=name=='bsler')
        ht,diag=ht_records(primary,auxiliary,logu,logp,args.evaluations,args.records)
        rows.extend(ht)
        search=make_search(name,rep);step=0
        for n in grid:
            while step<n and search.candid_to_predicted_score:
                search.evolve();step+=1
            rows.append(dict(method='NWSS',evaluations=int(n),kl=support_kl(search.inner_opad.state_to_weight,ids,logp),
                status='valid',particles=0,actual_evaluations=step))
        for k in particles:rows.extend(dpvi_records(name,rep,k,args.evaluations,ids,logp))
        pd.DataFrame(rows).to_csv(folder/f'replicate_{rep:02d}.csv',index=False)
        pd.DataFrame(diag).to_csv(folder/f'ht_diagnostics_{rep:02d}.csv',index=False)
        meta['completed_chains']=rep+1;save_metadata(folder,meta)
    meta['status']='complete';save_metadata(folder,meta)
    plot(folder)
    return folder


def plot(folder):
    import json
    meta=json.loads((folder/'metadata.json').read_text())
    reps=[pd.read_csv(folder/f'replicate_{i:02d}.csv').assign(replicate=i) for i in range(meta['completed_chains'])]
    data=pd.concat(reps,ignore_index=True);plt=plot_style();summary=[]
    for k in meta['particles']:
        fig,ax=plt.subplots(figsize=(7,4.8));fig.subplots_adjust(left=.14,right=.97,bottom=.18,top=.97)
        for method in KL_METHODS:
            frame=data[(data.method==method)&(data.particles.isin([0,k]))]
            table=frame.pivot(index='evaluations',columns='replicate',values='kl')
            valid=table.notna().all(axis=1)&(table.shape[1]==meta['completed_chains'])
            mean=table.mean(axis=1).where(valid)
            lo=table.quantile(.05,axis=1).where(valid);hi=table.quantile(.95,axis=1).where(valid)
            ax.plot(table.index,mean,color=COLORS[method],ls='-.' if method=='MCMC RB/WR' else STYLES[method],lw=3,label=method)
            ax.fill_between(table.index,lo,hi,color=COLORS[method],alpha=.12,linewidth=0)
            summary.extend(dict(particles=k,method=method,evaluations=int(n),kl=m,lower=l,upper=h)
                           for n,m,l,h in zip(table.index,mean,lo,hi))
        ax.set(xlabel='Total '+meta['unit'],ylabel='KL',yscale='log',xlim=(0,meta['evaluations']))
        if meta['model']=='bvs':ax.set_ylim(.01,10)
        if meta['model']=='ising15':ax.set_ylim(.1,10)
        from matplotlib.ticker import NullFormatter
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([0,meta['evaluations']/2,meta['evaluations']]);ax.grid(alpha=.2)
        ax.legend(loc='upper center',framealpha=.9,handlelength=1.5,handletextpad=.5,labelspacing=.2,borderpad=.3)
        save_figure(fig,folder,f'{meta["model"]}_dpvi{k}_kl');plt.close(fig)
    pd.DataFrame(summary).to_csv(folder/'summary.csv',index=False)
    print('Saved KL plots:',folder,flush=True)
