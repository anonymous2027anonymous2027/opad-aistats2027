"""Generalized R-hat on the three Ising targets, using recorded run moments."""
import json
import numpy as np
import pandas as pd
from om.experiments.common import (make_model,sampler_and_initial,make_search,new_run,
    save_metadata,plot_style,save_figure,TIME_METHODS,COLORS,STYLES)
from om.experiments.moments import compute_mcmc_expected_error,compute_greedy_expected_error


def generalized_rhat(means, within_variances):
    means=np.asarray(means);within_variances=np.asarray(within_variances)
    if means.shape!=within_variances.shape or means.ndim!=2 or means.shape[1]<2:
        raise ValueError('Need matching checkpoints-by-replicates arrays and at least two replicates.')
    w=within_variances.mean(axis=1)
    out=np.full(len(w),np.nan)
    good=(w>0)&np.isfinite(means).all(axis=1)&np.isfinite(within_variances).all(axis=1)
    out[good]=np.sqrt(1+means[good].var(axis=1,ddof=1)/w[good])
    return out


def run(name,args):
    folder=new_run(args.output_dir,'rhat_'+name)
    meta=dict(kind='rhat',model=name,chains=args.chains,seconds=args.seconds or 12.,
              records=args.records,status='running',completed_chains=0,burn_in=0,
              seeds=list(range(args.chains)),observable='sum of spins',
              timing='Recorded sampler/search evolution time, matching the R-hat experiment routines.')
    save_metadata(folder,meta)
    model=make_model(name)
    for rep in range(args.chains):
        print(f'R-hat {name}: replicate {rep+1}/{args.chains}',flush=True)
        sampler,initial=sampler_and_initial(name,rep)
        mc=compute_mcmc_expected_error('time',args.records,sampler,initial,model,np.sum,0.,
             total_sampling_time_per_chain_seconds=meta['seconds'])
        search=make_search(name,rep)
        nw=compute_greedy_expected_error('time',args.records,search,model,np.sum,0.,
             total_sampling_time_per_chain_seconds=meta['seconds'])
        grid=np.linspace(meta['seconds']/args.records,meta['seconds'],args.records)
        rows=[]
        for label,key in [('MCMC','MCMC'),('MCMC RB/WR','RB'),('MCMC OPAD','OPAD'),('MCMC OPAD+','OPAD+')]:
            rows.extend(dict(method=label,seconds=float(t),recorded_seconds=float(actual),mean=float(m),within_variance=float(v))
                        for t,actual,m,v in zip(grid,mc['Times'],mc[key+'.means'],mc[key+'.variances']))
        rows.extend(dict(method='NWSS',seconds=float(t),recorded_seconds=float(actual),mean=float(m),within_variance=float(v))
                    for t,actual,m,v in zip(grid,nw['Times'],nw['Means'],nw['Variances']))
        pd.DataFrame(rows).to_csv(folder/f'replicate_{rep:02d}.csv',index=False)
        meta['completed_chains']=rep+1;save_metadata(folder,meta)
    meta['status']='complete';save_metadata(folder,meta);plot(folder)
    return folder


def plot(folder):
    meta=json.loads((folder/'metadata.json').read_text())
    data=pd.concat([pd.read_csv(folder/f'replicate_{r:02d}.csv').assign(replicate=r)
                    for r in range(meta['completed_chains'])],ignore_index=True)
    plt=plot_style();fig,ax=plt.subplots(figsize=(7,4.8))
    fig.subplots_adjust(left=.18,right=.97,bottom=.18,top=.97);rows=[]
    for method in TIME_METHODS[:5]:
        frame=data[data.method==method]
        means=frame.pivot(index='seconds',columns='replicate',values='mean')
        variances=frame.pivot(index='seconds',columns='replicate',values='within_variance')
        values=generalized_rhat(means.to_numpy(),variances.to_numpy())
        ax.plot(means.index,values,label=method,color=COLORS[method],ls=STYLES[method],lw=3)
        rows.extend(dict(method=method,seconds=t,rhat=y) for t,y in zip(means.index,values))
    limits={'ising15':(.9999,1.003),'ising30':(.999,1.01),'ising40':(.999,1.02)}
    ax.set(xlabel='Time (seconds)',ylabel=r'$\widehat R$',xlim=(0,meta['seconds']),ylim=limits[meta['model']])
    ax.set_xticks([0,meta['seconds']/2,meta['seconds']]);ax.grid(alpha=.2)
    ax.legend(loc='upper right',framealpha=.9,handlelength=1.5,handletextpad=.5,labelspacing=.2)
    save_figure(fig,folder,meta['model']+'_rhat_time');plt.close(fig)
    pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False)
