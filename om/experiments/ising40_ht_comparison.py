"""ising40 MSE, absolute bias and variance with six estimators.

Run through: python reproduce.py tradeoff --model ising40
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from om.models.ising1D_high_dim import IsingModel1D_Large_Symmetric
from om.samplers.mh import BiValueArrayMetropolisSampler
from om.samplers.greedy import GreedyExplore
from om.tools.ratio_ht import RatioHT

METHODS=['MCMC','MCMC RB/WR','MCMC OPAD','MCMC OPAD+','NWSS','Clyde-Ghosh HT']


class RecordedIsing(IsingModel1D_Large_Symmetric):
    """Observe scores already evaluated by the unchanged kernel; never cache them."""
    def __init__(self,N=40):
        super().__init__(N=N,J=1,beta=.5)
        self.ids={};self.log_u=[];self.f=[]

    def calc_neg_log_unnormalized_prob(self,state):
        value=super().calc_neg_log_unnormalized_prob(state)
        key=state.tobytes()
        if key not in self.ids:
            self.ids[key]=len(self.ids)
            self.log_u.append(-float(value))
            self.f.append(float(state.sum()))
        return value


def chain(model,seed,seconds=None,length=None):
    start=time.perf_counter()
    np.random.seed(seed)
    sampler=BiValueArrayMetropolisSampler(model)
    state=model.generate_init_state();initial_key=state.tobytes()
    elapsed=time.perf_counter()-start
    initial_seconds=elapsed
    states=[];proposals=[];alpha=[];times=[]
    while (len(states)<length if length is not None else elapsed<seconds):
        start=time.perf_counter()
        state,proposed,acceptance=sampler.next_sample_proposals_acceptances(state)
        elapsed+=time.perf_counter()-start
        states.append(model.ids[state.tobytes()])
        proposals.append(model.ids[proposed[0].tobytes()])
        alpha.append(acceptance[0]);times.append(elapsed)
    return dict(initial=np.array(model.ids[initial_key]),states=np.array(states,dtype=np.uint32),
                proposals=np.array(proposals,dtype=np.uint32),alpha=np.array(alpha),
                seconds=np.array(times),seed=np.array(seed),initialization_seconds=np.array(initial_seconds))


def baseline_records(primary,log_u,f):
    """Online sums and unique supports, with a separate clock for each estimator."""
    start=time.perf_counter();mc_total=0.;mc_extra=time.perf_counter()-start
    start=time.perf_counter();rb_total=0.;rb_extra=time.perf_counter()-start
    start=time.perf_counter();seen=set();opad_num=opad_den=0.;opad_extra=time.perf_counter()-start
    start=time.perf_counter();plus=set();plus_num=plus_den=0.;plus_extra=time.perf_counter()-start
    n=len(primary['states']);last=0;rows=[]
    previous=np.r_[primary['initial'],primary['states'][:-1]]
    for end in np.unique(np.r_[np.arange(72,n+1,72),n]):
        s=primary['states'][last:end];p=primary['proposals'][last:end]
        a=primary['alpha'][last:end];old=previous[last:end]
        start=time.perf_counter()
        mc_total+=f[s].sum();mc_mean=mc_total/end
        mc_extra+=time.perf_counter()-start
        start=time.perf_counter()
        rb_total+=np.sum((1-a)*f[old]+a*f[p]);rb_mean=rb_total/end
        rb_extra+=time.perf_counter()-start
        start=time.perf_counter()
        for state_id in s:
            i=int(state_id)
            if i not in seen:
                seen.add(i);u=np.exp(log_u[i]);opad_den+=u;opad_num+=u*f[i]
        opad_mean=opad_num/opad_den
        opad_extra+=time.perf_counter()-start
        start=time.perf_counter()
        for state_id in np.r_[s,p]:
            i=int(state_id)
            if i not in plus:
                plus.add(i);u=np.exp(log_u[i]);plus_den+=u;plus_num+=u*f[i]
        plus_mean=plus_num/plus_den
        plus_extra+=time.perf_counter()-start
        for method,mean,extra in zip(METHODS[:4],[mc_mean,rb_mean,opad_mean,plus_mean],
                                    [mc_extra,rb_extra,opad_extra,plus_extra]):
            rows.append(dict(method=method,evaluations=int(end),seconds=primary['seconds'][end-1]+extra,
                             estimate=mean,status='valid'))
        last=end
    return rows


def ht_records(primary,auxiliary,log_u,f):
    # Only values on S or A are used; no sum over the model space is available.
    # Exp conversion is performed lazily when a state first enters either support.
    start=time.perf_counter()
    u=np.zeros(len(log_u));known=np.zeros(len(log_u),dtype=bool)
    estimator=RatioHT(u,f);extra=time.perf_counter()-start
    max_r=min(len(primary['states'])//8,len(auxiliary['states']))
    rows=[];diagnostics=[];last=0
    for r in np.unique(np.r_[np.arange(8,max_r+1,8),max_r]):
        start=time.perf_counter()
        retained=primary['states'][8*last+7:8*r:8];aux=auxiliary['states'][last:r]
        selected=np.unique(np.r_[retained,aux])
        new=selected[~known[selected]]
        u[new]=np.exp(log_u[new]);known[new]=True
        estimator.update(retained,aux)
        out=estimator.estimate()
        extra+=time.perf_counter()-start
        elapsed=primary['seconds'][8*r-1]+auxiliary['seconds'][r-1]+extra
        rows.append(dict(method='Clyde-Ghosh HT',evaluations=int(9*r),seconds=elapsed,
                         estimate=out['estimate'],status=out['status']))
        diagnostics.append(dict(r=int(r),evaluations=int(9*r),primary_evaluations=int(8*r),
            auxiliary_evaluations=int(r),primary_seconds=primary['seconds'][8*r-1],
            auxiliary_seconds=auxiliary['seconds'][r-1],ht_seconds=extra,seconds=elapsed,
            C_hat=out['C_hat'],overlap_count=out['overlap_count'],S_size=len(out['ids']),
            A_size=out['auxiliary_size'],status=out['status'],
            p_hat_max=float(out['p_hat'].max()),pi_hat_min=float(out['pi_hat'].min()),
            pi_hat_max=float(out['pi_hat'].max())))
        last=r
    return rows,diagnostics


def nwss_records(model,seed,seconds):
    start=time.perf_counter()
    np.random.seed(seed);random.seed(seed)
    search=GreedyExplore(model)
    search.initialize('NWSS',model.generate_init_state())
    elapsed=time.perf_counter()-start
    last=0.;step=0;rows=[]
    while elapsed<seconds and search.candid_to_predicted_score:
        start=time.perf_counter();search.evolve();elapsed+=time.perf_counter()-start
        step+=1
        if step%72==0 or elapsed-last>=.1 or elapsed>=seconds:
            start=time.perf_counter()
            numerator=0.
            for key,weight in search.inner_opad.state_to_weight.items():
                numerator+=weight*model.f[model.ids[key]]
            estimate=numerator/search.inner_opad.total_weight
            elapsed+=time.perf_counter()-start
            rows.append(dict(method='NWSS',evaluations=step,seconds=elapsed,estimate=estimate,status='valid'))
            last=elapsed
    return rows


def validate(primary,auxiliary,records,diagnostics,log_u,f):
    """Direct prefix checks outside all estimator clocks."""
    table=pd.DataFrame(records);diag=pd.DataFrame(diagnostics)
    np.testing.assert_allclose(diag.seconds,diag.primary_seconds+diag.auxiliary_seconds+diag.ht_seconds,atol=1e-12)
    assert (diag.evaluations==9*diag.r).all()
    for n in [72,(len(primary['states'])//72)*72]:
        s=primary['states'][:n];p=primary['proposals'][:n];a=primary['alpha'][:n]
        prev=np.r_[primary['initial'],s[:-1]]
        ids=np.unique(s);plus=np.union1d(s,p)
        u=np.exp(log_u[ids]);up=np.exp(log_u[plus])
        expected=[f[s].mean(),np.mean((1-a)*f[prev]+a*f[p]),u@f[ids]/u.sum(),up@f[plus]/up.sum()]
        for method,value in zip(METHODS[:4],expected):
            found=table[(table.method==method)&(table.evaluations==n)].estimate.iloc[0]
            np.testing.assert_allclose(found,value,atol=1e-9,rtol=1e-10)
    for pos in [0,len(diag)//2,len(diag)-1]:
        row=diag.iloc[pos];r=int(row.r)
        retained=primary['states'][7:8*r:8];a=np.unique(auxiliary['states'][:r])
        c=np.isin(retained,a).mean()/np.exp(log_u[a]).sum()
        np.testing.assert_allclose(c,row.C_hat,atol=1e-28,rtol=1e-10)
        if row.status=='valid':
            ids=np.unique(retained);u=np.exp(log_u[ids]);pi=-np.expm1(r*np.log1p(-c*u))
            w=u/pi;expected=w@f[ids]/w.sum()
            found=table[(table.method==METHODS[5])&(table.evaluations==9*r)].estimate.iloc[0]
            np.testing.assert_allclose(found,expected,atol=1e-9,rtol=1e-10)


def run(args):
    directory=Path(args.output_dir)/datetime.now().strftime('Ising40_HT_%Y-%m-%d_%H-%M-%S_%f')
    directory.mkdir(parents=True,exist_ok=False)
    meta=dict(status='running',completed_chains=0,chains=args.chains,seconds=args.seconds,
              N=40,J=1,beta=.5,h=0,observable='sum of spins',ground_truth=0.,
              energy='Unchanged IsingModel1D_Large_Symmetric; each bond is counted from both endpoints.',
              burn_in=0,thinning=8,allocation=[8,1],primary_seeds=list(range(args.chains)),
              auxiliary_seeds=list(range(100000,100000+args.chains)),
              timing='Sequential summed kernel time plus each estimator updates and mean calculations; includes initialization. Recording, disk I/O and validation excluded for every method.',
              score_recording='Record existing target evaluations without caching/replacing them. Bookkeeping inside kernel calls is included in the common sampling clock.',
              evaluations='One proposal per transition; initial states are not retained samples. HT=8r+r=9r. Evaluation plots use only prefixes available from the time runs.',
              invalid='Record undefined HT checkpoints without clipping or substitution; aggregate only when all replicates valid.',
              statistics='MSE=mean(error^2); abs bias=abs(mean(error)); variance=sample variance ddof=1. MSE shading is empirical 5th--95th percentiles of squared errors.',
              source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [Path(__file__).resolve(),ROOT/'om/models/ising1D_high_dim.py',ROOT/'om/samplers/mh.py',ROOT/'om/samplers/greedy.py',ROOT/'om/tools/ratio_ht.py']})
    def save(): (directory/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    save();print('Results:',directory,flush=True)
    for rep in range(args.chains):
        d=directory/f'replicate_{rep:02d}';d.mkdir()
        model=RecordedIsing()
        print(f'Replicate {rep+1}/{args.chains}: primary chain...',flush=True)
        primary=chain(model,rep,seconds=args.seconds)
        auxiliary=chain(model,100000+rep,length=len(primary['states'])//8)
        np.savez_compressed(d/'primary_trajectory.npz',**primary)
        np.savez_compressed(d/'auxiliary_trajectory.npz',**auxiliary)
        log_u=np.array(model.log_u);f=np.array(model.f)
        records=baseline_records(primary,log_u,f)
        ht,diagnostics=ht_records(primary,auxiliary,log_u,f);records.extend(ht)
        validate(primary,auxiliary,records,diagnostics,log_u,f)
        print(f'Replicate {rep+1}/{args.chains}: NWSS...',flush=True)
        records.extend(nwss_records(model,rep,args.seconds))
        packed=np.array([np.packbits(np.frombuffer(key,dtype=np.int64)>0) for key in model.ids],dtype=np.uint8)
        np.savez_compressed(d/'encountered_states.npz',packed_spins=packed,log_u=np.array(model.log_u),f=np.array(model.f))
        pd.DataFrame(records).to_csv(d/'estimates.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(d/'ht_diagnostics.csv',index=False)
        meta['completed_chains']=rep+1;save()
        bad=sum(row['status']!='valid' for row in diagnostics)
        print(f'Replicate {rep+1}/{args.chains} saved; {len(primary["states"])} primary transitions; {bad} invalid HT checkpoints.',flush=True)
    meta['status']='complete';save()
    return directory


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta=json.loads((directory/'metadata.json').read_text())
    assert meta['status']=='complete'
    reps=[pd.read_csv(directory/f'replicate_{i:02d}/estimates.csv') for i in range(meta['chains'])]
    # A common evaluation range needs no extra transitions from any method/run.
    common_max=min(v.groupby('method').evaluations.max().min() for v in reps)
    common_max=int(common_max//72)*72
    grids={'seconds':np.linspace(meta['seconds']/100,meta['seconds'],100),
           'evaluations':np.arange(72,common_max+1,72)}
    rows=[]
    for axis,grid in grids.items():
        for method in METHODS:
            errors=np.full((len(grid),meta['chains']),np.nan)
            for i,frame in enumerate(reps):
                v=frame[frame.method==method].sort_values(axis)
                index=np.searchsorted(v[axis],grid,side='right')-1
                valid=index>=0
                errors[valid,i]=v.estimate.to_numpy()[index[valid]]
            np.savez_compressed(directory/f'{method.replace("/","_").replace(" ","_")}_{axis}_aligned.npz',grid=grid,errors=errors)
            for j,x in enumerate(grid):
                valid=np.isfinite(errors[j]).all()
                e=errors[j]
                rows.append(dict(axis=axis,budget=x,method=method,valid_replicates=int(np.isfinite(e).sum()),
                    mse=float(np.mean(e**2)) if valid else np.nan,
                    abs_bias=float(abs(e.mean())) if valid else np.nan,
                    variance=float(e.var(ddof=1)) if valid else np.nan,
                    mse_lower=float(np.quantile(e**2,.05)) if valid else np.nan,
                    mse_upper=float(np.quantile(e**2,.95)) if valid else np.nan))
    summary=pd.DataFrame(rows);summary.to_csv(directory/'summary.csv',index=False)
    good=summary.dropna(subset=['mse'])
    np.testing.assert_allclose(good.mse,good.abs_bias**2+(meta['chains']-1)/meta['chains']*good.variance,atol=1e-10,rtol=1e-12)
    colors=['blue','orange','red','black','#00BFC4','#7B2CBF']
    styles=['-',(0,(1,2)),'-','-','-','--']
    metrics=[('mse','MSE'),('abs_bias','Abs. bias'),('variance','Variance')]
    limits={'mse':(.01,2680.2539378000765),'abs_bias':(.0007506817884547799,9.881710918713189),
            'variance':(.2849540396749539,615.1923954671112)}
    plt.rcParams.update({'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm',
        'axes.formatter.use_mathtext':True,'axes.unicode_minus':False,'pdf.fonttype':42,
        'font.size':18,'axes.labelsize':20,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':13})
    def draw(ax,axis,metric,label):
        for method,color,style in zip(METHODS,colors,styles):
            v=summary[(summary.axis==axis)&(summary.method==method)]
            if metric=='mse':
                ax.fill_between(v.budget.to_numpy(),v.mse_lower.to_numpy(),v.mse_upper.to_numpy(),color=color,alpha=.1,linewidth=0)
            ax.plot(v.budget,v[metric],color=color,ls=style,lw=3,label=method)
        xmax=meta['seconds'] if axis=='seconds' else common_max
        ax.set(yscale='log',ylim=limits[metric],xlim=(0,xmax),ylabel=label,
               xlabel='Total summed runtime (sec)' if axis=='seconds' else 'Total target evaluations')
        ax.set_xticks([0,xmax/2,xmax]);ax.grid(alpha=.2)
    for axis in grids:
        fig,axes=plt.subplots(1,3,figsize=(18,5))
        for ax,(metric,label) in zip(axes,metrics):draw(ax,axis,metric,label)
        h,l=axes[0].get_legend_handles_labels()
        fig.legend(h,l,loc='upper center',ncol=6,frameon=False,bbox_to_anchor=(.5,1))
        fig.text(.5,.008,f'MSE shading: empirical 5th--95th percentiles of squared errors across {meta["chains"]} independent replicates.',ha='center',fontsize=12)
        fig.tight_layout(rect=[0,.045,1,.88])
        for ext in ['pdf','png']:fig.savefig(directory/f'ising40_ht_{axis}.{ext}',dpi=180)
        plt.close(fig)
    for metric,label in metrics:
        fig,ax=plt.subplots(figsize=(7,4.8));fig.subplots_adjust(left=.16,right=.97,bottom=.18,top=.97)
        draw(ax,'seconds',metric,label)
        ax.legend(loc='upper right',framealpha=.9,handlelength=1.5,handletextpad=.5,labelspacing=.2,borderpad=.3)
        for ext in ['pdf','png']:fig.savefig(directory/f'ising40_ht_{metric}_time.{ext}',dpi=180)
        plt.close(fig)
    diagnostic=pd.concat([pd.read_csv(directory/f'replicate_{i:02d}/ht_diagnostics.csv').assign(replicate=i)
                          for i in range(meta['chains'])],ignore_index=True)
    diagnostic.to_csv(directory/'ht_diagnostics.csv',index=False)
    print('Invalid HT checkpoints:',diagnostic.status.value_counts().to_dict(),flush=True)
    print(summary[(summary.axis=='seconds')&(summary.budget==meta['seconds'])].to_string(index=False),flush=True)
    print('Saved plots. Evaluation range from the same time runs:',common_max,flush=True)
