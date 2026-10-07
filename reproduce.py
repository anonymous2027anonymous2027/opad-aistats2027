#!/usr/bin/env python3
"""Reproduce the non-LSAC experiments in Optimal Reweighting for Discrete MCMC."""
import argparse
from pathlib import Path
import importlib
import json
import logging
import math
import os
import sys
import tempfile
from types import SimpleNamespace

# Set before importing numerical libraries; keep small-matrix workloads predictable.
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(name,'1')
logging.getLogger('fontTools.ttLib.tables._h_e_a_d').setLevel(logging.ERROR)
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'opad-matplotlib'))

MODELS={'kl':['ising15','bvs','bsler'],'tradeoff':['bvs','ising15','ising30','ising40'],
        'rhat':['ising15','ising30','ising40']}


def replot(folder):
    meta=json.loads((folder/'metadata.json').read_text())
    if meta['status']!='complete':raise ValueError('Plotting requires a complete run.')
    kind=meta['kind']
    if kind in ['kl','rhat']:
        importlib.import_module('om.experiments.'+kind).plot(folder)
    else:
        module=importlib.import_module(f'om.experiments.{meta["model"]}_ht_comparison')
        module.plot(folder)
        if meta['model']=='bvs':bvs_panels(folder)


def bvs_panels(folder):
    import numpy as np
    from om.experiments.common import plot_style,save_figure,TIME_METHODS,COLORS,STYLES
    from om.experiments.plot_bvs_ht_paper_style import LIMITS
    meta=json.loads((folder/'metadata.json').read_text());plt=plot_style()
    for metric,label in [('mse','MSE'),('abs_bias','Abs. bias'),('variance','Variance')]:
        fig,ax=plt.subplots(figsize=(7,4.8));fig.subplots_adjust(left=.16,right=.97,bottom=.18,top=.97)
        for method in TIME_METHODS:
            stem=method.replace('/','_').replace(' ','_')
            with np.load(folder/f'{stem}_seconds_aligned.npz') as z:
                grid=z['grid'];errors=z['estimates']-meta['ground_truth']
            valid=np.isfinite(errors).all(axis=1);values=np.full(len(grid),np.nan)
            e=errors[valid]
            if metric=='mse':
                values[valid]=np.mean(e**2,axis=1)
                lo=values.copy();hi=values.copy()
                if len(e):lo[valid],hi[valid]=np.quantile(e**2,[.05,.95],axis=1)
                ax.fill_between(grid,lo,hi,color=COLORS[method],alpha=.1,linewidth=0)
            elif metric=='abs_bias':values[valid]=abs(e.mean(axis=1))
            else:values[valid]=e.var(axis=1,ddof=1)
            ax.plot(grid,values,label=method,color=COLORS[method],ls=STYLES[method],lw=3)
        ax.set(yscale='log',ylim=LIMITS[metric],xlim=(0,meta['seconds']),ylabel=label,xlabel='Total summed runtime (sec)')
        ax.set_xticks([0,meta['seconds']/2,meta['seconds']]);ax.grid(alpha=.2)
        ax.legend(loc='upper right',framealpha=.9,handlelength=1.5,handletextpad=.5,labelspacing=.2,borderpad=.3)
        save_figure(fig,folder,'bvs_ht_'+metric+'_time');plt.close(fig)


def execute(kind,name,args):
    settings=SimpleNamespace(**vars(args))
    if kind=='tradeoff':
        settings.seconds=args.seconds or (24. if name=='ising40' else 12.)
        module=importlib.import_module(f'om.experiments.{name}_ht_comparison')
        folder=module.run(settings)
        meta=json.loads((folder/'metadata.json').read_text())
        meta.update(kind='tradeoff',model=name)
        (folder/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        module.plot(folder)
        if name=='bvs':bvs_panels(folder)
    else:
        folder=importlib.import_module('om.experiments.'+kind).run(name,settings)
    print('Results:',folder,flush=True)
    return folder


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('experiment',nargs='?',choices=['illustration','kl','tradeoff','rhat','all','plot'])
    parser.add_argument('run_directory',nargs='?',type=Path,help='Saved run directory for the plot command')
    parser.add_argument('--list',action='store_true',help='List the experiments and paper defaults')
    parser.add_argument('--model',default='all',choices=['all','bvs','ising15','ising30','ising40','bsler'])
    parser.add_argument('--chains',type=int,default=20,help='Independent replicates')
    parser.add_argument('--evaluations',type=int,default=20000,help='KL/evaluation budget; HT pays for both chains')
    parser.add_argument('--seconds',type=float,help='Override runtime per method; defaults to paper budgets')
    parser.add_argument('--particles',type=int,nargs='+',help='DPVI K values; default Ising15: 10 100, other KL models: 100')
    parser.add_argument('--records',type=int,default=100,help='Number of recorded KL/R-hat checkpoints')
    parser.add_argument('--output-dir',type=Path,default=Path('results'),help='Parent directory for new timestamped runs')
    parser.add_argument('--quick',action='store_true',help='Smoke test: 2 replicates, 400 evaluations, 0.5 seconds, 10 checkpoints')
    args=parser.parse_args(argv)
    if args.list:
        print('illustration : Figure 1, 101-state IS comparison\nkl          : Figure 2, Ising15 K=10/100, BVS K=100, BslEr K=100; 20,000 evaluations\ntradeoff    : Figure 3, BVS/Ising15/Ising30 12 s, Ising40 24 s; 20 replicates\nrhat        : Appendix D, Ising15/30/40, 12 s; 20 replicates\nall         : Run all of the above\nplot DIR    : Regenerate PDFs/PNGs from a saved run without resampling')
        return
    if args.experiment is None:parser.error('Choose an experiment, or use --list.')
    if args.experiment=='plot':
        if args.run_directory is None:parser.error('plot requires a saved run directory.')
        replot(args.run_directory.resolve());return
    if args.run_directory is not None:parser.error('A positional run directory is used only with plot.')
    if args.quick:
        args.chains=2;args.evaluations=400;args.seconds=.5;args.records=10
    if args.chains<2 or args.evaluations<90 or args.records<2 or (args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds<=0)):
        parser.error('Require at least 2 chains, 90 evaluations, 2 checkpoints and positive runtime.')
    if args.records>args.evaluations:parser.error('The number of checkpoints cannot exceed evaluations.')
    if args.particles and any(k<1 or k>args.evaluations for k in args.particles):
        parser.error('Each particle count must lie between 1 and the evaluation budget.')
    if args.experiment in ['illustration','all']:
        from om.experiments.common import new_run
        from om.experiments.is_vs_opad_from_q_draws import main as illustration
        output=new_run(args.output_dir,'illustration')
        illustration(['--output-dir',str(output),'--repetitions','2' if args.quick else '100'])
        if args.experiment=='illustration':return
    for kind in (['kl','tradeoff','rhat'] if args.experiment=='all' else [args.experiment]):
        if args.model=='all':models=MODELS[kind]
        elif args.model in MODELS[kind]:models=[args.model]
        elif args.experiment=='all':continue
        else:parser.error(f'{kind} supports: '+', '.join(MODELS[kind]))
        for name in models:execute(kind,name,args)


if __name__=='__main__':
    main()
