"""Replot saved BVS/HT results using the paper's uncertainty-band convention.

MSE shading is the empirical 5th--95th percentile range of individual-run
squared errors, NOT a confidence interval for their mean. Bias and variance
have no shading. This script only reads saved replicate estimates.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    ('MCMC','blue','-',3),
    ('MCMC RB/WR','orange',(0,(1,2)),4),
    ('MCMC OPAD','red','-',2),
    ('MCMC OPAD+','black','-',2),
    ('NWSS','#00BFC4','-',2),
    ('Clyde-Ghosh HT','#7B2CBF','--',3),
]
LIMITS = {'mse':(1.1713734478086446e-9,.041285760450045374),
          'abs_bias':(2.9190893492049618e-5,.18926646232317967),
          'variance':(2.988239953097035e-9,.005955655556673623)}


def main(folder):
    meta=json.loads((folder/'metadata.json').read_text())
    plt.rcParams.update({'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm',
        'axes.formatter.use_mathtext':True,'axes.unicode_minus':False,'pdf.fonttype':42,
        'font.size':15,'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13,'legend.fontsize':12})
    fig,axes=plt.subplots(3,2,figsize=(11.5,10.5))
    bounds={}
    for col,axis in enumerate(['evaluations','seconds']):
        for method,color,style,zorder in METHODS:
            stem=method.replace('/','_').replace(' ','_')
            with np.load(folder/f'{stem}_{axis}_aligned.npz') as data:
                x=data['grid'].copy();error=data['estimates']-meta['ground_truth']
            valid=np.isfinite(error).all(axis=1)
            mse=np.full(len(x),np.nan);bias=mse.copy();variance=mse.copy()
            low=mse.copy();high=mse.copy()
            mse[valid]=(error[valid]**2).mean(axis=1)
            bias[valid]=abs(error[valid].mean(axis=1))
            variance[valid]=error[valid].var(axis=1,ddof=1)
            low[valid],high[valid]=np.quantile(error[valid]**2,[.05,.95],axis=1)
            bounds[f'{stem}_{axis}_grid']=x
            bounds[f'{stem}_{axis}_mse_lower']=low
            bounds[f'{stem}_{axis}_mse_upper']=high
            axes[0,col].fill_between(x,low,high,color=color,alpha=.10,linewidth=0,zorder=1)
            for row,y in enumerate([mse,bias,variance]):
                axes[row,col].plot(x,y,color=color,linestyle=style,linewidth=3.,
                                   dash_capstyle='round',zorder=zorder,label=method)
        xmax=meta['evaluations'] if axis=='evaluations' else meta['seconds']
        for row,(metric,label) in enumerate([('mse','MSE'),('abs_bias','Abs. bias'),('variance','Variance')]):
            ax=axes[row,col]
            ax.set(yscale='log',ylim=LIMITS[metric],ylabel=label,xlim=(0,xmax))
            ax.set_xticks([0,xmax/2,xmax]);ax.grid(alpha=.2)
            if row==2:ax.set_xlabel('Total target evaluations' if axis=='evaluations' else 'Total summed runtime (sec)')
    axes[0,0].set_title('Target-evaluation budget');axes[0,1].set_title('Runtime budget')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,1),frameon=False)
    fig.text(.5,.012,f'MSE shading: empirical 5th--95th percentiles across {meta["completed_chains"]} independent replicates.',
             ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.03,1,.93])
    fig.savefig(folder/'bvs_ht_comparison_paper_style.pdf')
    fig.savefig(folder/'bvs_ht_comparison_paper_style.png',dpi=180)
    plt.close(fig)
    np.savez_compressed(folder/'paper_style_mse_bands.npz',**bounds)
    (folder/'paper_style_plot.json').write_text(json.dumps(dict(
        source='Saved per-replicate estimates.',
        mse_band='Empirical 5th--95th percentiles of individual-run squared errors, matching the current paper figure.',
        abs_bias_band=None,variance_band=None,replicates=meta['completed_chains'],
        invalid='All original replicates must be valid; undefined HT checkpoints remain missing.',
        y_limits=LIMITS),indent=2)+'\n')
    print('Saved paper-style plot:',folder/'bvs_ht_comparison_paper_style.pdf')
