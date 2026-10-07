"""Shared model settings and figure styling for the submission experiments."""
from functools import lru_cache
from pathlib import Path
from datetime import datetime
import json
import random
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
TIME_METHODS = ['MCMC', 'MCMC RB/WR', 'MCMC OPAD', 'MCMC OPAD+', 'NWSS', 'Clyde-Ghosh HT']
KL_METHODS = ['DPVI', 'DPVI OPAD+', 'Clyde-Ghosh HT', *TIME_METHODS[:5]]
COLORS = dict(zip(TIME_METHODS, ['blue','orange','red','black','#00BFC4','#7B2CBF']))
COLORS.update({'DPVI':'green', 'DPVI OPAD+':'magenta'})
STYLES = {m:'-' for m in COLORS}
STYLES.update({'DPVI':'--','DPVI OPAD+':'--','Clyde-Ghosh HT':'--','MCMC RB/WR':(0,(1,2))})


def new_run(output, name):
    path = Path(output) / (name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_metadata(path, meta):
    (path/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')


@lru_cache(maxsize=None)
def make_model(name):
    if name == 'ising15':
        from om.models.ising1D import IsingModel1D
        return IsingModel1D(15, J=1, h=.1, beta=.5)
    if name in ['ising30', 'ising40']:
        from om.models.ising1D_high_dim import IsingModel1D_Large_Symmetric
        return IsingModel1D_Large_Symmetric(int(name[5:]), J=1, beta=.5)
    if name == 'bvs':
        from om.models.var_select import BayesianVarSelectModel, fetch_mince_nutrition_data, standardize_add_one_column
        x,y,names = fetch_mince_nutrition_data(str(ROOT/'om/models/lifespan-merged.csv'))
        x,_ = standardize_add_one_column(x,names)
        return BayesianVarSelectModel(x,y,a=3.,b=1.,c=len(x),pi=.5)
    if name == 'bsler':
        from om.models.bsl_er import BslErModel, generate_er_data
        data,_,_ = generate_er_data()
        return BslErModel(data)
    raise ValueError(f'Unknown model: {name}')


@lru_cache(maxsize=None)
def reference(name):
    model = make_model(name)
    if name == 'bsler':
        states,logp,logz = model.exact_posterior()
        logu = logp + logz
    else:
        states = np.asarray(model.generate_all_states())
        logu = np.array([-model.calc_neg_log_unnormalized_prob(s) for s in states])
        logz = logsumexp(logu)
        logp = logu-logz
    return states, {s.tobytes():i for i,s in enumerate(states)}, logu, logp, float(logz)


def sampler_and_initial(name, seed):
    model = make_model(name)
    np.random.seed(seed)
    if name == 'bsler':
        from om.samplers.structure_mcmc import StructureMCMC
        rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(2)[0])
        sampler = StructureMCMC(model,rng)
        return sampler,model.generate_init_state(rng)
    from om.samplers.mh import BiValueArrayMetropolisSampler, FixedFirstElementArraySampler
    sampler = (FixedFirstElementArraySampler if name=='bvs' else BiValueArrayMetropolisSampler)(model)
    return sampler,model.generate_init_state()


def make_search(name, seed):
    np.random.seed(seed); random.seed(seed)
    model = make_model(name)
    if name == 'bsler':
        from om.samplers.structure_mcmc import StructureMCMC
        from om.samplers.dag_nwss import DAGNWSS
        search = DAGNWSS(model, StructureMCMC(model))
        initial = model.generate_init_state(np.random.default_rng(seed))
    else:
        from om.samplers.greedy import GreedyExplore
        search = GreedyExplore(model, **({'possible_values':(0,1),'fixed_first_value':1} if name=='bvs' else {}))
        initial = model.generate_init_state()
    search.initialize('NWSS',initial)
    return search


def plot_style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.serif':['cmr10'],
        'mathtext.fontset':'cm','axes.formatter.use_mathtext':True,
        'axes.unicode_minus':False,'pdf.fonttype':42,'font.size':18,
        'axes.labelsize':20,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':13})
    return plt


def save_figure(fig, folder, stem):
    for ext in ['pdf','png']:
        fig.savefig(folder/(stem+'.'+ext), dpi=180)
