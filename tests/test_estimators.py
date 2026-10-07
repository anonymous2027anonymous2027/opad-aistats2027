import unittest
from types import SimpleNamespace
import numpy as np
from om.experiments.kl import mc_records,ht_records,dpvi_records
from om.experiments.rhat import generalized_rhat
from om.experiments.common import make_model,reference
from om.models.bsl_er import enumerate_dags
from om.experiments.is_vs_opad_from_q_draws import symmetric_proposal, mh_sample,compare_distributions


class EstimatorTests(unittest.TestCase):
    def test_mc_wr_opad_and_enlargement(self):
        u=np.array([.2,.3,.5]);lp=np.log(u)
        p=dict(initial=0,states=np.array([0,0,1]),proposals=np.array([1,2,1]),alpha=np.array([.2,.4,1.]))
        rows=mc_records(p,lp,lp,[1,3]);last={r['method']:r['kl'] for r in rows if r['evaluations']==3}
        q=np.array([2/3,1/3,0.]);keep=q>0
        self.assertAlmostEqual(last['MCMC'],q[keep]@(np.log(q[keep])-lp[keep]))
        self.assertAlmostEqual(last['MCMC OPAD'],-np.log(.5))
        self.assertAlmostEqual(last['MCMC OPAD+'],0.)
        rb=np.array([1.4,1.2,.4])/3
        self.assertAlmostEqual(last['MCMC RB/WR'],rb@(np.log(rb)-lp))

    def test_ht_budget_accounts_for_both_chains(self):
        p={'states':np.tile(np.arange(3),80)};a={'states':np.tile(np.arange(3),10)}
        rows,diag=ht_records(p,a,np.log([.2,.3,.5]),np.log([.2,.3,.5]),180,10)
        for row in diag:
            self.assertEqual(row['evaluations'],9*row['r'])
            self.assertEqual(row['evaluations'],row['primary_evaluations']+row['auxiliary_evaluations'])
        self.assertEqual(rows[-1]['evaluations'],180)

    def test_rhat_matches_definition(self):
        means=np.array([[1.,2.,3.],[4.,4.,4.]])
        variances=np.ones_like(means)*2
        np.testing.assert_allclose(generalized_rhat(means,variances),[np.sqrt(1.5),1.])
        self.assertTrue(np.isnan(generalized_rhat(means,np.zeros_like(means))).all())

    def test_dag_enumeration(self):
        self.assertEqual(len(enumerate_dags(3)),25)
        self.assertEqual(len(enumerate_dags(5)),29281)

    def test_nwss_adds_unique_states_with_target_weights(self):
        from om.experiments.common import make_search
        search=make_search('ising15',7)
        model=make_model('ising15')
        for _ in range(20):
            before=search.inner_opad.num_entries()
            search.evolve()
            self.assertEqual(search.inner_opad.num_entries(),before+1)
        for state in search.inner_opad.generate_all_states():
            self.assertAlmostEqual(search.inner_opad.calc_unnormalized_prob(state),model.calc_unnormalized_prob(state))

    def test_full_dpvi_budget_and_support_optimality(self):
        _,ids,_,lp,_=reference('ising15')
        rows=dpvi_records('ising15',0,10,200,ids,lp)
        dv=[r['kl'] for r in rows if r['method']=='DPVI']
        plus=[r['kl'] for r in rows if r['method']=='DPVI OPAD+']
        self.assertEqual(rows[-1]['evaluations'],200)
        self.assertTrue(np.all(np.diff(dv)<=1e-10))
        self.assertTrue(np.all(np.array(plus)<=np.array(dv)+1e-10))

    def test_illustration_matches_submission(self):
        x=np.arange(101)
        p=np.exp(-.5*((x-25)/6)**2)+np.exp(-.5*((x-75)/6)**2);p/=p.sum()
        q=np.sqrt(p);q/=q.sum()
        samples,_=mh_sample(q,2300,start=5,burn_in=500,seed=7)
        _,_,metrics=compare_distributions(samples,p,q)
        self.assertAlmostEqual(metrics['opad_kl'],.120,delta=.001)
        self.assertAlmostEqual(metrics['is_kl'],.512,delta=.001)


if __name__=='__main__':unittest.main()
