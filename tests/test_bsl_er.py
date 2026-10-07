import unittest
import numpy as np
from scipy.special import multigammaln
from om.models.bsl_er import BslErModel,generate_er_data
from om.samplers.structure_mcmc import StructureMCMC

class BslErTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.coefficients, _ = generate_er_data(num_nodes=3, expected_degree=1, data_seed=11)
        cls.model = BslErModel(cls.data)

    def test_local_bge_against_joint_normal_wishart_marginals(self):
        # Independent determinant/multivariate-gamma calculation of
        # log p(D_{parents + child}) - log p(D_parents).
        model = self.model
        n, d = model.data.shape

        def log_marginal(nodes):
            r = len(nodes)
            if r == 0:
                return 0.0
            nu = model.alpha_w - d + r
            _, logdet = np.linalg.slogdet(model.posterior_scale[np.ix_(nodes, nodes)])
            return (-n * r / 2 * np.log(np.pi)
                    + r / 2 * np.log(model.alpha_mu / (model.alpha_mu + n))
                    + multigammaln((nu + n) / 2, r) - multigammaln(nu / 2, r)
                    + nu / 2 * r * np.log(model.prior_scale)
                    - (nu + n) / 2 * logdet)

        for child in range(d):
            for mask in range(1 << d):
                if mask & (1 << child):
                    continue
                parents = [i for i in range(d) if mask & (1 << i)]
                expected = log_marginal(parents + [child]) - log_marginal(parents)
                self.assertAlmostEqual(model.local_scores[child, mask], expected, places=10)

    def test_score_equivalence(self):
        scores = []
        for edges in [[(0, 1), (1, 2)], [(1, 0), (1, 2)], [(1, 0), (2, 1)]]:
            state = np.array([edge in edges for edge in self.model.edges], dtype=np.uint8)
            scores.append(self.model.calc_neg_log_unnormalized_prob(state))
        np.testing.assert_allclose(scores, scores[0], atol=1e-10)

    def test_structure_mcmc_transition_matrix(self):
        model = self.model
        states, log_p, _ = model.exact_posterior()
        indices = {state.tobytes(): i for i, state in enumerate(states)}
        sampler = StructureMCMC(model)
        transition = np.zeros((len(states), len(states)))
        for i, state in enumerate(states):
            neighbors = sampler.neighbors(state)
            for neighbor in neighbors:
                j = indices[neighbor.tobytes()]
                self.assertIn(state.tobytes(), {x.tobytes() for x in sampler.neighbors(neighbor)})
                alpha = sampler.acceptance_probability(state, neighbor)
                self.assertTrue(0 <= alpha <= 1)
                transition[i, j] += alpha / len(neighbors)
                transition[i, i] += (1 - alpha) / len(neighbors)
        p = np.exp(log_p)
        flux = p[:, None] * transition
        np.testing.assert_allclose(transition.sum(axis=1), 1, atol=1e-14)
        np.testing.assert_allclose(flux, flux.T, rtol=1e-11, atol=1e-15)
        np.testing.assert_allclose(p @ transition, p, rtol=1e-11, atol=1e-15)
        self.assertTrue((transition.diagonal() > 0).all())
