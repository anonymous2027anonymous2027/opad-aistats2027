import unittest
import numpy as np
from om.tools.ratio_ht import RatioHT, ratio_weights, inclusion_probabilities


class RatioHTTests(unittest.TestCase):
    def test_primary_multiplicity_and_retrospective_overlap(self):
        u, f = np.array([.2, .3, .5]), np.array([0., 1., 2.])
        h = RatioHT(u, f)
        primary, auxiliary = [0, 0, 1, 2], [2, 2, 0, 1]
        for r in range(1, 5):
            h.update(primary[r-1:r], auxiliary[r-1:r])
            a = set(auxiliary[:r])
            overlap = sum(x in a for x in primary[:r])
            out = h.estimate()
            self.assertEqual(out['overlap_count'], overlap)
            self.assertAlmostEqual(out['C_hat'], overlap / r / u[list(a)].sum())
            if overlap:
                ids = out['ids']
                pi = 1 - (1-out['C_hat']*u[ids])**r
                raw = u[ids]/pi
                self.assertAlmostEqual(out['estimate'], raw @ f[ids]/raw.sum())

    def test_invalid_is_reported_without_clipping(self):
        h = RatioHT([.9, .1], [0, 1])
        h.update([0], [1])
        self.assertEqual(h.estimate()['status'], 'zero_overlap')
        h.update([1], [1])  # C_hat=5; p_hat(state 0)=4.5
        out = h.estimate()
        self.assertEqual(out['status'], 'invalid_model_probability')
        self.assertGreater(out['p_hat'].max(), 1)
        self.assertTrue(np.isnan(out['estimate']))

    def test_stable_small_probabilities_and_limit(self):
        self.assertAlmostEqual(inclusion_probabilities([1e-20], 1000)[0]/1e-17, 1.)
        u = np.array([.2, .3, .5])
        w, pi = ratio_weights(u, u, 10000)
        np.testing.assert_allclose(pi, 1, atol=1e-14)
        np.testing.assert_allclose(w, u/u.sum(), atol=1e-14)
        self.assertEqual(inclusion_probabilities([1.], 2)[0], 1.)

    def test_exact_ht_totals_under_small_iid_design(self):
        # Enumerate all length-two sampling sequences. HT totals are unbiased;
        # this test does not assert that their ratio is exactly unbiased.
        import itertools
        u, q, f = np.array([2., 3., 5.]), np.array([.2, .3, .5]), np.array([1., 4., 2.])
        pi = inclusion_probabilities(q, 2)
        expected_num = expected_den = 0.
        for seq in itertools.product(range(3), repeat=2):
            ids = list(set(seq)); prob = q[list(seq)].prod()
            expected_num += prob * np.sum(f[ids]*u[ids]/pi[ids])
            expected_den += prob * np.sum(u[ids]/pi[ids])
        self.assertAlmostEqual(expected_num, u @ f)
        self.assertAlmostEqual(expected_den, u.sum())


if __name__ == '__main__':
    unittest.main()
