"""Small Bayesian DAG model with a BGe score and an exact reference posterior.

States are uint8 vectors of off-diagonal adjacency entries in row-major order:
entry (i, j) represents i -> j. The graph prior is uniform over labeled DAGs.
"""

from functools import lru_cache
from itertools import permutations

import numpy as np
from scipy.special import gammaln, logsumexp

from om.models.model import DiscreteDistributionModel


def edge_order(num_nodes):
    return tuple((i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j)


def is_acyclic(adjacency):
    """Kahn's algorithm; does not require an enumerated graph space."""
    indegree = adjacency.sum(axis=0).astype(int)
    ready = list(np.flatnonzero(indegree == 0))
    visited = 0
    while ready:
        node = ready.pop()
        visited += 1
        for child in np.flatnonzero(adjacency[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(int(child))
    return visited == len(adjacency)


@lru_cache(maxsize=4)
def enumerate_dags(num_nodes):
    """Enumerate once per graph, not once per compatible topological order."""
    if not 2 <= num_nodes <= 5:
        raise ValueError("Exact enumeration is restricted to 2--5 nodes.")
    edges = edge_order(num_nodes)
    positions = {edge: k for k, edge in enumerate(edges)}
    masks = set()
    for order in permutations(range(num_nodes)):
        bits = [1 << positions[order[i], order[j]]
                for i in range(num_nodes) for j in range(i + 1, num_nodes)]
        subsets = [0]
        for bit in bits:
            subsets += [mask | bit for mask in subsets]
        masks.update(subsets)
    masks = np.array(sorted(masks), dtype=np.uint32)
    states = ((masks[:, None] >> np.arange(len(edges))) & 1).astype(np.uint8)
    states.flags.writeable = False
    return states


def generate_er_data(num_nodes=5, num_observations=200, expected_degree=2.0,
                     coefficient_min=0.5, coefficient_max=2.5, noise_sd=1.0,
                     data_seed=0):
    """Random-order ER DAG and X_j = sum_i W_ij X_i + independent N(0,sd²).

    expected_degree is the expected TOTAL (in + out) degree. Thus each
    unordered pair has an edge with probability expected_degree/(d-1).
    """
    if num_nodes < 2 or num_observations < 2:
        raise ValueError("Need at least two nodes and two observations.")
    if not 0 <= expected_degree <= num_nodes - 1:
        raise ValueError("Expected degree must lie in [0, num_nodes-1].")
    if not 0 < coefficient_min <= coefficient_max or noise_sd <= 0:
        raise ValueError("Coefficient magnitudes and noise_sd must be positive.")
    rng = np.random.default_rng(data_seed)
    order = rng.permutation(num_nodes)
    coefficients = np.zeros((num_nodes, num_nodes))
    for a in range(num_nodes):
        for b in range(a + 1, num_nodes):
            if rng.random() < expected_degree / (num_nodes - 1):
                magnitude = rng.uniform(coefficient_min, coefficient_max)
                coefficients[order[a], order[b]] = rng.choice([-1, 1]) * magnitude
    data = np.zeros((num_observations, num_nodes))
    noise = rng.normal(scale=noise_sd, size=data.shape)
    for child in order:
        data[:, child] = data @ coefficients[:, child] + noise[:, child]
    return data, coefficients, order


class BslErModel(DiscreteDistributionModel):
    """BGe posterior using a compatible normal--Wishart prior.

    Prior mean is zero, mean precision alpha_mu=1, Wishart degrees of freedom
    alpha_w=d+2, and scale T=t I with t=alpha_mu*(alpha_w-d-1)/(alpha_mu+1).
    Scores use the supplied data without centering or standardizing it first.
    Local scores are cached; exact enumeration is only for normalization/KL.
    """

    def __init__(self, data, alpha_mu=1.0, alpha_w=None):
        self.data = np.asarray(data, dtype=float).copy()
        if self.data.ndim != 2 or not np.isfinite(self.data).all():
            raise ValueError("Data must be a finite observations-by-nodes matrix.")
        n, self.num_nodes = self.data.shape
        d = self.num_nodes
        if n < 2 or not 2 <= d <= 5:
            raise ValueError("Need >=2 observations and 2--5 nodes.")
        self.alpha_mu = float(alpha_mu)
        self.alpha_w = float(d + 2 if alpha_w is None else alpha_w)
        if self.alpha_mu <= 0 or self.alpha_w <= d + 1:
            raise ValueError("Require alpha_mu > 0 and alpha_w > d+1.")
        self.edges = edge_order(d)
        self.rows, self.cols = np.array(self.edges).T
        self.dimension = len(self.edges)
        self.prior_scale = self.alpha_mu * (self.alpha_w - d - 1) / (self.alpha_mu + 1)
        mean = self.data.mean(axis=0)
        centered = self.data - mean
        self.posterior_scale = (self.prior_scale * np.eye(d) + centered.T @ centered
                                + self.alpha_mu * n / (self.alpha_mu + n) * np.outer(mean, mean))
        self.local_scores = np.full((d, 1 << d), -np.inf)
        for child in range(d):
            for mask in range(1 << d):
                if mask & (1 << child):
                    continue
                parents = [i for i in range(d) if mask & (1 << i)]
                p = len(parents)
                nu = self.alpha_w - d + p + 1
                constant = (-n / 2 * np.log(np.pi)
                            + 0.5 * np.log(self.alpha_mu / (self.alpha_mu + n))
                            + gammaln((nu + n) / 2) - gammaln(nu / 2)
                            + (nu + p) / 2 * np.log(self.prior_scale))
                residual = self.posterior_scale[child, child]
                logdet = 0.0
                if parents:
                    chol = np.linalg.cholesky(self.posterior_scale[np.ix_(parents, parents)])
                    v = np.linalg.solve(chol, self.posterior_scale[parents, child])
                    residual -= v @ v
                    logdet = 2 * np.log(np.diag(chol)).sum()
                if residual <= 0:
                    raise ValueError("Nonpositive BGe Schur complement; check data conditioning.")
                self.local_scores[child, mask] = constant - (nu + n) / 2 * np.log(residual) - logdet / 2
        self._exact = None
        self.exact_posterior()

    def get_dimension(self):
        return self.dimension

    def adjacency(self, state):
        state = np.asarray(state)
        if state.shape != (self.dimension,) or not np.isin(state, [0, 1]).all():
            raise ValueError("State must be a binary off-diagonal adjacency vector.")
        adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)
        adjacency[self.rows, self.cols] = state
        return adjacency

    def is_valid_state(self, state):
        try:
            return is_acyclic(self.adjacency(state))
        except ValueError:
            return False

    def generate_init_state(self, rng=None):
        """Random-order random DAG, independent of the exact posterior."""
        rng = np.random.default_rng() if rng is None else rng
        order = rng.permutation(self.num_nodes)
        adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                adjacency[order[i], order[j]] = rng.integers(2)
        return adjacency[self.rows, self.cols]

    def calc_neg_log_unnormalized_prob(self, state):
        adjacency = self.adjacency(state)
        if not is_acyclic(adjacency):
            return np.inf
        masks = (adjacency * (1 << np.arange(self.num_nodes))[:, None]).sum(axis=0)
        return self.log_weight_offset - self.local_scores[np.arange(self.num_nodes), masks].sum()

    def calc_unnormalized_prob(self, state):
        log_weight = -self.calc_neg_log_unnormalized_prob(state)
        weight = np.exp(np.longdouble(log_weight))
        if weight == 0 and np.isfinite(log_weight):
            raise FloatingPointError("BGe weight underflow: use log-space weighting for this dataset.")
        return weight

    def generate_all_states(self):
        return enumerate_dags(self.num_nodes)

    def exact_posterior(self):
        if self._exact is None:
            states = self.generate_all_states()
            scores = np.zeros(len(states))
            for child in range(self.num_nodes):
                indices = np.flatnonzero(self.cols == child)
                masks = (states[:, indices] * (1 << self.rows[indices])).sum(axis=1)
                scores += self.local_scores[child, masks]
            span = float(np.ptp(scores))
            self.log_weight_offset = float(scores.max()) - min(600.0, max(0.0, span - 700.0))
            log_weights = scores - self.log_weight_offset
            if np.exp(np.longdouble(log_weights.min())) == 0:
                raise FloatingPointError("BGe score range exceeds the available weight precision.")
            log_z = float(logsumexp(log_weights))
            self._exact = (states, log_weights - log_z, log_z)
        return self._exact

    def calc_normalization_factor(self):
        return np.exp(np.longdouble(self.exact_posterior()[2]))
