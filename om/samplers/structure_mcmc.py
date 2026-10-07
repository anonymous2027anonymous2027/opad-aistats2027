"""Structure MCMC and coordinate DPVI restricted to valid DAGs."""

import numpy as np

from om.tools.opad import OPADDistribution


class StructureMCMC:
    """Uniform proposal over valid add/delete/reverse neighbors plus a self move.

    The neighborhood size depends on the graph, so MH includes |N(x)|/|N(y)|.
    The self move ensures aperiodicity; deletions and additions ensure irreducibility.
    """

    def __init__(self, model, rng=None):
        self.model = model
        self.rng = np.random.default_rng() if rng is None else rng
        self._neighbors = {}
        self._reverse = {k: model.edges.index((j, i))
                         for k, (i, j) in enumerate(model.edges)}

    def neighbors(self, state):
        state = np.asarray(state, dtype=np.uint8)
        key = state.tobytes()
        if key not in self._neighbors:
            if not self.model.is_valid_state(state):
                raise ValueError("Structure MCMC requires a DAG.")
            candidates = {key: state.copy()}
            for k in range(self.model.dimension):
                candidate = state.copy()
                candidate[k] ^= 1
                if self.model.is_valid_state(candidate):
                    candidates[candidate.tobytes()] = candidate
                if state[k]:
                    candidate = state.copy()
                    candidate[k] = 0
                    candidate[self._reverse[k]] = 1
                    if self.model.is_valid_state(candidate):
                        candidates[candidate.tobytes()] = candidate
            neighbors = np.array(list(candidates.values()), dtype=np.uint8)
            neighbors.flags.writeable = False
            self._neighbors[key] = neighbors
        return self._neighbors[key]

    def acceptance_probability(self, current, proposed):
        log_ratio = (self.model.calc_neg_log_unnormalized_prob(current)
                     - self.model.calc_neg_log_unnormalized_prob(proposed)
                     + np.log(len(self.neighbors(current)))
                     - np.log(len(self.neighbors(proposed))))
        return float(np.exp(min(0.0, log_ratio)))

    def next_sample_proposals_acceptances(self, current_state):
        neighbors = self.neighbors(current_state)
        proposed = neighbors[self.rng.integers(len(neighbors))].copy()
        alpha = self.acceptance_probability(current_state, proposed)
        current = proposed if self.rng.random() < alpha else current_state.copy()
        return current, [proposed], [alpha]


class DAGParticleVariationalInference:
    """Binary coordinate DPVI: retain K best unique valid DAG candidates.

    One update tries one edge toggle per current particle. Invalid toggles are
    excluded; original particles remain candidates. OPAD+ can retain every
    valid candidate, including those DPVI discards.
    """

    def __init__(self, model, num_particles=100, rng=None):
        if not isinstance(num_particles, int) or num_particles < 1:
            raise ValueError("num_particles must be a positive integer.")
        counts = {2: 3, 3: 25, 4: 543, 5: 29281}
        if num_particles > counts[model.num_nodes]:
            raise ValueError("More particles requested than distinct DAGs.")
        self.model = model
        self.num_particles = num_particles
        self.rng = np.random.default_rng() if rng is None else rng
        self.index_to_flip = 0
        self.particles_scores = OPADDistribution(model)
        while self.particles_scores.num_entries() < num_particles:
            self.particles_scores.add_array(model.generate_init_state(self.rng))

    def evolve_all_states_in_one_dim(self):
        candidates = OPADDistribution(self.model)
        for particle in self.particles_scores.generate_all_states():
            candidates.add_array(particle)
            candidate = particle.copy()
            candidate[self.index_to_flip] ^= 1
            if self.model.is_valid_state(candidate):
                candidates.add_array(candidate)
        self.particles_scores = OPADDistribution(self.model)
        for particle in candidates.fetch_K_top_states(self.num_particles):
            self.particles_scores.add_array(particle)
        self.index_to_flip = (self.index_to_flip + 1) % self.model.dimension
        return candidates
