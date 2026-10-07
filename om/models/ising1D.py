import numpy as np
import itertools
from om.models.model import DiscreteDistributionModel


class IsingModel1D(DiscreteDistributionModel):
    """
    Assuming that the interaction strength is constant and by default there is no external magnetic field
    """

    def __init__(self, N, J=1, h=0.0, beta=1.0):
        self.N = N  # Size of the 1D lattice (N,)
        self.J = J  # Interaction strength
        self.h = h  # external magnetic field (assuming that it is fixed)
        self.beta = beta  # Inverse temperature

        _, unnormalized_probs = self.calc_all_states_and_unnormalized_probs()
        self.Z = sum(unnormalized_probs)

    def get_dimension(self):
        return self.N

    def calc_normalization_factor(self):
        return self.Z

    def calc_neg_log_unnormalized_prob(self, x):
        """
        Calculate the energy of a given an N-vector configuration, x (i.e. x is sigma).
        where H(sigma) = -sum_{i in {1..N}} (J * [ sigma_{i} sigma_{i+1}])
                         -sum_{i in {1..N}} mu * (h * sigma_{i})  # assuming mu=1 and h=h_{i} is fixed
        """
        mu = 1

        assert x.shape == (self.N,)
        energy = 0

        for i in range(self.N):
            s_i = x[i]
            assert s_i in {-1, 1}
            s_right = x[(i + 1) % self.N]
            energy -= self.J * s_i * s_right  # interaction term
            energy -= mu * self.h * s_i

        energy *= self.beta
        return energy

    def generate_init_state(self):
        state = np.random.choice([-1, 1], size=self.N)
        return state

    def generate_all_states(self):
        """Generate all possible configurations."""
        all_states = itertools.product([-1, 1], repeat=self.N)
        return [np.reshape(np.array(state), (self.N,)) for state in all_states]
