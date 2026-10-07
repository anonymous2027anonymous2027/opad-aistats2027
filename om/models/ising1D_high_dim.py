import numpy as np
import itertools
from om.models.model import DiscreteDistributionModel


class IsingModel1D_Large_Symmetric(DiscreteDistributionModel):
    """
    Assuming that the interaction strength is constant and by default there is no external magnetic field
    """

    def __init__(self, N, J=1, beta=1.0, generating_all_states_is_allowed=False):
        self.N = N  # Size of the 1D lattice (N,)
        self.J = J  # Interaction strength
        self.h = 0  # external magnetic field (assuming that it is 0 to make it symmetric)
        self.beta = beta  # Inverse temperature
        self.generating_all_states_is_allowed = generating_all_states_is_allowed


    def get_dimension(self):
        return self.N

    def calc_normalization_factor(self):
        raise Exception('Not implemented for this high dimensional model')

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
            s_left = x[(i - 1) % self.N]
            energy -= self.J * s_i * s_right  # interaction term
            energy -= self.J * s_i * s_left  # interaction term
            energy -= mu * self.h * s_i

        energy *= self.beta
        return energy

    def generate_init_state(self):
        state = np.random.choice([-1, 1], size=self.N)
        return state

    def generate_all_states(self):
        if self.generating_all_states_is_allowed:
            """Generate all possible configurations -- can be computationally demanding for high dim models"""
            all_states = itertools.product([-1, 1], repeat=self.N)
            return [np.reshape(np.array(state), (self.N,)) for state in all_states]
        else:
            raise Exception('Generating all answered is not allowed')
