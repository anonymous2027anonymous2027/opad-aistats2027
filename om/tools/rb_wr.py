import numpy as np

from om.models.model import DiscreteDistributionModel
import heapq

"""
RB/WR: Rao-Blackwell / Waste-Recycling
The effective weight of (unique) sate x is:
w_{RB/WR} = 1/N (sum_{t:x_t = x} (1 - a_t) + \sum_{t:x'_t = x} a_t)
where:
    x_t = current state at step t
    x'_t = proposed state at step t
    a_t = acceptance probability of transiting from x_t to x'_t
If the current state of an MCMC chain at step t is x_t
References:
    1. Douc, Randal, and Christian P. Robert. "A vanilla rao–blackwellization of metropolis–hastings algorithms." (2011): 261-277.
    2. Frenkel, D. "Waste-recycling monte carlo." Computer Simulations in Condensed Matter Systems: From Materials to Chemical Biology Volume 1. Berlin, Heidelberg: Springer Berlin Heidelberg, 2006. 127-137.
"""


class RB_WR_Distrib(DiscreteDistributionModel):
    def __init__(self):
        self.state_to_weight = dict()
        self.shape = None
        self.dtype = None
        self.N = 0  # total number of visited states in the MCMC chain (not in the RB/WR empirical distribution that also includes rejected proposals

    def add_previous_and_proposed_states(self, previous_state, proposed_state, acceptance_prob):
        """
        Adds x_t and x'_t the RB/WR support and caches its unnormalized weight.

        :param previous_state: numpy array, x_t
        :param proposed_state: numpy array, x'_t
        :param acceptance_prob: acceptance probability a(x_t -> x'_t)
        """
        if self.shape is None:
            self.shape = previous_state.shape
            self.dtype = previous_state.dtype
        else:
            assert 0 <= acceptance_prob <= 1

        self._add_weighted_state(state=previous_state, weight=1 - acceptance_prob)
        self._add_weighted_state(state=proposed_state, weight=acceptance_prob)
        self.N +=1

    def _add_weighted_state(self, state, weight):
        array_bytes = state.tobytes()
        if array_bytes not in self.state_to_weight:
            self.state_to_weight[array_bytes] = weight
        else:
            self.state_to_weight[array_bytes] += weight

    def calc_normalization_factor(self):
        return self.N

    def calc_unnormalized_prob(self, array):
        """
        Returns the cached unnormalized target weight for a visited state.

        Args:
        - array (np.ndarray): The numpy array to query in the histogram.

        Returns:
        - weight (float): The cached unnormalized weight of the state.
        """
        array_bytes = array.tobytes()
        return self.state_to_weight[array_bytes]

    def calc_neg_log_unnormalized_prob(self, x):
        return -np.log(self.calc_unnormalized_prob(x))

    def generate_all_states(self):
        """
        Gets all unique arrays currently in the histogram.

        Returns:
        - unique_arrays (List[np.ndarray]): A list of unique numpy arrays in the histogram.
        """
        return [np.frombuffer(key, dtype=self.dtype).reshape(self.shape) for key in self.state_to_weight.keys()]

    def fetch_K_top_states(self, K: int):
        top_k_keys = heapq.nlargest(K, self.state_to_weight, key=self.state_to_weight.get)

        return [np.frombuffer(key, dtype=self.dtype).reshape(self.shape) for key in top_k_keys]

    def num_entries(self):
        return len(self.state_to_weight)

    def arraybyte_to_numpy_array(self, array_byte):
        return np.frombuffer(array_byte, dtype=self.dtype).reshape(self.shape)

    def __repr__(self):
        result = "RB_WR_Distribution:\n"
        for key, weight in self.state_to_weight.items():
            array = np.frombuffer(key, dtype=self.dtype).reshape(self.shape)
            result += f"Array:\t{array}\t {weight}\n"
        return result
