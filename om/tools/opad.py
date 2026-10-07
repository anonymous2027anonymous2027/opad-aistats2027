import numpy as np

from om.models.model import DiscreteDistributionModel
import heapq



class OPADDistribution(DiscreteDistributionModel):
    def __init__(self, target_model: DiscreteDistributionModel):
        self.target_model = target_model
        self.state_to_weight = dict()
        self.shape = None
        self.dtype = None
        self.total_weight = 0

    def add_array(self, array):
        """
        Adds a unique state to the OPAD support and caches its unnormalized weight.

        Args:
        - array (np.ndarray): The numpy array to add to the histogram.
        """
        if self.shape is None:
            self.shape = array.shape
            self.dtype = array.dtype
        else:
            assert array.shape == self.shape
            assert array.dtype == self.dtype

        array_bytes = array.tobytes()
        if array_bytes not in self.state_to_weight:
            weight = self.target_model.calc_unnormalized_prob(array)
            assert weight > 0
            self.state_to_weight[array_bytes] = weight
            self.total_weight += weight

    def calc_normalization_factor(self):
        return self.total_weight

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
        result = "OPADDistribution:\n"
        for key, weight in self.state_to_weight.items():
            array = np.frombuffer(key, dtype=self.dtype).reshape(self.shape)
            result += f"Array:\t{array}\t {weight}\n"
        return result
