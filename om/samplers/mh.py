import numpy as np
import numpy.random as random
from om.models.model import DiscreteDistributionModel
from om.samplers.sampler import Sampler


class BiValueArrayMetropolisSampler(Sampler):
    def __init__(self, model: DiscreteDistributionModel, possible_values=(-1, 1)):
        """
        :param model: a model that assigns probabilities to numpy arrays with elements being -1 and +1 only.
        E.g. Ising model (of arbitrary dimensions)
        :param possible_values: 2 possible values that fill each object grid. default is a SpinArray
        :return:
        """
        self.model = model
        self.possible_values = possible_values

    def rand_index(self, state):
        """
        :param state: a state is a numpy array of arbitrary dimension(s)
        :return: random index of an element of the state
        E.g. if state is
        [[a, b, c],
         [d, e, f]] which is (2, 3) then the index will be a random typle(i<2, j<3)
        """
        return tuple(random.choice(range(dim)) for dim in state.shape)

    def propose_next(self, current_state):
        """
        :param current_state: can be a vector or matrix or tensor
        :return:
        """
        proposed_state = current_state.copy()

        index_to_flip = self.rand_index(proposed_state)

        if proposed_state[index_to_flip] == self.possible_values[0]:
            proposed_state[index_to_flip] = self.possible_values[1]
        elif proposed_state[index_to_flip] == self.possible_values[1]:
            proposed_state[index_to_flip] = self.possible_values[0]
        else:
            raise Exception(
                "proposed_state[index_to_flip] = {v} is not a valid value".format(
                    v=proposed_state[index_to_flip]
                )
            )

        return proposed_state

    def next_sample_proposals_acceptances(self, current_state):
        proposed_state = self.propose_next(current_state)

        current_neg_log_prob = self.model.calc_neg_log_unnormalized_prob(current_state)
        proposed_neg_log_prob = self.model.calc_neg_log_unnormalized_prob(proposed_state)

        acceptance_prob = min(1.0, np.exp(current_neg_log_prob - proposed_neg_log_prob))

        if random.uniform(0, 1) < acceptance_prob:
            current_state = proposed_state

        return current_state, [proposed_state], [acceptance_prob]


class FixedFirstElementArraySampler(BiValueArrayMetropolisSampler):
    def __init__(self, model: DiscreteDistributionModel, possible_values=(0, 1)):
        """
        Used for Bayesian variable selection where the states are vectors
        and the first element is always fixed
        """
        super().__init__(model=model, possible_values=possible_values)

    def rand_index(self, state):
        assert len(state.shape) == 1
        return random.choice(range(1, state.shape[0]))
