import random

from om.models.model import DiscreteDistributionModel
from om.tools.opad import OPADDistribution


class GreedyExplore:
    def __init__(self, model: DiscreteDistributionModel,
                 possible_values=(-1, 1), fixed_first_value=None):
        """
        :param model: a model that assigns probabilities to numpy arrays with elements being -1 and +1 only.
        E.g. Ising model (of arbitrary dimensions)
        :param possible_values: 2 possible values that fill each object grid. default is a SpinArray
        :param fixed_first_value: if fix_first_element_to is not None, then it should be a possible value and the fist element will be fixed to this
        :return: Greedy algorith that returns all the neighbours of the top-most unbred state
        """

        if fixed_first_value is not None:
            assert fixed_first_value in possible_values

        init_state = model.generate_init_state()
        if fixed_first_value is not None:
            assert init_state[
                       0] == fixed_first_value, f'Model does not give a valid init state: {init_state} does not start with {fixed_first_value}'

        self.inner_opad = None
        self.__strategy = None

        self.fixed_first_element = fixed_first_value
        self.model = model
        self.possible_values = possible_values
        self.dim = model.get_dimension()  # (N)

        self.candid_to_predicted_score = None
        self.candid_to_num_proposals = None

    def initialize(self, strategy, init_state):
        assert strategy == 'NWSS'
        self.__strategy = strategy

        self.inner_opad = OPADDistribution(self.model)  # the inner OPAD distribution
        self.inner_opad.add_array(init_state.copy())  # to make sure the init state is never changed


        if strategy == 'NWSS':
            self.candid_to_predicted_score = dict()  # numpy array.tobyte() -> average of the
            self.candid_to_num_proposals = dict()  # numpy array.tobyte() -> dict('avg':?, 'n'=?)
            self._update_candid_dicts_with_neighbours(next(iter(self.inner_opad.state_to_weight)))

    def _update_candid_dicts_with_neighbours(self, state_byte):

        suggested_score = self.inner_opad.state_to_weight[
            state_byte]  # the score of this state will be suggested as the predicted score of the neighbours
        predictor_numpy = self.inner_opad.arraybyte_to_numpy_array(state_byte)

        start_index = 0 if self.fixed_first_element is None else 1
        for index_to_flip in range(start_index, self.dim):
            neighbour = predictor_numpy.copy()
            neighbour[index_to_flip] = self.flip_value(neighbour[index_to_flip])
            neighbour_bytes = neighbour.tobytes()
            if neighbour_bytes not in self.inner_opad.state_to_weight:
                if neighbour_bytes in self.candid_to_num_proposals:
                    n = self.candid_to_num_proposals[neighbour_bytes]
                    prev_score = self.candid_to_predicted_score[neighbour_bytes]
                    self.candid_to_num_proposals[neighbour_bytes] = n + 1
                    self.candid_to_predicted_score[neighbour_bytes] = (prev_score * n + suggested_score) / (n + 1)
                else:
                    self.candid_to_num_proposals[neighbour_bytes] = 1
                    self.candid_to_predicted_score[neighbour_bytes] = suggested_score

    def flip_value(self, value):
        if value == self.possible_values[0]:
            return self.possible_values[1]
        if value == self.possible_values[1]:
            return self.possible_values[0]
        raise

    def evolve(self):
        self._partially_breed_a_probabilistically_chosen_unbred_particle()



    def _partially_breed_a_probabilistically_chosen_unbred_particle(self,
                                                                    max_candid_size=2000,
                                                                    retain_candid_size=1000,
                                                                    do_raise_exception_on_exhausting_space=False):
        """
        add A SINGLE neighbours of a state (that is chosen proportional to its probability) to the inner OM
        """
        assert self.__strategy == 'NWSS'

        self._prune_candid_dicts(max_size=max_candid_size, retain_size=retain_candid_size)

        candid_list = list(self.candid_to_predicted_score.keys())

        if len(candid_list) == 0:
            if do_raise_exception_on_exhausting_space:
                raise Exception("It seems all the space states are already visited")
            else:
                return

        candid_pred_scores = [self.candid_to_predicted_score[candid] for candid in candid_list]

        selected_candid_byte = random.choices(candid_list, weights=candid_pred_scores, k=1)[0]
        assert selected_candid_byte not in self.inner_opad.state_to_weight, "Candidate should not be there"
        selected_candid_state = self.inner_opad.arraybyte_to_numpy_array(selected_candid_byte)
        self.inner_opad.add_array(selected_candid_state)
        self._update_candid_dicts_with_neighbours(selected_candid_byte)
        del self.candid_to_predicted_score[selected_candid_byte]
        del self.candid_to_num_proposals[selected_candid_byte]

    def num_mutations_per_evolutions(self):
        return 1

    def _prune_candid_dicts(self, max_size: int, retain_size: int):
        """
        if the size of candid dicts exceeds the max_size then only keeps the top retain_size ones
        :param max_size:
        :param retain_size: note: retain_size should be less than the max_size
        :return:
        """
        if len(self.candid_to_predicted_score) <= max_size:
            return
        assert len(self.candid_to_predicted_score) == len(self.candid_to_num_proposals)
        assert retain_size < max_size

        top_candid_byte_list = sorted(self.candid_to_predicted_score, key=self.candid_to_predicted_score.get,
                                      reverse=True)[:retain_size]

        pruned_candid_to_predicted_score = {cand: self.candid_to_predicted_score[cand] for cand in top_candid_byte_list}
        pruned_candid_to_num_proposals = {cand: self.candid_to_num_proposals[cand] for cand in top_candid_byte_list}

        self.candid_to_predicted_score.clear()
        self.candid_to_predicted_score.update(pruned_candid_to_predicted_score)

        self.candid_to_num_proposals.clear()
        self.candid_to_num_proposals.update(pruned_candid_to_num_proposals)
