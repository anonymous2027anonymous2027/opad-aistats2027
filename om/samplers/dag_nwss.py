from om.samplers.greedy import GreedyExplore

class DAGNWSS(GreedyExplore):
    """NWSS on the same valid add/delete/reverse neighborhood as StructureMCMC."""
    def __init__(self, model, neighborhood):
        self.neighborhood = neighborhood
        super().__init__(model, possible_values=(0, 1))

    def _update_candid_dicts_with_neighbours(self, state_byte):
        score = self.inner_opad.state_to_weight[state_byte]
        state = self.inner_opad.arraybyte_to_numpy_array(state_byte)
        for neighbor in self.neighborhood.neighbors(state):
            key = neighbor.tobytes()
            if key in self.inner_opad.state_to_weight:
                continue
            n = self.candid_to_num_proposals.get(key, 0)
            previous = self.candid_to_predicted_score.get(key, 0)
            self.candid_to_num_proposals[key] = n + 1
            self.candid_to_predicted_score[key] = (previous * n + score) / (n + 1)
