from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def next_sample_proposals_acceptances(self, current_state):
        pass
