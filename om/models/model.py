from abc import ABC, abstractmethod
import numpy as np
import warnings


class DiscreteDistributionModel(ABC):
    def get_dimension(self):
        raise Exception('Not implemented')

    def generate_init_state(self):
        """
        :return: a random state with positive probability (used as the initial state for sampling)
        """
        raise Exception('Not implemented')

    @abstractmethod
    def calc_neg_log_unnormalized_prob(self, x):
        pass

    def calc_unnormalized_prob(self, x):
        neg_log_prob = self.calc_neg_log_unnormalized_prob(x)
        return np.exp(-neg_log_prob)

    @abstractmethod
    def generate_all_states(self):
        pass

    def calc_all_states_and_unnormalized_probs(self):
        """Calculate unnormalized probabilities for all configurations."""
        states = self.generate_all_states()
        unnormalized_probs = np.zeros(len(states))
        for i, config in enumerate(states):
            unnormalized_probs[i] = self.calc_unnormalized_prob(config)
        return states, unnormalized_probs

    def print_unnormalized_probabilities(self):
        """Print states and their unnormalized probabilities."""
        states, unnormalized_probs = self.calc_all_states_and_unnormalized_probs()
        for stat, prob in zip(states, unnormalized_probs):
            print(f"State:\n{stat}\nUnnormalized Probability: {prob}\n")

    def calc_normalization_factor(self):
        """
        :return: Z
        """
        warnings.warn(
            "Warning: This is not efficient! If you can compute it in advance (and only once) then override this method")
        _, unnormalized_probs = self.calc_all_states_and_unnormalized_probs()
        return sum(unnormalized_probs)

    def calc_expected_value(self, func=lambda x: x):
        """
        :param func: a function with scalar output that accepts the distribution states. Default: identity function
        :return: E[f(X)] or E[X] if no function is provided (w.r.t. this distribution)
        """
        all_states, unnormalized_probs = self.calc_all_states_and_unnormalized_probs()
        out = 0
        for state, unnorm_prob in zip(all_states, unnormalized_probs):
            out += unnorm_prob * func(state)

        return out / sum(unnormalized_probs)

    def calc_variance(self, func=lambda x: x):
        """
        :param func: a function with scalar output that accepts the distribution states. Default: identity function
        :return: E[(f(X)-E[f(X)])^2] = E[f(X)^2] - E[f(X)]^2
        """
        all_states, unnormalized_probs = self.calc_all_states_and_unnormalized_probs()
        z = sum(unnormalized_probs)
        a = 0
        a2 = 0
        for state, unnorm_prob in zip(all_states, unnormalized_probs):
            a += unnorm_prob * func(state)
            a2 += unnorm_prob * (func(state)**2)
        a /= z
        a2 /= z
        return a2 - a**2
