from om.models.model import DiscreteDistributionModel
from om.tools.opad import OPADDistribution


class DiscreteParticleVariationalInference:
    def __init__(self, model: DiscreteDistributionModel,
                 num_particles: int,
                 possible_values=(-1, 1)):
        """
        :param model: a model that assigns probabilities to numpy arrays with elements being -1 and +1 only.
        E.g. Ising model (of arbitrary dimensions)
        :param possible_values: 2 possible values that fill each object grid. default is a SpinArray
        :return:
        """
        self.model = model
        self.possible_values = possible_values

        self.dim = self.model.get_dimension()  # (N)
        self.index_to_flip = self.first_flippable_index()  # (n in {0, .., N-1})  -- this dimension of the particles will be updated
        self.num_particles = num_particles  # (K)

        self.particles_scores = OPADDistribution(self.model)
        while self.particles_scores.num_entries() < self.num_particles:
            new_state = model.generate_init_state()
            if self.is_valid_state(new_state):
                self.particles_scores.add_array(new_state)

    def first_flippable_index(self):
        return 0  # override if the first indices are not flippable (e.g. the first element associates with a bias term and is always 1)

    def is_valid_state(self, state):
        return True  # override if say the state should always start with 1.

    def evolve_all_states_in_one_dim(self):
        """
        internally evolves the states, updates the index_to_flip (a.k.a. the current dimension)
        :return: all_newly_fetched_particles_scores even though only a subset of them will affect the (internal) variational particles
        NOTE: TO update all particles in all dimensions, use this method self.dim times
        """
        all_newly_fetched_particles_scores = OPADDistribution(self.model)

        for particle in self.particles_scores.generate_all_states():
            particle0 = particle.copy()
            particle1 = particle.copy()
            particle0[self.index_to_flip] = self.possible_values[0]
            particle1[self.index_to_flip] = self.possible_values[1]
            all_newly_fetched_particles_scores.add_array(particle0)
            all_newly_fetched_particles_scores.add_array(particle1)

        top_new_particles = all_newly_fetched_particles_scores.fetch_K_top_states(K=self.num_particles)

        self.particles_scores = OPADDistribution(self.model)  # empty the current particle set
        for particle in top_new_particles:
            self.particles_scores.add_array(particle)

        self.index_to_flip = (self.index_to_flip + 1) % self.dim
        if self.index_to_flip == 0:
            self.index_to_flip = self.first_flippable_index()

        return all_newly_fetched_particles_scores


class FixedFirstDiscreteParticleVariationalInference(DiscreteParticleVariationalInference):
    def __init__(self, model: DiscreteDistributionModel,
                 num_particles: int,
                 possible_values=(0, 1), fixed_first_value=1):
        self.fixed_first_value = fixed_first_value
        super().__init__(model=model, num_particles=num_particles, possible_values=possible_values)

    def first_flippable_index(self):
        return 1  # element with index 0 is always fixed

    def is_valid_state(self, state):
        return state[0] == self.fixed_first_value
