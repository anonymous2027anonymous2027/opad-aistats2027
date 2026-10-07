"""Moment-recording routines used by the Ising R-hat experiments."""
from tqdm import tqdm
from om.experiments.utils import TimeKeeper
from om.tools.histogram import ArrayHistogram
from om.tools.opad import OPADDistribution
from om.tools.rb_wr import RB_WR_Distrib


def _require_axis_mode(axis_mode):
    if axis_mode not in ('time', 'iters'):
        raise ValueError(axis_mode)


def compute_greedy_expected_error(
        axis_mode,
        num_records,
        greedy,
        model,
        func_state,
        ground_truth_expected_func_value,
        equivalent_num_mcmc_samples_per_chain=None,
        total_sampling_time_per_chain_seconds=None,
        do_compute_rhat=True,
):
    _require_axis_mode(axis_mode)
    num_mutations_per_evolution = greedy.num_mutations_per_evolutions()

    recorded_evals = []
    recorded_times = []
    recorded_errors = []
    recorded_means = []
    recorded_variances = []

    time_keeper = TimeKeeper()

    if axis_mode == 'iters':
        num_evolution_steps = int(equivalent_num_mcmc_samples_per_chain / num_mutations_per_evolution)
        recorded_every = int(num_evolution_steps / num_records) + 1
        evolution_steps = range(1, num_evolution_steps + 1)

        for evolution_step in tqdm(evolution_steps):
            time_keeper.start()
            greedy.evolve()
            time_keeper.stop()

            if evolution_step % recorded_every != 0:
                continue

            approx_mean = greedy.inner_opad.calc_expected_value(func=func_state)
            recorded_evals.append(evolution_step * num_mutations_per_evolution)
            recorded_times.append(time_keeper.get_passed_time())
            recorded_errors.append(approx_mean - ground_truth_expected_func_value)

            if do_compute_rhat:
                recorded_means.append(approx_mean)
                recorded_variances.append(greedy.inner_opad.calc_variance(func=func_state))

        print('Search passed time: ', time_keeper.get_passed_time())
    else:
        recording_interval = total_sampling_time_per_chain_seconds / num_records
        num_recorded_entries = 0
        passed_time = 0
        evolution_step = 0

        while passed_time <= total_sampling_time_per_chain_seconds:
            time_keeper.start()
            greedy.evolve()
            time_keeper.stop()

            evolution_step += 1
            passed_time = time_keeper.get_passed_time()

            while passed_time > recording_interval * (num_recorded_entries + 1):
                num_recorded_entries += 1
                if num_recorded_entries > num_records:
                    break

                approx_mean = greedy.inner_opad.calc_expected_value(func=func_state)
                recorded_evals.append(evolution_step * num_mutations_per_evolution)
                recorded_times.append(passed_time)
                recorded_errors.append(approx_mean - ground_truth_expected_func_value)

                if do_compute_rhat:
                    recorded_means.append(approx_mean)
                    recorded_variances.append(greedy.inner_opad.calc_variance(func=func_state))

        assert len(recorded_times) == num_records, (
            f'no. recorded times: {len(recorded_times)} but it had to be: {num_records}'
        )
        print("Search total Passed time:", passed_time)

    return {
        'Evals': recorded_evals,
        'Times': recorded_times,
        'Greedy': recorded_errors,
        'Means': recorded_means,
        'Variances': recorded_variances,
    }

def compute_mcmc_expected_error(
        axis_mode,
        num_records,
        sampler,
        init_state,
        model,
        func_state,
        ground_truth_expected_func_value,
        num_mcmc_samples=None,
        total_sampling_time_per_chain_seconds=None,
        do_compute_rhat=True,
):
    _require_axis_mode(axis_mode)
    current_state = init_state.copy()
    mcmc_hist = ArrayHistogram()
    opad_accepted_only = OPADDistribution(target_model=model)  # OPAD
    opad_with_proposals = OPADDistribution(target_model=model)  # OPAD+
    rb_distrib = RB_WR_Distrib()

    recorded_iters = []
    recorded_times = []
    recorded_mcmc_errors = []
    recorded_opad_errors = []
    recorded_opad_plus_errors = []
    recorded_rb_errors = []  # RB/WR

    recorded_mcmc_means = []
    recorded_mcmc_variances = []
    recorded_opad_means = []
    recorded_opad_variances = []
    recorded_opad_plus_means = []
    recorded_opad_plus_variances = []
    recorded_rb_means = []
    recorded_rb_variances = []

    time_keeper = TimeKeeper()

    def record_snapshot(sample_count, passed_time):
        approx_mcmc = mcmc_hist.calc_expected_value(func=func_state)
        approx_opad = opad_accepted_only.calc_expected_value(func=func_state)
        approx_opad_plus = opad_with_proposals.calc_expected_value(func=func_state)
        approx_rb = rb_distrib.calc_expected_value(func=func_state)

        recorded_iters.append(sample_count)
        recorded_times.append(passed_time)
        recorded_mcmc_errors.append(approx_mcmc - ground_truth_expected_func_value)
        recorded_opad_errors.append(approx_opad - ground_truth_expected_func_value)
        recorded_opad_plus_errors.append(approx_opad_plus - ground_truth_expected_func_value)
        recorded_rb_errors.append(approx_rb - ground_truth_expected_func_value)

        if do_compute_rhat:
            recorded_mcmc_means.append(approx_mcmc)
            recorded_opad_means.append(approx_opad)
            recorded_opad_plus_means.append(approx_opad_plus)
            recorded_rb_means.append(approx_rb)

            recorded_mcmc_variances.append(mcmc_hist.calc_variance(func=func_state))
            recorded_opad_variances.append(opad_accepted_only.calc_variance(func=func_state))
            recorded_opad_plus_variances.append(opad_with_proposals.calc_variance(func=func_state))
            recorded_rb_variances.append(rb_distrib.calc_variance(func=func_state))

    if axis_mode == 'iters':
        recorded_every = int(num_mcmc_samples / num_records) + 1

        for sample_count in tqdm(range(1, num_mcmc_samples + 1)):
            time_keeper.start()
            previous_state = current_state.copy()
            current_state, proposed_states, acceptances = sampler.next_sample_proposals_acceptances(current_state)
            time_keeper.stop()

            assert len(proposed_states) == 1
            proposed_state = proposed_states[0]
            acceptance = acceptances[0]

            mcmc_hist.add_array(current_state)
            opad_accepted_only.add_array(current_state)
            opad_with_proposals.add_array(current_state)
            opad_with_proposals.add_array(proposed_state)
            rb_distrib.add_previous_and_proposed_states(previous_state=previous_state,
                                                        proposed_state=proposed_state, acceptance_prob=acceptance)

            if sample_count % recorded_every == 0:
                record_snapshot(sample_count=sample_count, passed_time=time_keeper.get_passed_time())

        print(f"One MCMC chain total samplng time: {time_keeper.get_passed_time()} seconds")
    else:
        recording_interval = total_sampling_time_per_chain_seconds / num_records
        num_recorded_entries = 0
        passed_time = 0
        sample_count = 0

        while passed_time <= total_sampling_time_per_chain_seconds:
            time_keeper.start()
            previous_state = current_state.copy()
            current_state, proposed_states, acceptances = sampler.next_sample_proposals_acceptances(current_state)
            time_keeper.stop()

            passed_time = time_keeper.get_passed_time()
            sample_count += 1

            assert len(proposed_states) == 1
            proposed_state = proposed_states[0]
            acceptance = acceptances[0]

            mcmc_hist.add_array(current_state)
            opad_accepted_only.add_array(current_state)
            opad_with_proposals.add_array(current_state)
            opad_with_proposals.add_array(proposed_state)
            rb_distrib.add_previous_and_proposed_states(previous_state=previous_state,
                                                        proposed_state=proposed_state,
                                                        acceptance_prob=acceptance)

            while passed_time > recording_interval * (num_recorded_entries + 1):
                num_recorded_entries += 1
                if num_recorded_entries > num_records:
                    break
                record_snapshot(sample_count=sample_count, passed_time=passed_time)

        assert len(recorded_times) == num_records, (
            f'no. recorded times: {len(recorded_times)} but it had to be: {num_records}'
        )
        print('MCMC passed_time: ', passed_time)

    return {
        'Iters': recorded_iters,
        'Times': recorded_times,
        'MCMC': recorded_mcmc_errors,
        'OPAD': recorded_opad_errors,
        'OPAD+': recorded_opad_plus_errors,
        'RB': recorded_rb_errors,
        'MCMC.means': recorded_mcmc_means,
        'OPAD.means': recorded_opad_means,
        'OPAD+.means': recorded_opad_plus_means,
        'RB.means': recorded_rb_means,
        'MCMC.variances': recorded_mcmc_variances,
        'OPAD.variances': recorded_opad_variances,
        'OPAD+.variances': recorded_opad_plus_variances,
        'RB.variances': recorded_rb_variances
    }
