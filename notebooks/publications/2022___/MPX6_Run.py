import os
import itertools
import numpy as np
from utils.run_utils import Runner

import gym
import gym_multiplexer
from gym_multiplexer.utils import get_correct_answer

from lcs import Perception
from lcs.agents.acs2 import ACS2, Configuration as CFG_ACS2, Classifier
from lcs.agents.acs2er import ACS2ER, Configuration as CFG_ACS2ER, ReplayMemory, ReplayMemorySample
from lcs.metrics import population_metrics

# Logger
import logging
logging.basicConfig(level=logging.INFO)


BITS = 6  # 6 | 11 | 20 | 37
MPX = f'boolean-multiplexer-{BITS}bit-v0'
EXPLORE_TRIALS = 5000
EXPLOIT_TRIALS = 1000
METRICS_FREQUENCY = 1
# applies only when 20 or 37 bits, otherwise all possible states verified
KNOWLEDGE_STATE_SAMPLES = 1000

# The size of ER replay memory buffer
ER_BUFFER_SIZE = 10000
# The minimum number of samples of ER replay memory buffer to start replying samples (warm-up phase)
ER_BUFFER_MIN_SAMPLES = 100
# The number of samples to be replayed druing ER phase
ER_SAMPLES_NUMBER_LIST = [3]


#######

REPEAT_START = 1
REPEAT = 1

# Please edit if running new experiment to do not override saved results.
EXPERIMENT_NAME = "MPX6_PER_EXP1"


runner = Runner('MPX', EXPERIMENT_NAME, MPX)


class MpxObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return [str(x) for x in observation]


knowledge_env = MpxObservationWrapper(gym.make(MPX))


def get_transitions(states):
    transitions = list(map(lambda s:
                           (Perception([str(float(x)) for x in s] + ['0.0']),
                            get_correct_answer(
                                list(s) + [0], knowledge_env.env.env.control_bits),
                            Perception([str(float(x)) for x in s] + ['1.0'])),
                           states))

    return transitions


def _mpx_knowledge(population, transitions, transitions_length) -> float:
    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    if(len(reliable_classifiers) == 0):
        return 0

    nr_correct = 0

    for p0, correct_answer, p1 in transitions:
        if any([True for cl in reliable_classifiers if
                cl.predicts_successfully(
                    p0,
                    correct_answer,
                    p1)]):

            nr_correct += 1

    return nr_correct / transitions_length


if BITS == 6 or BITS == 11:  # Verify all
    def get_all_transitions():
        states = list(itertools.product([0, 1], repeat=BITS))
        return get_transitions(states)

    TRANSITIONS = get_all_transitions()
    TRANSITIONS_LENGTH = len(TRANSITIONS)

    def mpx_knowledge(population) -> float:
        return _mpx_knowledge(population, TRANSITIONS, TRANSITIONS_LENGTH)

elif BITS == 20 or BITS == 37:  # Verify samples
    def get_sampled_transitions():
        states = np.random.randint(2, size=(KNOWLEDGE_STATE_SAMPLES, BITS))
        return get_transitions(states)

    def mpx_knowledge(population) -> float:
        return _mpx_knowledge(population, get_sampled_transitions(), KNOWLEDGE_STATE_SAMPLES)
else:
    raise Exception(f'Unsupported BITS number: {BITS}')


def mpx_specificity(population) -> float:
    pop_len = len(population)
    if(pop_len) == 0:
        return 0
    return sum(map(lambda c: c.specificity, population)) / pop_len


def mpx_metrics(agent, env):
    metrics = {
        # "knowledge": mpx_knowledge(agent.population),
        "specificity": mpx_specificity(agent.population)
    }
    metrics.update(population_metrics(agent.population, env))

    return metrics


def _weight_func_reward(rm: ReplayMemory, sample: ReplayMemorySample):
    if(sample.reward == 0):
        return 1

    return 2


def _weight_func_unique(rm: ReplayMemory, sample: ReplayMemorySample):
    existing_count = sum(1 for s in rm if sample.state == s.state and sample.action ==
                         s.action and sample.reward == s.reward and sample.next_state == s.next_state and sample.done == s.done)

    return 1 / (existing_count + 1)


def _weight_func_unique_reward(rm: ReplayMemory, sample: ReplayMemorySample):
    return _weight_func_reward(rm, sample) * _weight_func_unique(rm, sample)


def _run_experiment(agent, path):
    runner.run_experiment(agent, MpxObservationWrapper(
        gym.make(MPX)), EXPLORE_TRIALS, EXPLOIT_TRIALS, path)


def run_acs2_experiment():
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2(
            classifier_length=knowledge_env.env.observation_space.n,
            number_of_possible_actions=2,
            do_ga=True,
            metrics_trial_frequency=METRICS_FREQUENCY,
            user_metrics_collector_fcn=mpx_metrics)
        agent = ACS2(cfg)

        _run_experiment(agent, f'{i}')


def _run_acs2er_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2ER(
            classifier_length=knowledge_env.env.observation_space.n,
            number_of_possible_actions=2,
            do_ga=True,
            metrics_trial_frequency=METRICS_FREQUENCY,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            user_metrics_collector_fcn=mpx_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-ER', f'{i}'))


def run_acs2er_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2ER - {er_samples_number}")
        _run_acs2er_experiment(er_samples_number)
        print(f"END - ACS2ER - {er_samples_number}")


def _run_acs2per_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2ER(
            classifier_length=knowledge_env.env.observation_space.n,
            number_of_possible_actions=2,
            do_ga=True,
            metrics_trial_frequency=METRICS_FREQUENCY,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            er_weight_function=_weight_func_reward,
            user_metrics_collector_fcn=mpx_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-pER_reward', f'{i}'))


def run_acs2per_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2pER - reward")
        _run_acs2per_experiment(er_samples_number)
        print(f"END - ACS2pER - reward")


def _run_acs2per2_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2ER(
            classifier_length=knowledge_env.env.observation_space.n,
            number_of_possible_actions=2,
            do_ga=True,
            metrics_trial_frequency=METRICS_FREQUENCY,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            er_weight_function=_weight_func_unique,
            user_metrics_collector_fcn=mpx_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-pER_unique', f'{i}'))


def run_acs2per2_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2pER - unique")
        _run_acs2per2_experiment(er_samples_number)
        print(f"END - ACS2pER - unique")


def _run_acs2per3_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2ER(
            classifier_length=knowledge_env.env.observation_space.n,
            number_of_possible_actions=2,
            do_ga=True,
            metrics_trial_frequency=METRICS_FREQUENCY,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            er_weight_function=_weight_func_unique_reward,
            user_metrics_collector_fcn=mpx_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-pER_unique_reward', f'{i}'))


def run_acs2per3_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2pER - unique + reward")
        _run_acs2per3_experiment(er_samples_number)
        print(f"END - ACS2pER - unique + reward")
