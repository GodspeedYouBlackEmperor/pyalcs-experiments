import os
from utils.run_utils import Runner

import gym
import gym_maze

from lcs.agents import Agent
from lcs.agents.acs2 import ACS2, Configuration as CFG_ACS2
from lcs.agents.acs2er import ACS2ER, Configuration as CFG_ACS2ER, ReplayMemory, ReplayMemorySample
from lcs.agents.acs2rer import ACS2RER, Configuration as CFG_ACS2RER
from lcs.metrics import population_metrics

# Logger
import logging
logging.basicConfig(level=logging.INFO)

MAZE = "Maze5-v0"
EXPLORE_TRIALS = 3000
EXPLOIT_TRIALS = 500

# The size of ER replay memory buffer
ER_BUFFER_SIZE = 10000
# The minimum number of samples of ER replay memory buffer to start replying samples (warm-up phase)
ER_BUFFER_MIN_SAMPLES = 1000
# The number of samples to be replayed druing ER phase
ER_SAMPLES_NUMBER_LIST = [3]


#######

REPEAT_START = 1
REPEAT = 1

# Please edit if running new experiment to do not override saved results.
EXPERIMENT_NAME = "Maze5_PER_EXP1"

runner = Runner('MAZE', EXPERIMENT_NAME, MAZE)


def _get_transitions():
    knowledge_env = gym.make(MAZE)
    transitions = knowledge_env.env.get_transitions()
    transitions = list(map(lambda t: [knowledge_env.env.maze.perception(
        t[0]), t[1], knowledge_env.env.maze.perception(t[2])], transitions))

    return transitions


TRANSITIONS = _get_transitions()
TRANSITIONS_LENGTH = len(TRANSITIONS)


def _maze_knowledge(population) -> float:
    # Take into consideration only reliable classifiers
    reliable_classifiers = [c for c in population if c.is_reliable()]

    # Count how many transitions are anticipated correctly
    nr_correct = 0

    # For all possible destinations from each path cell
    for p0, action, p1 in TRANSITIONS:
        if any([True for cl in reliable_classifiers
                if cl.predicts_successfully(p0, action, p1)]):
            nr_correct += 1

    return nr_correct / TRANSITIONS_LENGTH * 100.0


def _maze_specificity(population) -> float:
    pop_len = len(population)
    if(pop_len) == 0:
        return 0
    return sum(map(lambda c: c.specificity, population)) / pop_len


def _maze_metrics(agent, env):
    pop = agent.population
    metrics = {
        'knowledge': _maze_knowledge(pop),
        "specificity": _maze_specificity(agent.population)
    }
    metrics.update(population_metrics(pop, env))

    return metrics


def _weight_func_reward(rm: ReplayMemory, sample: ReplayMemorySample):
    if(sample.reward == 0):
        return 1

    return 5


def _weight_func_unique(rm: ReplayMemory, sample: ReplayMemorySample):
    existing_count = sum(1 for s in rm if sample.state == s.state and sample.action ==
                         s.action and sample.reward == s.reward and sample.next_state == s.next_state and sample.done == s.done)

    return 1 / (existing_count * 2 + 1)


def _weight_func_unique_reward(rm: ReplayMemory, sample: ReplayMemorySample):
    return _weight_func_reward(rm, sample) * _weight_func_unique(rm, sample)


def _run_experiment(agent, path):
    runner.run_experiment(agent, gym.make(
        MAZE), EXPLORE_TRIALS, EXPLOIT_TRIALS, path)


def run_acs2_experiment():
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2(
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            user_metrics_collector_fcn=_maze_metrics)
        agent = ACS2(cfg)

        _run_experiment(agent, f'{i}')


def _run_acs2er_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2ER(
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            user_metrics_collector_fcn=_maze_metrics)
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
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            er_weight_function=_weight_func_reward,
            user_metrics_collector_fcn=_maze_metrics)
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
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            er_weight_function=_weight_func_unique,
            user_metrics_collector_fcn=_maze_metrics)
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
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            er_weight_function=_weight_func_unique_reward,
            user_metrics_collector_fcn=_maze_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-pER_unique_reward', f'{i}'))


def run_acs2per3_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2pER - unique + reward")
        _run_acs2per3_experiment(er_samples_number)
        print(f"END - ACS2pER - unique + reward")


def _run_acs2rer_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2ER(
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            user_metrics_collector_fcn=_maze_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-RER', f'{i}'))


def run_acs2rer_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2ER - {er_samples_number}")
        _run_acs2rer_experiment(er_samples_number)
        print(f"END - ACS2ER - {er_samples_number}")


run_acs2_experiment()
run_acs2er_experiments()
run_acs2per_experiments()
run_acs2per2_experiments()
run_acs2per3_experiments()
run_acs2rer_experiments()
