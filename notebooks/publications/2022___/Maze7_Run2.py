import os
from collections import defaultdict
from utils.run_utils import Runner

import gym
import gym_maze

from lcs.agents import Agent
from lcs.agents.acs2 import ACS2, Configuration as CFG_ACS2, ClassifiersList
from lcs.agents.acs2er import ACS2ER, Configuration as CFG_ACS2ER, ReplayMemory, ReplayMemorySample
from lcs.agents.acs2eer import ACS2EER, Configuration as CFG_ACS2EER, TrialReplayMemory
from lcs.agents.acs2rer import ACS2RER, Configuration as CFG_ACS2RER, TrialReplayMemory
from lcs.metrics import population_metrics

# Logger
import logging
logging.basicConfig(level=logging.INFO)

MAZE = "Maze7-v0"
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
REPEAT = 2

# Please edit if running new experiment to do not override saved results.
EXPERIMENT_NAME = "Maze7_PER_EXP10"

runner = Runner('MAZE', EXPERIMENT_NAME, MAZE)

MAZE_PATH = 0
MAZE_REWARD = 9

optimal_paths_env = gym.make(MAZE)
matrix = optimal_paths_env.matrix
X = matrix.shape[1]
Y = matrix.shape[0]


def get_reward_pos():
    for i in range(Y):
        for j in range(X):
            if(matrix[i, j] == MAZE_REWARD):
                return(i, j)


def get_possible_neighbour_cords(pos_y, pos_x):
    n = ((pos_y - 1, pos_x), 4)
    ne = ((pos_y - 1, pos_x + 1), 5)
    e = ((pos_y, pos_x + 1), 6)
    se = ((pos_y + 1, pos_x + 1), 7)
    s = ((pos_y + 1, pos_x), 0)
    sw = ((pos_y + 1, pos_x - 1), 1)
    w = ((pos_y, pos_x - 1), 2)
    nw = ((pos_y - 1, pos_x - 1), 3)

    return [n, ne, e, se, s, sw, w, nw]


optimal_actions = []

root_node = get_reward_pos()


def is_included(cords, level):
    return any(op_cords[0] == cords[0] and op_cords[1] == cords[1] and level != op_level for op_cords, _, op_level in optimal_actions)


def get_optimal_actions_to(node, level):
    neighbour_cords = get_possible_neighbour_cords(node[0], node[1])

    next_level_cords = []
    for (pos_y, pos_x), action in neighbour_cords:
        if (not is_included((pos_y, pos_x), level)) and matrix[pos_y, pos_x] == MAZE_PATH:
            optimal_actions.append(((pos_y, pos_x), action, level))
            next_level_cords.append((pos_y, pos_x))

    return next_level_cords


LEVEL = 0
next_level_cords = get_optimal_actions_to(root_node, LEVEL)

while len(next_level_cords) > 0:
    LEVEL += 1
    new_next_level_cords = []
    for nlc in next_level_cords:
        new_next_level_cords += get_optimal_actions_to(nlc, LEVEL)

    next_level_cords = new_next_level_cords

positions_actions = defaultdict(set)
for cords, a, _ in optimal_actions:
    positions_actions[cords].add(a)

positions_actions = positions_actions.items()
POSITIONS_OPTIMAL_ACTIONS = list(map(lambda pa: (
    optimal_paths_env.env.maze.perception(pa[0]), list(pa[1])), positions_actions))
POSITIONS_OPTIMAL_ACTIONS_LENGTH = len(POSITIONS_OPTIMAL_ACTIONS)


def _maze_optimal(classifiers) -> float:
    nr_correct = 0

    for p0, optimal_actions_list in POSITIONS_OPTIMAL_ACTIONS:
        match_set = classifiers.form_match_set(p0)
        cl = match_set.get_best_classifier()

        if cl is not None and optimal_actions_list.count(cl.action) > 0:
            nr_correct += 1

    return nr_correct / POSITIONS_OPTIMAL_ACTIONS_LENGTH * 100.0


def _maze_optimal_reliable(classifiers) -> float:
    return _maze_optimal(ClassifiersList(*[c for c in classifiers if c.is_reliable()]))


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
        "specificity": _maze_specificity(agent.population),
        "optimal": _maze_optimal(agent.population),
        "optimal_reliable": _maze_optimal_reliable(agent.population)
    }
    metrics.update(population_metrics(pop, env))

    return metrics


def _weight_func_reward(rm: ReplayMemory, sample: ReplayMemorySample):
    if(sample.reward == 0):
        return 1

    return 10


def _weight_func_unique(rm: ReplayMemory, sample: ReplayMemorySample):
    existing_count = sum(1 for s in rm if sample.state == s.state and sample.action ==
                         s.action and sample.reward == s.reward and sample.next_state == s.next_state and sample.done == s.done)

    return 1 / (existing_count + 1)


def _weight_func_unique_reward(rm: ReplayMemory, sample: ReplayMemorySample):
    return _weight_func_reward(rm, sample) * _weight_func_unique(rm, sample)


def _are_samples_equal(s1: ReplayMemory, s2: ReplayMemorySample):
    return s1.state == s2.state and s1.action == s2.action and s1.reward == s2.reward and s1.next_state == s2.next_state and s1.done == s2.done


def _are_trials_equal(t1: TrialReplayMemory, t2: TrialReplayMemory):
    if len(t1) != len(t2):
        return False

    for i in range(len(t1)):
        if not _are_samples_equal(t1[i], t2[i]):
            return False

    return True


def _are_trials_equal_reverted(t1: TrialReplayMemory, t2: TrialReplayMemory):
    if len(t1) != len(t2):
        return False

    for i in range(1, len(t1) + 1):
        if not _are_samples_equal(t1[-i], t2[-i]):
            return False

    return True


def _rer_weight_func_reward(rm: ReplayMemory, sample: TrialReplayMemory):
    if(sample[0].reward == 0):
        return 1

    return 5


def _rer_weight_func_unique(rm: ReplayMemory, sample: TrialReplayMemory):
    existing_count = sum(
        1 for s in rm if _are_trials_equal_reverted(sample, s))

    return 1 / (existing_count * 2 + 1)


def _rer_weight_func_unique_reward(rm: ReplayMemory, sample: TrialReplayMemory):
    return _rer_weight_func_reward(rm, sample) * _rer_weight_func_unique(rm, sample)


def _eer_weight_func_reward(rm: ReplayMemory, sample: TrialReplayMemory):
    if(sample[-1].reward == 0):
        return 1

    return 5


def _eer_weight_func_unique(rm: ReplayMemory, sample: TrialReplayMemory):
    existing_count = sum(1 for s in rm if _are_trials_equal(sample, s))

    return 1 / (existing_count * 2 + 1)


def _eer_weight_func_unique_reward(rm: ReplayMemory, sample: TrialReplayMemory):
    return _eer_weight_func_reward(rm, sample) * _eer_weight_func_unique(rm, sample)


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


def _run_acs2eer_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2EER(
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=int(ER_BUFFER_MIN_SAMPLES / 20),
            er_samples_number=er_samples_number,
            user_metrics_collector_fcn=_maze_metrics)
        agent = ACS2EER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-EER', f'{i}'))


def run_acs2eer_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2EER - {er_samples_number}")
        _run_acs2eer_experiment(er_samples_number)
        print(f"END - ACS2EER - {er_samples_number}")


def _run_acs2peer_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2EER(
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=int(ER_BUFFER_MIN_SAMPLES / 20),
            er_samples_number=er_samples_number,
            er_weight_function=_eer_weight_func_reward,
            user_metrics_collector_fcn=_maze_metrics)
        agent = ACS2EER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-EER_reward', f'{i}'))


def run_acs2peer_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2EER - reward - {er_samples_number}")
        _run_acs2peer_experiment(er_samples_number)
        print(f"END - ACS2EER - reward - {er_samples_number}")


def _run_acs2peer2_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent
        cfg = CFG_ACS2EER(
            classifier_length=8,
            number_of_possible_actions=8,
            metrics_trial_frequency=1,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=int(ER_BUFFER_MIN_SAMPLES / 20),
            er_samples_number=er_samples_number,
            er_weight_function=_eer_weight_func_unique_reward,
            user_metrics_collector_fcn=_maze_metrics)
        agent = ACS2EER(cfg)

        _run_experiment(agent, os.path.join(f'm_3-EER_reward-unique', f'{i}'))


def run_acs2peer2_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2EER - reward - unique - {er_samples_number}")
        _run_acs2peer2_experiment(er_samples_number)
        print(f"END - ACS2EER - reward - {er_samples_number}")


run_acs2_experiment()
run_acs2er_experiments()
run_acs2eer_experiments()
run_acs2peer_experiments()
run_acs2peer2_experiments()
