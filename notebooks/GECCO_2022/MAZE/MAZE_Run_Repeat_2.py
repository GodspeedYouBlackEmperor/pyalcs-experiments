import numpy as np
import pandas as pd
import dill
import os

import gym
import gym_maze

from lcs.agents import Agent
from lcs.agents.acs2 import ACS2, Configuration as CFG_ACS2
from lcs.agents.acs2er import ACS2ER, Configuration as CFG_ACS2ER
from lcs.metrics import population_metrics

# Logger
import logging
logging.basicConfig(level=logging.INFO)


MAZE = "Maze5-v0" 
EXPLORE_TRIALS = 5000
EXPLOIT_TRIALS = 1000

# The size of ER replay memory buffer
ER_BUFFER_SIZE = 10000
# The minimum number of samples of ER replay memory buffer to start replying samples (warm-up phase)
ER_BUFFER_MIN_SAMPLES = 1000
# The number of samples to be replayed druing ER phase
ER_SAMPLES_NUMBER_LIST = [1,2,3,5,8,13]



#######

REPEAT_START = 11
REPEAT = 2

EXPERIMENT_NAME = "20" # Please edit if running new experiment to do not override saved results.
DATA_BASE_PATH = "" # CURRENT LOCATION
DATA_PATH = os.path.join(DATA_BASE_PATH, 'MAZE', EXPERIMENT_NAME, MAZE)

# if os.path.isdir(DATA_PATH):
#   raise Exception(f"The experiment with name: '{EXPERIMENT_NAME}' for '{MAZE}' environment was run already.")



def _get_transitions():
    knowledge_env = gym.make(MAZE)
    transitions = knowledge_env.env.get_transitions()
    transitions = list(map(lambda t: [knowledge_env.env.maze.perception(t[0]), t[1], knowledge_env.env.maze.perception(t[2])], transitions))

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

def _maze_metrics(agent, env):
    pop = agent.population
    metrics = {
        'knowledge': _maze_knowledge(pop)
    }
    metrics.update(population_metrics(pop, env))
    
    return metrics


def _save_data(data, path, file_name):
    full_dir_path = os.path.join(DATA_PATH, path)
    full_file_path = os.path.join(full_dir_path, f'{file_name}.dill')
    if not os.path.isdir(full_dir_path):
        os.makedirs(full_dir_path)

    dill.dump(data, open(full_file_path, 'wb'))

def _save_agent_data(agent, data, path, file_name):
    path = os.path.join(type(agent).__name__, path)
    _save_data(data, path, file_name)

def _save_metrics(agent, metrics, path, metrics_name):
    _save_agent_data(agent, metrics, path, f'metrics_{metrics_name}')

def _save_explore_metrics(agent, metrics, path):
    _save_metrics(agent, metrics, path, 'EXPLORE')

def _save_exploit_metrics(agent, metrics, path):
    _save_metrics(agent, metrics, path, 'EXPLOIT')

def _save_population(agent: Agent, path):
    _save_agent_data(agent, agent.get_population(), path, 'population')

def _save_environment(agent, env, path):
    _save_agent_data(agent, env, path, 'env')
    
def _save_experiment_data(agent, env, explore_metrics, exploit_metrics, path):
    _save_explore_metrics(agent, explore_metrics, path)
    _save_exploit_metrics(agent, exploit_metrics, path)
    _save_population(agent, path)
    _save_environment(agent, env, path)


def _run_experiment(agent: Agent, data_path = ''):
    maze = gym.make(MAZE)
    # Explore the environment
    explore_metrics = agent.explore(maze, EXPLORE_TRIALS)
    # Exploit the environment
    exploit_metrics = agent.exploit(maze, EXPLOIT_TRIALS)

    _save_experiment_data(agent, maze, explore_metrics, exploit_metrics, data_path)

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

        _run_experiment(agent, os.path.join(f'm_{er_samples_number}', f'{i}'))

def run_acs2er_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2ER - {er_samples_number}")
        _run_acs2er_experiment(er_samples_number)
        print(f"END - ACS2ER - {er_samples_number}")


run_acs2_experiment()

run_acs2er_experiments()