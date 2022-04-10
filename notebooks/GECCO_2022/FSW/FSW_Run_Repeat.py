import numpy as np
import pandas as pd
import dill
import os
import itertools

import gym
import gym_fsw

from lcs import Perception
from lcs.agents import Agent
from lcs.agents.acs2 import ACS2, Configuration as CFG_ACS2
from lcs.agents.acs2er import ACS2ER, Configuration as CFG_ACS2ER
from lcs.metrics import population_metrics

# Logger
import logging
logging.basicConfig(level=logging.INFO)


FSW_SIZE = 40  # 5 | 10 | 20 | 40
FSW = f'fsw-{FSW_SIZE}-v0'
EXPLORE_TRIALS = 200
EXPLOIT_TRIALS = 100
METRICS_FREQUENCY = 1

# The size of ER replay memory buffer
ER_BUFFER_SIZE = 10000
# The minimum number of samples of ER replay memory buffer to start replying samples (warm-up phase)
ER_BUFFER_MIN_SAMPLES = 100
# The number of samples to be replayed druing ER phase
ER_SAMPLES_NUMBER_LIST = [1,3,5,13]


REPEAT_START = 1
REPEAT = 6

EXPERIMENT_NAME = "840" # Please edit if running new experiment to do not override saved results.
DATA_BASE_PATH = "" # CURRENT LOCATION
DATA_PATH = os.path.join(DATA_BASE_PATH, 'FSW', EXPERIMENT_NAME, FSW)



class FSWObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return observation,

env = FSWObservationWrapper(gym.make(FSW))


def fsw_metrics(agent, env):
    pop = agent.population
    metrics = {
        # 'knowledge': _fsw_knowledge(pop, env)
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
    cor = FSWObservationWrapper(gym.make(FSW))
    # Explore the environment
    explore_metrics = agent.explore(cor, EXPLORE_TRIALS)
    # Exploit the environment
    exploit_metrics = agent.exploit(cor, EXPLOIT_TRIALS)

    _save_experiment_data(agent, cor, explore_metrics, exploit_metrics, data_path)

def run_acs2_experiment():
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent 
        cfg = CFG_ACS2(
            classifier_length=1,
            number_of_possible_actions=2,
            metrics_trial_frequency=METRICS_FREQUENCY,
            user_metrics_collector_fcn=fsw_metrics)
        agent = ACS2(cfg)

        _run_experiment(agent, f'{i}')

def _run_acs2er_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent 
        cfg = CFG_ACS2ER(    
            classifier_length=1,
            number_of_possible_actions=2,
            metrics_trial_frequency=METRICS_FREQUENCY,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            user_metrics_collector_fcn=fsw_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_{er_samples_number}', f'{i}'))

def run_acs2er_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2ER - {er_samples_number}")
        _run_acs2er_experiment(er_samples_number)
        print(f"END - ACS2ER - {er_samples_number}")


run_acs2_experiment()
run_acs2er_experiments()