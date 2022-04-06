import numpy as np
import pandas as pd
import dill
import os
import itertools
import math

import gym

from lcs import Perception
from lcs.agents import Agent
from lcs.agents.acs2 import ACS2, Configuration as CFG_ACS2
from lcs.agents.acs2er import ACS2ER, Configuration as CFG_ACS2ER
from lcs.metrics import population_metrics

# Logger
import logging
logging.basicConfig(level=logging.INFO)



CART_POLE = "CartPole-v0" 
EXPLORE_TRIALS = 5000
EXPLOIT_TRIALS = 2000
METRICS_FREQUENCY = 1

# The size of ER replay memory buffer
ER_BUFFER_SIZE = 10000
# The minimum number of samples of ER replay memory buffer to start replying samples (warm-up phase)
ER_BUFFER_MIN_SAMPLES = 1000
# The number of samples to be replayed druing ER phase
ER_SAMPLES_NUMBER_LIST = [1,2,3,5,8,13]



#######

REPEAT_START = 1
REPEAT = 2

EXPERIMENT_NAME = "5" # Please edit if running new experiment to do not override saved results.
DATA_BASE_PATH = "" # CURRENT LOCATION
DATA_PATH = os.path.join(DATA_BASE_PATH, 'CP', EXPERIMENT_NAME, CART_POLE)

# if os.path.isdir(DATA_PATH):
#   raise Exception(f"The experiment with name: '{EXPERIMENT_NAME}' for '{MAZE}' environment was run already.")


env = gym.make(CART_POLE)

_high = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
_low = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]

# working (1, 1, 10, 14)
buckets = (1, 1, 6, 6)

class CartPoleObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        ratios = [(obs[i] + abs(_low[i])) / (_high[i] - _low[i]) for i in range(len(obs))]
        new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return [str(o) for o in new_obs]

def avg_fitness(pop):
    return np.mean([cl.fitness for cl in pop if cl.is_reliable()])

def cp_metrics(agent, env):
    metrics = {}
    metrics['avg_fitness'] = avg_fitness(agent.population)
    metrics.update(population_metrics(agent.population, env))

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
    cp = CartPoleObservationWrapper(gym.make(CART_POLE))
    # Explore the environment
    explore_metrics = agent.explore(cp, EXPLORE_TRIALS)
    # Exploit the environment
    exploit_metrics = agent.exploit(cp, EXPLOIT_TRIALS)

    _save_experiment_data(agent, cp, explore_metrics, exploit_metrics, data_path)

def run_acs2_experiment():
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent 
        cfg = CFG_ACS2(
            classifier_length=4,
            number_of_possible_actions=2,
            epsilon=0.95,
            biased_exploration=0.5,
            beta=0.05,
            gamma=0.99,
            theta_exp=50,
            theta_ga=50,
            mu=0.03,
            u_max=4,
            metrics_trial_frequency=METRICS_FREQUENCY,
            user_metrics_collector_fcn=cp_metrics)
        agent = ACS2(cfg)

        _run_experiment(agent, f'{i}')

def _run_acs2er_experiment(er_samples_number: int):
    for i in range(REPEAT_START, REPEAT_START + REPEAT):
        # Create agent 
        cfg = CFG_ACS2ER(    
            classifier_length=4,
            number_of_possible_actions=2,
            epsilon=0.95,
            biased_exploration=0.5,
            beta=0.05,
            gamma=0.99,
            theta_exp=50,
            theta_ga=50,
            mu=0.03,
            u_max=4,
            metrics_trial_frequency=METRICS_FREQUENCY,
            er_buffer_size=ER_BUFFER_SIZE,
            er_min_samples=ER_BUFFER_MIN_SAMPLES,
            er_samples_number=er_samples_number,
            user_metrics_collector_fcn=cp_metrics)
        agent = ACS2ER(cfg)

        _run_experiment(agent, os.path.join(f'm_{er_samples_number}', f'{i}'))

def run_acs2er_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2ER - {er_samples_number}")
        _run_acs2er_experiment(er_samples_number)
        print(f"END - ACS2ER - {er_samples_number}")


run_acs2_experiment()
run_acs2er_experiments()