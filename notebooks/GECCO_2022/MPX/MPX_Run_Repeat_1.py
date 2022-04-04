import numpy as np
import pandas as pd
import dill
import os
import itertools

import gym
import gym_multiplexer
from gym_multiplexer.utils import get_correct_answer

from lcs import Perception
from lcs.agents import Agent
from lcs.agents.acs2 import ACS2, Configuration as CFG_ACS2
from lcs.agents.acs2er import ACS2ER, Configuration as CFG_ACS2ER
from lcs.metrics import population_metrics

# Logger
import logging
logging.basicConfig(level=logging.INFO)


BITS = 11 # 6 | 11 | 20 | 37
MPX = f'boolean-multiplexer-{BITS}bit-v0'
EXPLORE_TRIALS = 10000
EXPLOIT_TRIALS = 1000
METRICS_FREQUENCY = 10
KNOWLEDGE_STATE_SAMPLES = 1000 # applies only when 20 or 37 bits, otherwise all possible states verified

# The size of ER replay memory buffer
ER_BUFFER_SIZE = 10000
# The minimum number of samples of ER replay memory buffer to start replying samples (warm-up phase)
ER_BUFFER_MIN_SAMPLES = 100
# The number of samples to be replayed druing ER phase
ER_SAMPLES_NUMBER_LIST = [1,2,3,5,8,13]



#######

REPEAT_START = 1
REPEAT = 2

EXPERIMENT_NAME = "311" # Please edit if running new experiment to do not override saved results.
DATA_BASE_PATH = "" # CURRENT LOCATION
DATA_PATH = os.path.join(DATA_BASE_PATH, 'MPX', EXPERIMENT_NAME, MPX)

# if os.path.isdir(DATA_PATH):
#   raise Exception(f"The experiment with name: '{EXPERIMENT_NAME}' for '{MAZE}' environment was run already.")


class MpxObservationWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        return [str(x) for x in observation]
        
knowledge_env = MpxObservationWrapper(gym.make(MPX))

def get_transitions(states):
    transitions = list(map(lambda s: 
            (Perception([str(float(x)) for x in s] + ['0.0']), 
            get_correct_answer(list(s) + [0], knowledge_env.env.env.control_bits), 
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

if BITS == 6 or BITS == 11: # Verify all 
    def get_all_transitions():
        states = list(itertools.product([0, 1], repeat=BITS))
        return get_transitions(states)

    TRANSITIONS = get_all_transitions()
    TRANSITIONS_LENGTH = len(TRANSITIONS)
    
    def mpx_knowledge(population) -> float:
        return _mpx_knowledge(population, TRANSITIONS, TRANSITIONS_LENGTH)

elif BITS == 20 or BITS == 37: # Verify samples
    def get_sampled_transitions():
        states = np.random.randint(2, size=(KNOWLEDGE_STATE_SAMPLES, BITS))
        return get_transitions(states)

    def mpx_knowledge(population) -> float:
        return _mpx_knowledge(population, get_sampled_transitions(), KNOWLEDGE_STATE_SAMPLES)
else:
    raise Exception(f'Unsupported BITS number: {BITS}')
    

def mpx_metrics(agent, env):
    metrics = {
        "knowledge": mpx_knowledge(agent.population)
    }
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
    mpx = MpxObservationWrapper(gym.make(MPX))
    # Explore the environment
    explore_metrics = agent.explore(mpx, EXPLORE_TRIALS)
    # Exploit the environment
    exploit_metrics = agent.exploit(mpx, EXPLOIT_TRIALS)

    _save_experiment_data(agent, mpx, explore_metrics, exploit_metrics, data_path)

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

        _run_experiment(agent, os.path.join(f'm_{er_samples_number}', f'{i}'))

def run_acs2er_experiments():
    for er_samples_number in ER_SAMPLES_NUMBER_LIST:
        print(f"START - ACS2ER - {er_samples_number}")
        _run_acs2er_experiment(er_samples_number)
        print(f"END - ACS2ER - {er_samples_number}")


run_acs2_experiment()
run_acs2er_experiments()
