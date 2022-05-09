import os
import dill
from lcs.agents import Agent

class Runner:
    def __init__(self, BASE_NAME, EXP_NAME, ENV_NAME, DATA_BASE_PATH='') -> None:
        self.DATA_PATH = os.path.join(DATA_BASE_PATH, BASE_NAME, EXP_NAME, ENV_NAME)

    def run_experiment(self, agent: Agent, env, explore_trials, exploit_trials, data_path = ''):
        # Explore the environment
        explore_metrics = agent.explore(env, explore_trials)
        # Exploit the environment
        exploit_metrics = agent.exploit(env, exploit_trials)

        self.__save_experiment_data(agent, env, explore_metrics, exploit_metrics, data_path)

    def __save_data(self, data, path, file_name):
        full_dir_path = os.path.join(self.DATA_PATH, path)
        full_file_path = os.path.join(full_dir_path, f'{file_name}.dill')
        if not os.path.isdir(full_dir_path):
            os.makedirs(full_dir_path)

        dill.dump(data, open(full_file_path, 'wb'))

    def __save_agent_data(self, agent, data, path, file_name):
        path = os.path.join(type(agent).__name__, path)
        self.__save_data(data, path, file_name)

    def __save_metrics(self, agent, metrics, path, metrics_name):
        self.__save_agent_data(agent, metrics, path, f'metrics_{metrics_name}')

    def __save_explore_metrics(self, agent, metrics, path):
        self.__save_metrics(agent, metrics, path, 'EXPLORE')

    def __save_exploit_metrics(self, agent, metrics, path):
        self.__save_metrics(agent, metrics, path, 'EXPLOIT')

    def __save_population(self, agent: Agent, path):
        self.__save_agent_data(agent, agent.get_population(), path, 'population')

    def __save_environment(self, agent, env, path):
        self.__save_agent_data(agent, env, path, 'env')
        
    def __save_experiment_data(self, agent, env, explore_metrics, exploit_metrics, path):
        self.__save_explore_metrics(agent, explore_metrics, path)
        self.__save_exploit_metrics(agent, exploit_metrics, path)
        self.__save_population(agent, path)
        self.__save_environment(agent, env, path)