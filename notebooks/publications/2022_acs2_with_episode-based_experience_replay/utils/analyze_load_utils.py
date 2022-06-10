import os
import dill
import pandas as pd


class Loader:
    def __init__(self, BASE_NAME, EXP_NAME, ENV_NAME, DATA_BASE_PATH='', M=[1, 2, 3, 5, 8, 13], LOAD_DATA_FULL=True, MODEL_NAME="ACS2", ER_MODEL_NAME="ACS2ER") -> None:
        self.DATA_PATH = os.path.join(
            DATA_BASE_PATH, BASE_NAME, EXP_NAME, ENV_NAME)
        self.M = M
        self.LOAD_DATA_FULL = LOAD_DATA_FULL
        self.MODEL_NAME = MODEL_NAME
        self.ER_MODEL_NAME = ER_MODEL_NAME

    def load(self):
        def _get_acs2_experiment_data():
            explore_metrics, exploit_metrics, population, env = self._load_acs2_experiment_data()
            metrics_average, full_metrics = self._parse_metrics_to_df(
                explore_metrics, exploit_metrics)

            return (metrics_average, population[0], env[0]), full_metrics

        def _get_acs2er_experiments_data():
            data = []
            data_full = []
            for m, (explore_metrics, exploit_metrics, population, env) in self._load_acs2er_experiments_data(self.M):
                metrics_average, full_metrics = self._parse_metrics_to_df(
                    explore_metrics, exploit_metrics)

                data.append((m, (metrics_average, population[0], env[0])))
                data_full.append((m, full_metrics))

            return data, data_full

        acs2_data, acs2_full_data = _get_acs2_experiment_data()
        acs2er_data, acs2er_full_data = _get_acs2er_experiments_data()
        return (acs2_data, acs2er_data, acs2_full_data, acs2er_full_data)

    def _load_data(self, path, file_name):
        full_dir_path = os.path.join(self.DATA_PATH, path)
        data_list = []
        for d in os.listdir(full_dir_path):
            full_file_path = os.path.join(
                full_dir_path, d, f'{file_name}.dill')
            data_list.append(dill.load(open(full_file_path, 'rb')))

        return data_list

    def _load_agent_data(self, agent_name, path, file_name):
        path = os.path.join(agent_name, path)
        return self._load_data(path, file_name)

    def _load_metrics(self, agent_name, path, metrics_name):
        return self._load_agent_data(agent_name, path, f'metrics_{metrics_name}')

    def _load_explore_metrics(self, agent_name, path):
        return self._load_metrics(agent_name, path, 'EXPLORE')

    def _load_exploit_metrics(self, agent_name, path):
        return self._load_metrics(agent_name, path, 'EXPLOIT')

    def _load_population(self, agent_name, path):
        return self._load_agent_data(agent_name, path, 'population')

    def _load_environment(self, agent, path):
        return self._load_agent_data(agent, path, 'env')

    def _load_experiment_data(self, agent_name, path):
        explore_metrics = self._load_explore_metrics(agent_name, path)
        exploit_metrics = self._load_exploit_metrics(agent_name, path)
        population = self._load_population(agent_name, path)
        env = [None]  # self._load_environment(agent_name, path)

        return (explore_metrics, exploit_metrics, population, env)

    def _load_acs2_experiment_data(self):
        return self._load_experiment_data(self.MODEL_NAME, '')

    def _load_acs2er_experiment_data(self, er_samples_number):
        return self._load_experiment_data(self.ER_MODEL_NAME, f'm_{er_samples_number}')

    def _load_acs2er_experiments_data(self, M):
        return list(map(lambda m: (m, self._load_acs2er_experiment_data(m)), M))

    def _parse_metrics_to_df(self, explore_metrics, exploit_metrics):
        def extract_details(row):
            row['steps'] = row['steps_in_trial']
            return row

        explore_metrics_list = []
        exploit_metrics_list = []
        full_metrics_list = []
        for i in range(len(explore_metrics)):
            explore_df = pd.DataFrame(explore_metrics[i])
            exploit_df = pd.DataFrame(exploit_metrics[i])
            explore_df = explore_df.apply(extract_details, axis=1)
            exploit_df = exploit_df.apply(extract_details, axis=1)

            explore_metrics_list.append(explore_df)
            exploit_metrics_list.append(exploit_df)

            if(self.LOAD_DATA_FULL):
                explore_df = pd.DataFrame(explore_metrics[i])
                exploit_df = pd.DataFrame(exploit_metrics[i])
                explore_df = explore_df.apply(extract_details, axis=1)
                exploit_df = exploit_df.apply(extract_details, axis=1)

                explore_df['phase'] = 'explore'
                exploit_df['phase'] = 'exploit'

                exploit_df['trial'] = exploit_df.apply(
                    lambda r: r['trial']+len(explore_df), axis=1)

                df = pd.concat([explore_df, exploit_df])
                df.set_index('trial', inplace=True)
                full_metrics_list.append(df)

        explore_df = pd.concat(explore_metrics_list)
        explore_df = explore_df.groupby(explore_df.index).mean()

        exploit_df = pd.concat(exploit_metrics_list)
        exploit_df = exploit_df.groupby(exploit_df.index).mean()

        explore_df['phase'] = 'explore'
        exploit_df['phase'] = 'exploit'

        exploit_df['trial'] = exploit_df.apply(
            lambda r: r['trial']+len(explore_df), axis=1)

        df = pd.concat([explore_df, exploit_df])
        df.set_index('trial', inplace=True)

        return df, full_metrics_list
