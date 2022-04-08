import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import dill
from scipy import stats
from tabulate import tabulate

KNOWLEDGE_ATTRIBUTE = 'knowledge'
NUMEROSITY_ATTRIBUTE = 'numerosity'
RELIABLE_ATTRIBUTE = 'reliable'
STEPS_ATTRIBUTE = 'steps'

KNOWLEDGE_METRIC = 'knowledge'
CLASSIFIERS_METRIC = 'classifiers'
STEPS_METRIC = 'steps'

ACS2_NAME = 'ACS'
ACS2ER_NAME = 'ACS-ER'


class AnalyzerConfiguration:
    def __init__(self, BASE_NAME, EXP_NAME, ENV_NAME, FRIENDLY_ENV_NAME,
                 DATA_BASE_PATH='',
                 RESULTS_BASE_PATH='',
                 M=[1, 2, 3, 5, 8, 13],
                 SAVE=True,
                 PRIMARY_COLORS=['#FF9634', '#D3D600', '#07B44B',
                                 '#49BDFA', '#316EF9', '#965FFF', '#FE4DFE', '#FF63A5'],
                 SECONDARY_COLORS=['#FFCD9F', '#E8EB93', '#87CAA2',
                                   '#C0EAFF', '#C2D5FF', '#DAC7FF', '#FDAAFD', '#F6C5DA'],
                 PRIMARY_COLOR='#FF2E2E',
                 SECONDARY_COLOR='#FF9898',
                 PRIMARY_MARKERS=['*', 'x', '4', '^'],
                 SECONDARY_MARKERS=['*', 'x', '4', '^'],
                 PRIMARY_MARKER='o',
                 SECONDARY_MARKER='o',
                 FIG_SIZE=[13, 10],
                 LINE_WIDTH=2,
                 TITLE_TEXT_SIZE=24,
                 AXIS_TEXT_SIZE=18,
                 LEGEND_TEXT_SIZE=16,
                 TICKS_TEXT_SIZE=12) -> None:
        self.M = M
        self.BASE_NAME = BASE_NAME
        self.EXP_NAME = EXP_NAME
        self.ENV_NAME = ENV_NAME
        self.FRIENDLY_ENV_NAME = FRIENDLY_ENV_NAME
        self.DATA_BASE_PATH = DATA_BASE_PATH
        self.DATA_PATH = os.path.join(
            DATA_BASE_PATH, BASE_NAME, EXP_NAME, ENV_NAME)
        self.RESULTS_PATH = os.path.join(
            RESULTS_BASE_PATH, 'RESULTS', BASE_NAME, ENV_NAME, EXP_NAME)
        self.SAVE = SAVE
        self.PRIMARY_COLORS = PRIMARY_COLORS
        self.SECONDARY_COLORS = SECONDARY_COLORS
        self.PRIMARY_COLOR = PRIMARY_COLOR
        self.SECONDARY_COLOR = SECONDARY_COLOR
        self.PRIMARY_MARKERS = PRIMARY_MARKERS
        self.SECONDARY_MARKERS = SECONDARY_MARKERS
        self.PRIMARY_MARKER = PRIMARY_MARKER
        self.SECONDARY_MARKER = SECONDARY_MARKER
        self.FIG_SIZE = FIG_SIZE
        self.LINE_WIDTH = LINE_WIDTH
        self.TITLE_TEXT_SIZE = TITLE_TEXT_SIZE
        self.AXIS_TEXT_SIZE = AXIS_TEXT_SIZE
        self.LEGEND_TEXT_SIZE = LEGEND_TEXT_SIZE
        self.TICKS_TEXT_SIZE = TICKS_TEXT_SIZE


class Analyzer:
    def __init__(self, acs2_data, acs2er_data, config: AnalyzerConfiguration, acs2_data_full=None, acs2er_data_full=None) -> None:
        self.acs2_data = acs2_data
        self.acs2er_data = acs2er_data
        self.config = config
        self.acs2_data_full = acs2_data_full
        self.acs2er_data_full = acs2er_data_full

        self.metrics = {
            KNOWLEDGE_METRIC: ['Knowledge', 'TRIAL', 'KNOWLEDGE [%]', 'knowledge'],
            CLASSIFIERS_METRIC: ['Classifiers numerosity (num) and reliable (rel)', 'TRIAL', 'CLASSIFIERS', 'classifiers'],
            STEPS_METRIC: ['Steps', 'TRIAL', 'STEPS', 'steps']
        }

        if not os.path.isdir(self.config.RESULTS_PATH):
            os.makedirs(self.config.RESULTS_PATH)

        plt.rcParams['figure.figsize'] = self.config.FIG_SIZE

    def plot_knowledge(self, explore_avg_win=0, exploit_avg_win=0, width=0):
        def __plot_func(metrics_list, width):
            (title, x_lable, y_label, name) = self.metrics[KNOWLEDGE_METRIC]
            self.__plot_single_metric(metrics_list, title, KNOWLEDGE_ATTRIBUTE,
                                      x_lable, y_label, name, explore_avg_win, exploit_avg_win, width)
        self.__plot_metric(self.__get_metric_record_single,
                           __plot_func, width or self.config.LINE_WIDTH)

    def plot_classifiers(self, explore_avg_win=0, exploit_avg_win=0, width=0, use_markers=False, density=100):
        def __plot_func(metrics_list, width):
            (title, x_label, y_label, name) = self.metrics[CLASSIFIERS_METRIC]
            if use_markers:
                def __plot_attribute_func_num(x, width: float):
                    (label, _, secondary_color, _, secondary_marker), metric = x
                    self.__plot_attribute_marker(
                        metric, NUMEROSITY_ATTRIBUTE, f'{label} (num)', secondary_color, secondary_color, width, secondary_marker, density)

                def __plot_attribute_func_rel(x, width: float):
                    (label, primary_color, _, primary_marker, _), metric = x
                    self.__plot_attribute_marker(
                        metric, RELIABLE_ATTRIBUTE, f'{label} (rel)', primary_color, primary_color, width, primary_marker, density)
                self.__plot([__plot_attribute_func_num, __plot_attribute_func_rel],
                            metrics_list, title, x_label, y_label, name, width)
            elif not explore_avg_win:
                def __plot_attribute_func_num(x, width: float):
                    (label, _, secondary_color, _, _), metric = x
                    self.__plot_attribute_standard(
                        metric, NUMEROSITY_ATTRIBUTE, f'{label} (num)', secondary_color, secondary_color, width)

                def __plot_attribute_func_rel(x, width: float):
                    (label, primary_color, _, _, _), metric = x
                    self.__plot_attribute_standard(
                        metric, RELIABLE_ATTRIBUTE, f'{label} (rel)', primary_color, primary_color, width)
                self.__plot([__plot_attribute_func_num, __plot_attribute_func_rel],
                            metrics_list, title, x_label, y_label, name, width)
            else:
                def __plot_attribute_func_num(x, width: float):
                    (label, _, secondary_color, _, _), metric = x
                    self.__plot_attribute_moving_average(
                        metric, NUMEROSITY_ATTRIBUTE, explore_avg_win, exploit_avg_win, f'{label} (num)', secondary_color, secondary_color, width)

                def __plot_attribute_func_rel(x, width: float):
                    (label, primary_color, _, _, _), metric = x
                    self.__plot_attribute_moving_average(
                        metric, RELIABLE_ATTRIBUTE, explore_avg_win, exploit_avg_win, f'{label} (rel)', primary_color, primary_color, width)
                self.__plot([__plot_attribute_func_num, __plot_attribute_func_rel],
                            metrics_list, title, x_label, y_label, name, width)
        self.__plot_metric(self.__get_metric_record_double,
                           __plot_func, width or self.config.LINE_WIDTH, True)

    def plot_steps(self, explore_avg_win=0, exploit_avg_win=0, width=0):
        def __plot_func(metrics_list, width):
            (title, x_lable, y_label, name) = self.metrics[STEPS_METRIC]
            self.__plot_single_metric(metrics_list, title, STEPS_ATTRIBUTE,
                                      x_lable, y_label, name, explore_avg_win, exploit_avg_win, width)
        self.__plot_metric(self.__get_metric_record_single,
                           __plot_func, width or self.config.LINE_WIDTH)

    def __get_metrics(self, metrics, phase):
        return metrics.query(f"phase == '{phase}'")

    def __get_explore_metrics(self, metrics):
        return self.__get_metrics(metrics, 'explore')

    def __get_exploit_metrics(self, metrics):
        return self.__get_metrics(metrics, 'exploit')

    def __plot_attribute_moving_average(self, metrics, attribute_name: str, explore_moving_avg_win: int, exploit_moving_avg_win: int, label: str, explore_color: str, exploit_color: str, width: float):
        def __metric_func(metrics, win):
            return metrics.rolling(win, closed='both').mean()

        self.__plot_attribute(lambda m: __metric_func(m, explore_moving_avg_win), lambda m: __metric_func(m, exploit_moving_avg_win).shift(
            1 - exploit_moving_avg_win), metrics, attribute_name, label, explore_color, exploit_color, width)

    def __plot_attribute_standard(self, metrics, attribute_name: str, label: str, explore_color: str, exploit_color: str, width: float):
        def __metric_func(metrics):
            return metrics

        self.__plot_attribute(__metric_func, __metric_func, metrics,
                              attribute_name, label, explore_color, exploit_color, width)

    def __plot_attribute(self, explore_metrics_func, exploit_metrics_func, metrics, attribute_name: str, label: str, explore_color: str, exploit_color: str, width: float):
        explore = self.__get_explore_metrics(metrics)
        exploit = self.__get_exploit_metrics(metrics)

        x_axis_explore = range(1, len(explore) + 1)
        x_axis_exploit = range(
            len(explore) + 1, len(explore) + len(exploit) + 1)

        plt.plot(x_axis_explore, explore_metrics_func(
            explore[attribute_name]), c=explore_color, label=label, linewidth=width)
        plt.plot(x_axis_exploit, exploit_metrics_func(
            exploit[attribute_name]), c=exploit_color, linewidth=width)

    def __plot_attribute_marker(self, metrics, attribute_name: str, label: str, explore_color: str, exploit_color: str, width: float, marker: str, density: int):
        explore = self.__get_explore_metrics(metrics)
        exploit = self.__get_exploit_metrics(metrics)

        x_axis_explore = range(1, len(explore) + 1)
        x_axis_exploit = range(
            len(explore) + 1, len(explore) + len(exploit) + 1)

        plt.plot(x_axis_explore,
                 explore[attribute_name], c=explore_color, label=label, linewidth=width, marker=marker, markevery=density)
        plt.plot(x_axis_exploit,
                 exploit[attribute_name], c=exploit_color, linewidth=width, marker=marker, markevery=density)

    def __plot_moving_average(self, metrics_list, title, attribute, explore_moving_avg_win: int, exploit_moving_avg_win: int, x_label, y_label, name, width):
        def __plot_attribute_func(x, width: float):
            (label, explore_color, exploit_color, _, _), metric = x
            self.__plot_attribute_moving_average(
                metric, attribute, explore_moving_avg_win, exploit_moving_avg_win, label, explore_color, exploit_color, width)

        self.__plot([__plot_attribute_func], metrics_list,
                    title, x_label, y_label, name, width)

    def __plot_standard(self, metrics_list, title, attribute, x_label, y_label, name, width):
        def __plot_attribute_func(x, width: float):
            (label, explore_color, exploit_color, _, _), metric = x
            self.__plot_attribute_standard(
                metric, attribute, label, explore_color, exploit_color, width)

        self.__plot([__plot_attribute_func], metrics_list,
                    title, x_label, y_label, name, width)

    def __plot(self, plot_metric_funcs, metrics_list, title, x_label, y_label, name, width):
        plt.close()
        plt.title(f'{title} - {self.config.FRIENDLY_ENV_NAME}',
                  fontsize=self.config.TITLE_TEXT_SIZE)

        for x in metrics_list:
            for f in plot_metric_funcs:
                f(x, width)

        plt.axvline(x=len(self.__get_explore_metrics(
            metrics_list[0][1])), c='black', linestyle='dashed')

        plt.legend(fontsize=self.config.LEGEND_TEXT_SIZE)
        plt.xlabel(x_label, fontsize=self.config.AXIS_TEXT_SIZE)
        plt.ylabel(y_label, fontsize=self.config.AXIS_TEXT_SIZE)
        plt.xticks(fontsize=self.config.TICKS_TEXT_SIZE)
        plt.yticks(fontsize=self.config.TICKS_TEXT_SIZE)
        if(self.config.SAVE):
            plt.savefig(os.path.join(self.config.RESULTS_PATH, f"{name}.png"))
        plt.show()

    def __get_metric_record_single(self, index, data):
        m, (metric, _, _) = data

        return (f'{ACS2ER_NAME} m-{m}', self.config.PRIMARY_COLORS[index], self.config.PRIMARY_COLORS[index], self.config.PRIMARY_MARKERS[index], self.config.PRIMARY_MARKERS[index]), metric

    def __get_metric_record_double(self, index, data):
        m, (metric, _, _) = data

        return (f'{ACS2ER_NAME} m-{m}', self.config.PRIMARY_COLORS[index], self.config.SECONDARY_COLORS[index], self.config.PRIMARY_MARKERS[index], self.config.SECONDARY_MARKERS[index]), metric

    def __plot_metric(self, get_metric_record_func, plot_func, width, is_double=False):
        acs2_metric, _, _ = self.acs2_data

        metrics_list = list(map(lambda d: get_metric_record_func(
            d[0], d[1]), enumerate(self.acs2er_data)))
        metrics_list.insert(0, ((ACS2_NAME, self.config.PRIMARY_COLOR,
                                 self.config.SECONDARY_COLOR if is_double else self.config.PRIMARY_COLOR,
                                 self.config.PRIMARY_MARKER,
                                 self.config.SECONDARY_MARKER if is_double else self.config.PRIMARY_MARKER,), acs2_metric))

        plot_func(metrics_list, width)

    def __plot_single_metric(self, metrics_list, title, attribute, x_label, y_label, name, explore_avg_win, exploit_avg_win, width):
        if not explore_avg_win:
            self.__plot_standard(metrics_list, title,
                                 attribute, x_label, y_label, name, width)
        else:
            self.__plot_moving_average(
                metrics_list, title, attribute, explore_avg_win, exploit_avg_win, x_label, y_label, name, width)


################################### STATISCTIS ###############################


    def __get_knowledge_above_threshold_trial(self, metrics, threshold):
        explore = self.__get_explore_metrics(metrics)
        knowledge_completed = explore.query(
            f"{KNOWLEDGE_ATTRIBUTE} > {threshold}")

        return knowledge_completed.index[0]

    def __get_metric_information(self, info_func):
        info = []
        acs2_info = list(map(info_func, self.acs2_data_full))

        info.append((-1, acs2_info))
        for m, data_full in self.acs2er_data_full:
            acs2er_info_m = list(map(info_func, data_full))
            info.append((m, acs2er_info_m))

        return info

    def compare_knowledge_above_threshold_welch_test(self, threshold):
        info = self.__get_metric_information(
            lambda met: self.__get_knowledge_above_threshold_trial(met, threshold))

        test_results = []

        for i in range(len(info)):
            row_test_results = []
            for j in range(len(info)):
                result = stats.ttest_ind(info[i][1], info[j][1])
                row_test_results.append(result)

            test_results.append(row_test_results)

        self.__print_results([i[0] for i in info], test_results)

    def __print_results(self, M, results):
        headers = [self.__get_label(m) for m in M]
        headers.insert(0, "")
        print_results = []
        for index, result_row in enumerate(results):
            new_row = [
                f's: {round(r.statistic, 3)}, p: {round(r.pvalue, 3)}' for r in result_row]
            new_row.insert(0, self.__get_label(M[index]))
            print_results.append(new_row)

        print(tabulate(print_results, headers=headers))

    def __get_label(self, m):
        if m < 0:
            return ACS2_NAME

        return f'{ACS2ER_NAME} m-{m}'

    def print_knowledge_above_threshold(self, threshold):
        def __print(label, trial):
            print(f"{label}: Knowledge over {threshold} % at trial {trial}")

        acs2_metric, _, _ = self.acs2_data

        __print(ACS2_NAME, self.__get_knowledge_above_threshold_trial(
            acs2_metric, threshold))

        for m, (acs2er_metric, _, _) in self.acs2er_data:
            __print(f'{ACS2ER_NAME} m-{m}',
                    self.__get_knowledge_above_threshold_trial(acs2er_metric, threshold))

    def print_knowledge_above_threshold_full(self, threshold):
        def __print(label, trial):
            print(f"{label}: Knowledge over {threshold} % at trial {trial}")

        __print(ACS2_NAME, list(map(lambda met: self.__get_knowledge_above_threshold_trial(
            met, threshold), self.acs2_data_full)))

        for m, data_full in self.acs2er_data_full:
            __print(f'{ACS2ER_NAME} m-{m}',
                    list(map(lambda met: self.__get_knowledge_above_threshold_trial(met, threshold), data_full)))
