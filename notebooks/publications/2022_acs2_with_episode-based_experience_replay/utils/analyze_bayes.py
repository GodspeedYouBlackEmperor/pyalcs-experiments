import numpy as np
import pymc3 as pm
import dill
import os
import arviz
from tabulate import tabulate

ACS2_NAME = 'ACS'
ACS2ER_NAME = 'ACS-ER'


def get_label(m):

    if m == '-1':
        return ACS2_NAME
    return m


def print_results(M, results):
    headers = [get_label(m) for m in M]
    headers.insert(0, "")
    to_print_results = []
    for m, result_row in enumerate(results):
        new_row = [
            f'L: {round(r[0], 3)}, R: {round(r[1], 3)}' for r in result_row]
        # new_row = [f'{r}' for r in result_row]
        new_row.insert(0, get_label(M[m]))
        to_print_results.append(new_row)

    formatted_data = tabulate(to_print_results, headers=headers)
    print(formatted_data)


def analyze_models(models, metric_name):
    M = []
    test_results_95 = []
    test_results_999 = []
    for i in range(len(models)):
        M.append(models[i][0])
        row_test_results_95 = []
        row_test_results_999 = []
        test_results_95.append(row_test_results_95)
        test_results_999.append(row_test_results_999)
        for j in range(len(models)):
            id_1, m_1 = models[i]
            id_2, m_2 = models[j]

            data_diff = m_1['mu'] - m_2['mu']
            hdi_950 = arviz.hdi(data_diff, hdi_prob=0.950)

            row_test_results_95.append(hdi_950)

    print(f"\n\n\n {metric_name} \t HDI 95%: \n")
    print_results(M, test_results_95)


if __name__ == '__main__':

    METRICS_NAMES = ['knowledge95', 'steps']
    RESULTS_DIR = 'results'
    EXPS = ['MAZE4']

    for exp in EXPS:
        for metric_name in METRICS_NAMES:
            results_dir_path = os.path.join(RESULTS_DIR, exp, metric_name)
            models = []
            for f in os.listdir(results_dir_path):
                id = f.split('.')[0]
                model = dill.load(
                    open(os.path.join(results_dir_path, f), 'rb'))

                models.append((id, model))

            models = sorted(models, key=lambda x: len(x[0]))
            analyze_models(models, metric_name)
