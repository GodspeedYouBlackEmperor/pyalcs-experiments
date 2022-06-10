import numpy as np
import pymc3 as pm
import dill
import os


def bayes_estimate(data: np.ndarray, draws=100_000):
    # If all values are the same there is no point in calculating statistical model
    if np.all(data == data[0]):
        return {
            'mu': np.array([data[0]]),
            'std': np.array([0])
        }

    mean = data.mean()
    variance = data.std() * 2

    # prior
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=mean, sd=variance)
        std = pm.Uniform('std', 1 / 100, 1000)
        nu = pm.Exponential('nu', 1.0 / 29)  # degrees of freedom

    # posterior
    with model:
        obs = pm.StudentT('obs', mu=mu, lam=1.0 / std **
                          2, nu=nu + 1, observed=data)

    # sample
    with model:
        trace = pm.sample(draws, target_accept=0.95,
                          return_inferencedata=False, progressbar=False)

    return trace


if __name__ == '__main__':

    METRICS_NAMES = ['knowledge95' 'steps']  # ['reward', 'reward_exploit']
    DRAWS = 100000

    EXPS = ['MAZE4']  # SET PROPER ENVS
    DATA_DIR = 'bayes_data'
    RESULTS_DIR = 'results'

    for exp in EXPS:
        for metric_name in METRICS_NAMES:
            data_dir_path = os.path.join(exp, DATA_DIR, metric_name)
            results_dir_path = os.path.join(RESULTS_DIR, exp, metric_name)

            if not os.path.isdir(results_dir_path):
                os.makedirs(results_dir_path)

            for f in os.listdir(data_dir_path):
                label = f.split('.')[0]
                data = np.load(os.path.join(data_dir_path, f))

                model = bayes_estimate(data, DRAWS)
                dill.dump(model, open(os.path.join(
                    results_dir_path, f'{label}.dill'), 'wb'))
