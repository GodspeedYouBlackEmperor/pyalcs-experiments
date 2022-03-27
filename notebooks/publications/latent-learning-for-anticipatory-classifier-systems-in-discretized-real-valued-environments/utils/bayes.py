import arviz
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
from typing import List


def extract_steps(df, agent) -> List:
    return df.query(f"agent == '{agent}'")['trial_steps'].astype('int').tolist()


def inference(rope, hdi):
    if rope[0] < hdi[0] and rope[1] > hdi[1]:
        return 'Null hypothesis accepted (HDI inside ROPE)'
    elif min(rope) > max(hdi) or max(rope) < min(hdi):
        return 'Null hypothesis rejected (HDI and ROPE are disjoint)'
    else:
        return 'Null hypothesis cannot be determined'


def bayes_estimate(data: np.ndarray, draws=3000):
    mean = data.mean()
    variance = data.std() * 2

    # prior
    with pm.Model() as model:
        mu = pm.Normal('mu', mu=mean, sd=variance)
        std = pm.Uniform('std', 1/100, 1000)
        nu = pm.Exponential('nu', 1.0 / 29)  # degrees of freedom

    # posterior
    with model:
        obs = pm.StudentT('obs', mu=mu, lam=1.0/std**2, nu=nu+1, observed=data)

    # sample
    with model:
        trace = pm.sample(draws, target_accept=0.95, return_inferencedata=False, progressbar=False)

    return trace


def compare_two_classifiers(df, cl1, cl2, draws=100_000):
    data_1 = extract_steps(df, cl1)
    data_2 = extract_steps(df, cl2)

    # for hierarchical model compute the pooled mean and variance
    pooled_mean = np.r_[data_1, data_2].mean()
    pooled_std = np.r_[data_1, data_2].std()
    variance = pooled_std * 2

    # prior distributions
    with pm.Model() as model:
        mu_1 = pm.Normal('mu_1', mu=pooled_mean, sd=variance)
        mu_2 = pm.Normal('mu_2', mu=pooled_mean, sd=variance)
        std_1 = pm.Uniform('std_1', 1 / 100, 1000)
        std_2 = pm.Uniform('std_2', 1 / 100, 1000)
        nu = pm.Exponential('nu', 1.0 / 29)  # degrees of freedom

    # posterior
    with model:
        obs_1 = pm.StudentT('obs_1', mu=mu_1, lam=1.0 / std_1 ** 2, nu=nu + 1, observed=data_1)
        obs_2 = pm.StudentT('obs_2', mu=mu_2, lam=1.0 / std_2 ** 2, nu=nu + 1, observed=data_2)

    # sample
    with model:
        trace = pm.sample(draws, target_accept=0.95, return_inferencedata=False, progressbar=False)

    return trace


def visualize(trace, cl1='cl1', cl2='cl2', bins=50, rope=[-1, 1], hdi_prob=0.95):
    plt.suptitle(f'Comparison between {cl1} and {cl2}', size=16)

    plt.subplot(311)
    plt.hist(trace['mu_1'], bins=bins, label=f'Posterior of {cl1}')
    plt.hist(trace['mu_2'], bins=bins, label=f'Posterior of {cl2}')
    plt.title('Posterior of performance means')
    plt.legend(loc='upper right')

    plt.subplot(312)
    delta = trace['mu_1'] - trace['mu_2']
    hdi = arviz.hdi(delta, hdi_prob=0.95)

    plt.hist(delta, bins=bins, label=f'{cl1} - {cl2}')
    plt.axvline(x=rope[0], linestyle='--', color='red', label='ROPE')
    plt.axvline(x=rope[1], linestyle='--', color='red')
    plt.axvline(x=hdi[0], linestyle='--', color='blue', label=f'{int(hdi_prob * 100)}% HDI')
    plt.axvline(x=hdi[1], linestyle='--', color='blue')
    plt.title(f'Posterior distribution of the mean difference (avg: {delta.mean():.1f})')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.91)

    plt.show()

    return rope, hdi
