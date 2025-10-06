import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from bandit import BernoulliBandit
from solvers import Solver, EpsilonGreedy, UCB1, BayesianUCB, ThompsonSampling

def plot_results(solvers, solver_names, fig_name):

    b = solvers[0].bandit

    fig = plt.figure(figsize=(14,4))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    #Time vs Regret
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)),s.regrets, label=solver_names[i])
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Cumulative Regret')
    ax1.legend(loc=9,bbox_to_anchor = (1.82,-0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    #actions vs estimates
    sorted_ind = sorted(range(b.n), key=lambda x:b.probs[x])
    ax2.plot(range(b.n), [b.probs[x] for x in sorted_ind], 'k--', markersize=12)
    for s in solvers:
        ax2.plot(range(b.n), [s.estimated_probs[x] for  x in sorted_ind], 'x',markeredgewidth=2
)
    ax2.set_xlabel('Actions by ' + r'$\theta$')
    ax2.set_ylabel('Estimated')
    ax2.grid('k',ls='--', alpha=0.3)

    #Action vs frac the action is picked
    for s in solvers:
        ax3.plot(range(b.n), np.array(s.counts)/float(len(solvers[0].regrets)), drawstyle='steps', lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Fraction the Action is chosen')
    ax3.grid('k',ls='--', alpha=0.3)

    plt.savefig(fig_name)

def experiment(K,N):
    """
    Args:
    int K: # of slot machines
    int N: # of time steps
    """

    b = BernoulliBandit(K)
    print("Randomly gen Bernoulli bandit with reward probabilities:\n", b.probs)
    print(f"best machine has index: {max(range(K),key=lambda i:b.probs[i])} and probability: {max(b.probs)}")

    test_solvers = [
        EpsilonGreedy(b,0.01),
        UCB1(b),
        BayesianUCB(b,3,1,1),
        ThompsonSampling(b,1,1)
    ]

    names = [
        r'$\epsilon$' + '-Greedy',
        'UCB1',
        'Bayesian UCB',
        'Thompson Sampling'
    ]

    for s in test_solvers:
        s.run(N)

    plot_results(test_solvers, names, f"results_K{K}_N{N}")

if __name__ == '__main__':
    experiment(10,10000)