import numpy as np
import time
from scipy.stats import beta
from abc import ABC, abstractmethod

from bandit import BernoulliBandit

class Solver(ABC):
    def __init__(self, bandit):

        assert isinstance(bandit, BernoulliBandit)

        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0]*self.bandit.n
        self.actions = []
        self.regret = 0.0
        self.regrets = [0.0]

    def update_regret(self, i):
        self.regret += self.bandit.best_prob - self.bandit.probs[i]
        self.regrets.append(self.regret)

    @property
    @abstractmethod
    def estimated_probs(self):
        pass

    @abstractmethod
    def run_step(self):
        pass

    def run(self, num_steps):
        assert self.bandit is not None
        for _ in range(num_steps):
            i = self.run_step()

            self.counts[i] +=1
            self.actions.append(i)
            self.update_regret(i)


class EpsilonGreedy(Solver):
    def __init__(self, bandit, eps, init_prob=1.0):
        super().__init__(bandit)

        self.eps = eps

        self.estimates = [init_prob]* self.bandit.n

    @property
    def estimated_probs(self):
        return self.estimates
    
    def run_step(self):
        if np.random.random()<self.eps:
            i = np.random.randint(0, self.bandit.n)
        else:
            i = max(range(self.bandit.n), key =lambda x: self.estimates[x])
        
        r = self.bandit.generate_reward(i)
        self.estimates[i] += 1.0 / (self.counts[i] + 1) * (r - self.estimates[i])

        return i

class UCB1(Solver):
    def __init__(self, bandit, init_prob =1.0):
        super().__init__(bandit)
        self.t = 0
        self.estimates = [init_prob]*self.bandit.n

    @property
    def estimated_probs(self):
        return self.estimates

    def run_step(self):
        self.t +=1
        
        i = max(range(self.bandit.n), key = lambda x: self.estimates[x] + np.sqrt(2* np.log(self.t)/(1+self.counts[x])))
        r = self.bandit.generate_reward(i)

        self.estimates[i] +=1.0 / (self.counts[i]+1)*(r-self.estimates[i])

        return i


class BayesianUCB(Solver):
    def __init__(self, bandit, c=3, init_a =1, init_b=1):

        super().__init__(bandit)
        self.c = c
        self._as = [init_a]*self.bandit.n
        self._bs = [init_b]*self.bandit.n

    @property
    def estimated_probs(self):
        return [self._as[i]/float(self._as[i] + self._bs[i]) for i in range(self.bandit.n)]

    def run_step(self):
        i = max(range(self.bandit.n), key=lambda x: self._as[x]/float(self._as[x] + self._bs[x]) + beta.std(self._as[x], self._bs[x])*self.c)
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1-r)

        return i

class ThompsonSampling(Solver):
    def __init__(self, bandit, init_a =1, init_b=1):
        super().__init__(bandit)

        self._as = [init_a]*self.bandit.n
        self._bs = [init_b]*self.bandit.n

    @property
    def estimated_probs(self):
        return [self._as[i]/(self._as[i]+self._bs[i]) for i in range(self.bandit.n)]
        
    def run_step(self):
        samples = [np.random.beta(self._as[x], self._bs[x]) for x in range(self.bandit.n)]
        i = max(range(self.bandit.n), key=lambda x: samples[x])
        r = self.bandit.generate_reward(i)

        self._as[i] += r
        self._bs[i] += (1-r)

        return i