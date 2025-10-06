import time
import numpy as np
from abc import ABC, abstractmethod

class Bandit(ABC):

    @abstractmethod
    def generate_reward(self,i):
        pass


class BernoulliBandit(Bandit):

    def __init__(self, n, probs=None):
        assert probs is None or len(probs) == n

        self.n = n

        if probs is None:
            np.random.seed(int(time.time()))
            self.probs = [np.random.random() for _ in range(self.n)]
        else:
            self.probs = probs

        self.best_prob = max(self.probs)

    def generate_reward(self, i):
        if np.random.random() < self.probs[i]:
            return 1
        else:
            return 0