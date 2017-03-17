from botnet import *
from policy import *

import sys

sys.setrecursionlimit(10000)

class Fast(Botnet):
    """
    Botnet maybe minimizing the average time needed to hijack the whole network (O(n^2)).
    """

    def __init__(self, network):
        Botnet.__init__(self, network)

        self.time = {}      # time[(power, action)] is an estimation of the time to finish the job doing action
        self.min_time = {}  # min_time[power] is the min over all actions of time[(power, action)]
        self.total_power = network.total_power()
        self.type = "Fast_tentative"

    def compute_time(self, power, action):
        """ Returns time[(power, action)] and computes it if needed """
        if (power, action) in self.time:
            return self.time[power, action]

        if power >= self.total_power:
            res = 0 # We consider we have hijacked the whole network here
        else:
            res = 1. / self.network.success_probability_power(action, power) + \
                  self.compute_min_time(power + self.network.get_proselytism(action))

        self.time[power, action] = res
        return res

    def compute_min_time(self, power):
        """ Returns min_time[power] and computes it if needed """
        if power in self.min_time:
            return self.min_time[power]

        res = min([self.compute_time(power, action) \
                if self.network.get_proselytism(action) != 0 else 0 \
                for action in range(self.network.size)])

        self.min_time[power] = res
        return res

    def best_action(self, state, power): # power for avoiding recomputation
        best = None
        best_time = float("inf")
        for action in range(self.network.size):
            if action not in state and self.compute_time(power, action) < best_time:
                best = action
                best_time = self.compute_time(power, action)
        return best

    def compute_policy(self):
        n = self.network.size
        state = State(n)
        power = self.network.initial_power
        actions = []

        for _ in range(n):
            a = self.best_action(state, power)
            actions.append(a)
            state.add(a)
            power += self.network.get_proselytism(a)

        return Policy(self.network, actions)

    def choose_action(self, state):
        return self.best_action(state)
