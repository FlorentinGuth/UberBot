from botnet import *
from policy import *


class FastTentative(Botnet):
    """
    Botnet trying to minimize the average time needed to hijack the whole network (suboptimal, O(n^2)).
    """

    def __init__(self, network, gamma=0.9):
        Botnet.__init__(self, network, gamma)

        self.exp_time = {}  # exp_time[(power, action)] is an estimation of the time to finish the job doing action
        self.min_time = {}  # min_time[power] is the min over all actions of time[(power, action)]

        self.total_power = network.total_power

        self.type = "FastTentative"

    def compute_time(self, power, action):
        """
        Returns time[(power, action)] and computes it if needed.
        """
        if (power, action) in self.exp_time:
            return self.exp_time[power, action]

        if power >= self.total_power:
            res = 0  # We consider we have hijacked the whole network here
        else:
            res = 1. / self.network.success_probability_power(action, power) + \
                  self.compute_min_time(power + self.network.get_proselytism(action))

        self.exp_time[power, action] = res
        return res

    def compute_min_time(self, power):
        """
        Returns min_time[power] and computes it if needed.
        """
        if power in self.min_time:
            return self.min_time[power]

        res = min([self.compute_time(power, action) \
                if self.network.get_proselytism(action) != 0 else 0 \
                for action in range(self.network.size)])

        self.min_time[power] = res
        return res

    def best_action(self, state, power):
        """
        Returns the "best" action.
        :param state: 
        :param power: to avoid re-computation
        :return: 
        """
        best = None
        best_time = float("inf")
        for action in range(self.network.size):
            if action not in state and self.compute_time(power, action) < best_time:
                best = action
                best_time = self.compute_time(power, action)
        return best

    def exploitation(self):
        """
        Just calls best_action.
        :return: 
        """
        return self.best_action(self.state, self.network.current_power(self.state))

    def clear(self, all=False):
        """
        Clears internal storage.
        :param all: whether to also clear computed times
        :return: 
        """
        Botnet.clear(self)

        if all:
            self.exp_time = dict()
            self.min_time = dict()
