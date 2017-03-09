from botnet import *


class Policy:
    """
    A policy is a sequence of actions (the order of the hijackings).
    This class contains also methods for getting expected values of policies.
    """

    def __init__(self, network, actions):
        self.network = network
        self.actions = actions

    def expected_time(self):
        t = 0
        power = self.network.initial_power

        for node in self.actions:
            t += 1. / self.network.success_probability_power(node, power)
            power += self.network.get_proselytism(node)

        return t

    def expected_reward(self, state, gamma, i):
        """ Updated version with the term accounting for infinite horizon """
        res = 0

        if i == len(self.actions):
            return self.network.total_power() / (1 - gamma)

        a = self.actions[i]
        p = self.network.success_probability(a, state)
        res += (Botnet(self.network).immediate_reward(state, a) +   # Are we sure it shouldn't be the new state here?
                gamma * p * self.expected_reward(State.added(state, a), gamma, i + 1))
        res /= float(1 - gamma * (1 - p))

        return res
