import random
from math import log

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

    def expected_reward(self, gamma):
        """
        Computes the expected total reward of the policy
        :param gamma
        :return
        """
        # TODO: Change when it will change in network (--> network?)
        # Initialization to last reward (accounting for infinite horizon)
        power = self.network.initial_power + sum(self.network.get_proselytism(action) for action in self.actions)
        reward = float(power)

        # Backward computation, from end to start
        for action in reversed(self.actions):
            power -= self.network.get_proselytism(action)
            p = self.network.success_probability_power(action, power)
            expected_time_factor = 1 / (1 - log(gamma) / p)
            reward *= expected_time_factor
            reward += self.network.immediate_reward_power(power, action) * (1 - expected_time_factor)

        return reward
