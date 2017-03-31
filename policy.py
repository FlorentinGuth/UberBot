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
        """
        Updated version with the term accounting for infinite horizon
        You should not call this method, use value instead
        :param state the state from which to calculate the reward (usually the empty one)
        :param gamma
        :param i     the cardinality of state
        :return
        """
        res = 0
        if i == len(self.actions):
            return self.network.total_power() / (1 - gamma)  # TODO: Not quite accurate for incr botnets

        a = self.actions[i]
        p = self.network.success_probability(a, state)
        res += (Botnet(self.network).immediate_reward(state, a) +
                gamma * p * self.expected_reward(State.added(state, a), gamma, i + 1))
        res /= float(1 - gamma * (1 - p))

        return res

    def value(self, gamma):
        """
        Bottom-up (non-recursive) version of expected_reward
        :param gamma
        :return
        """
        # Initialization to last reward (accounting for infinite horizon)
        power = self.network.initial_power + sum(self.network.get_proselytism(action) for action in self.actions)
        reward = power / (1. - gamma)

        # Backward computation, from end to start
        for action in reversed(self.actions):
            power -= self.network.get_proselytism(action)
            p = self.network.success_probability_power(action, power)
            reward = float(self.network.immediate_reward_power(power, action) + gamma * p * reward) / (1 - gamma * (1 - p))

        return reward
