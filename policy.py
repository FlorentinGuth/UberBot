from botnet import *
import random


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
        # Initialization to last reward (accounting for infinite horizon)
        power = self.network.initial_power + sum(self.network.get_proselytism(action) for action in self.actions)
        reward = power / (1. - gamma)

        # Backward computation, from end to start
        for action in reversed(self.actions):
            power -= self.network.get_proselytism(action)
            p = self.network.success_probability_power(action, power)
            reward = float(self.network.immediate_reward_power(power, action) + gamma * p * reward) / (1 - gamma * (1 - p))

        return reward


def make_policy(best_action, network):
    """
    Automatically computes a policy from a best_action function
    :param best_action: a function mapping a state to a list of best actions (think about dict.get) 
                        (or a list/set of actions, in which case one will be randomly chosen)
    :param network      the network
    :return:            the policy
    """
    state = State(network.size)
    actions = []

    for _ in range(network.size):
        a = best_action(state)

        if type(a) != type(0):
            a = random.choice(a)

        actions.append(a)
        state.add(a)

    return Policy(network, actions)