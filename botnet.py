import abc
from state import *


class Botnet:
    """
    The Botnet class contains the botnet state and information about its current reward.
    It is an abstract base class (abc): it cannot be instantiated, and an actual botnet will inherit this class.
    """

    def __init__(self, network):
        self.network = network
        self.state = State(network.size)
        self.power = network.initial_power

        self.reward = 0
        self.time = 0
        self.time_factor = 1  # holds gamma ** T
        self.gamma = 0.9      # To change?

    def immediate_reward(self, state, action):
        if action in state:
            return -self.network.cost(action)
        return -self.network.cost(action) + self.network.current_power(state)

    def take_action(self, action):
        success = self.network.attempt_hijacking(action, self.power)

        if success:
            self.state.add(action)
            self.power += self.network.get_proselytism(action)

        # Reward takes into account the latest action
        self.reward += self.time_factor * self.immediate_reward(self.state, action)
        self.time_factor *= self.gamma
        self.time += 1

    def reset(self):
        self.state = State(self.network.size)
        self.power = self.network.initial_power
        
        self.reward = 0
        self.time = 0
        self.time_factor = 1

    @abc.abstractmethod
    def compute_policy(self):
        """ Returns a policy, must be implemented when inherited """
