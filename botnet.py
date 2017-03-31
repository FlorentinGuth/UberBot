import abc
from state import *


class Botnet:
    """
    The Botnet class contains the botnet state and information about its current reward.
    It is an abstract base class (abc): it cannot be instantiated, and an actual botnet will inherit this class.
    """

    def __init__(self, network, gamma=0.9):
        self.network = network
        self.state = State(network.size)
        self.power = network.initial_power
        self.type = None
        self.reward = 0
        self.time = 0
        self.time_factor = 1  # holds gamma ** T
        self.gamma = gamma

    def immediate_reward(self, state, action, success=None):
        """
        Wrapper for network.immediate_reward()
        :param state: 
        :param action: 
        :param success: unused here, but useful when inherited
        :return: 
        """
        return self.network.immediate_reward(state, action)

    def take_action(self, action):
        """
        Takes the given action, updates internal state and such
        :param action: 
        :return: True if the hijack was successful, False otherwise
        """
        success = self.network.attempt_hijacking(action, self.state)

        # Gets the immediate reward
        self.reward += self.time_factor * Botnet.immediate_reward(self, self.state, action)
        self.time_factor *= self.gamma
        self.time += 1

        if success:
            # Modify the state of the botnet in case of success
            self.state.add(action)
            self.power += self.network.get_proselytism(action)

            if self.state.is_full():
                self.reward += self.time_factor * self.network.total_power() / (1 - self.gamma)

        return success

    def reset(self):
        """
        Resets the internal state, power, current time, reward...
        :return: 
        """
        self.state = State(self.network.size)
        self.power = self.network.initial_power
        
        self.reward = 0
        self.time = 0
        self.time_factor = 1

    @abc.abstractmethod
    def compute_policy(self):
        """ Returns a policy, must be implemented when inherited """


def memoize(func):
    """
    Decorator whose type is (('a -> 'b) -> 'a -> 'b) -> ('a -> 'b), working with recursive functions.
    Usage:
    @memoize
    def f(f_mem, x):
        return f_mem(x-1) + 1
    """
    table = {}

    def func_mem(*args, **kwargs):
        if (args, kwargs) in table:
            return table[args, kwargs]

        res = func(func_mem, *args, **kwargs)
        table[args, kwargs] = res
        return res

    return func_mem
