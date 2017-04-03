import abc
from state import *
from learning_botnet import *
from strategy import *


class Botnet(LearningBotnet):
    """
    This class provides helper methods for non-learning botnets, who have perfect knowledge of the network.
    """

    def __init__(self, network, gamma):
        """
        Initializes the botnet. The parameters nb_trials and strategy are here irrelevant.
        :param network: 
        :param gamma:  
        """
        # Chooses a full random strategy because exploration doesn't matter here (as long as it is O(1))
        LearningBotnet.__init__(self, full_random, network.graph, gamma)

        self.network = network
        self.power = network.initial_power

        self.type = "Botnet"

    def receive_reward(self, action, time, reward=None):
        """
        Same as LearningBotnet.receive_reward, but updates also the botnet's power.
        :param action: 
        :param time:
        :param reward:  can be None, will be calculated
        :return: None
        """
        # Computes the reward if needed
        if reward is None:
            reward = self.network.immediate_reward(self.state, action) * (1 - )

        # Updates state, time and reward
        LearningBotnet.receive_reward(self, action, success, reward)

        # Updates power
        if success:
            self.power += self.network.get_proselytism(action)

    def clear(self):
        """
        Resets the internal state, power, current time, reward...
        :return: 
        """
        LearningBotnet.clear(self)

        self.power = self.network.initial_power
