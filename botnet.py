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
        LearningBotnet.__init__(self, full_exploration, network.graph, gamma, initial_nodes=network.initial_nodes)

        self.network = network
        self.power = network.initial_power

        self.type = "Botnet"

    def receive_reward(self, action, success, reward):
        """
        Same as LearningBotnet.receive_reward, but updates also the botnet's power.
        :param action: 
        :param success: 
        :param reward:
        :return:        None
        """
        # Computes the reward if needed
        reward = self.network.immediate_reward(self.state, action)

        # Updates state, time and reward
        LearningBotnet.receive_reward(self, action, success, reward)

        # Updates power
        if success:
            self.power += self.network.get_proselytism(action)

    def clear(self, all=False):
        """
        Resets the internal state, power, current time, reward...
        :param all: whether to also clear computed values, unused here
        :return: 
        """
        LearningBotnet.clear(self)

        self.power = self.network.initial_power
