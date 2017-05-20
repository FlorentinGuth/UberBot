from qlearning import QLearning
from math import sqrt
import random


class OrientedExploration(QLearning):
    """
    This class just defines a new of exploring, by choosing actions that are both good and unexplored.
    """

    def __init__(self, strategy, graph, gamma=0.9, nb_trials=None, alpha=0.01, potential=None, initial_nodes=None):
        """
        Initializes the Q-learning botnet.
        :param strategy:  defining how to resolve exploration vs. exploitation conflict
        :param graph:     the graph of the network
        :param gamma:     time discount factor
        :param alpha:     learning rate
        :param potential: apply the given potential for shaping, None if no such shaping
        """
        QLearning.__init__(self, strategy, graph, gamma, nb_trials, alpha, potential, initial_nodes)

        self.history = dict()  # Maps (state, action) to (success, total trials)

        self.type = "Oriented Exploration"


    def get_history(self, state, action):
        """
        Returns the number of successes and total trials
        :param state: 
        :param action: 
        :return: 
        """
        try:
            return self.history[state, action]
        except KeyError:
            return 0, 0

    def get_stats(self, state, action):
        """
        Returns the expected probability and variance of success given the history, assuming the probability follows a uniform law
        :param state: 
        :param action: 
        :return: 
        """
        k, n = self.get_history(state, action)
        exp = (k + 1) / float(n + 2)
        var = exp * (1 - exp) / float(n + 3)
        return exp, var

    def exploration(self):
        """
        This function selects the available action whose Q-value has the highest standard deviation, assuming the
        probability follows a uniform law.
        :return: 
        """
        best_std = 0
        best_actions = []
        for action in self.available_actions():
            std = self.get_stats(self.state, action)[1] * \
                  (self.get_best_actions(self.state)[0]**2 + self.get_best_actions(self.state.add(action))[0]**2)
            if std > best_std:
                best_actions = [action]
                best_std = std
            elif std == best_std:
                best_actions.append(action)
        return random.choice(best_actions)

    def receive_reward(self, action, success, reward):
        successes, total = self.get_history(self.state, action)
        self.history[self.state, action] = successes + success, total + 1
        QLearning.receive_reward(self, action, success, reward)

    def clear(self, all=False):
        QLearning.clear(self, all)
        if all:
            self.attempts = dict()