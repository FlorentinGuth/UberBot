import abc
import random
from state import *


class LearningBotnet:
    """
    This class provides the interface all botnets (learning or not) must match.
    Convention: the botnet knows nothing except:
                 - the graph of the network (could be learned after one trial, doing a depth first search)
                 - its state (which nodes are hijacked)
                 - the result of its actions (whether an attack was successful)
                 - the immediate reward at each step
    """

    def __init__(self, strategy, graph, gamma=0.9, nb_trials=None, initial_nodes=None):
        """
        Initializes the botnet.
        :param strategy:      function of signature LearningBotnet -> action 
                              (using q.exploration() and q.exploitation())
        :param graph:         the graph of the network (of type node set list)
        :param gamma:         to compute the reward
        :param nb_trials:     total number of trials (goal: this parameter is None)

        """
        self.graph = graph              # Graph of the network
        self.size = len(graph)          # Total number of nodes
        self.state = State(self.size, initial_nodes)
        self.initial_nodes = initial_nodes  # List of initial nodes
        print(initial_nodes)
        self.gamma = gamma
        self.reward = 0
        self.time = 0
        self.time_factor = 1            # Holds gamma ** T

        self.completed_trials = 0       # Number of already completed trials
        self.nb_trials = nb_trials      # Number of total trials (can be None)

        self.strategy = strategy

        self.type = "LearningBotnet"    # A string containing the name of the botnet

    def available_actions(self, state=None):
        """
        Computes the neighbours of the hijacked nodes.
        :param state: if not provided, defaults to self.state
        :return:      a set of actions 
        """
        if state is None:
            state = self.state

        if state.is_empty():
            # If the botnet starts in empty state, it is allowed to choose any action.
            return set(range(self.size))

        res = set()
        state = state.to_list()
        for i in state:
            res.update(self.graph[i])  # Adds the neighbours of node i
        for i in state:
            res.discard(i)             # Removes the nodes we already hijacked
        return res

    def exploration(self):
        """
        This method tells the botnet to try to learn as much information on the network as possible.
        By default, it chooses one of the available actions at random.
        :return: an action
        """
        return random.choice(list(self.available_actions()))

    @abc.abstractmethod
    def exploitation(self):
        """
        This method tells the botnet to try to maximize its reward.
        This function should only use the current state, not the current time or the current reward (recalls that an 
        optimal policy only depends on the state, not on the past successes and failures).
        :return: an action
        """
        pass

    def choose_action(self):
        """
        Either explore or exploit, depending on the strategy.
        :return: an action
        """
        return self.strategy(self)

    def receive_reward(self, action, success, reward):
        """
        Tells the botnet the result of its attack.
        :param action:  the node the botnet tried to hijack
        :param success: whether the hijacking succeeded
        :param reward:  the reward received for the current step
        :return:        None
        """
        if success:
            self.state = self.state.add(action)

        self.reward += self.time_factor * reward
        self.time += 1
        self.time_factor *= self.gamma

    def clear(self, all=False):
        """
        Prepares the botnet for another trial on the same network.
        :param all: whether to clear also learned values, unused here
        :return:    None
        """
        self.state = State(self.size, self.initial_nodes)

        self.reward = 0
        self.time = 0
        self.time_factor = 1

        self.completed_trials += 1

    def compute_policy(self):
        """
        Computes the full-exploitation policy, using what the botnet learned of the network.
        :return: a sequence of actions, which is the order of attacks on the nodes
        """
        temp = self.state              # We do not want to modify this state after the call
        self.state = State(self.size, self.initial_nodes)
        actions = []

        while not self.state.is_full():
            action = self.exploitation()
            actions.append(action)
            self.state = self.state.add(action)

        self.state = temp              # Restores the current state
        return actions
