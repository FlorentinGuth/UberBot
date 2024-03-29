import random
from learning_botnet import *


class QLearning(LearningBotnet):
    """
    This class computes an approximation of the exact Q*, by learning it incrementally.
    """
    # TODO Understand initialization of the Q values

    def __init__(self, strategy, graph, gamma=0.9, nb_trials=None, alpha=0.01, potential=None, initial_nodes=None):
        """
        Initializes the Q-learning botnet.
        :param strategy:  defining how to resolve exploration vs. exploitation conflict
        :param graph:     the graph of the network
        :param gamma:     time discount factor
        :param alpha:     learning rate
        :param potential: apply the given potential for shaping, None if no such shaping
        """
        LearningBotnet.__init__(self, strategy, graph, gamma, nb_trials, initial_nodes)

        self.q_value = dict()                     # Maps (state, action) to its Q-value
        self.initialization = 0                   # Initialization value for Q
        self.path = []                            # Sequence of actions the botnet achieved, for back-propagation

        self.alpha = alpha
        self.shape = (potential is not None)
        self.potential = potential

        if not self.shape:
            self.type = "QLearning"
        else:
            self.type = "QLearning - Potential"

    def set_q_value(self, state, action, value):
        """
        Remembers that doing this action in this state has the given Q-value.
        We cannot memoize the best actions, since the value of an action can decrease.
        :param state: 
        :param action: 
        :param value: 
        :return: 
        """
        self.q_value[state, action] = value
        # print("Setting new q to", value, self.q_value[state, action])

    def get_q_value(self, state, action):
        """
        Returns the Q-value.
        :param state: 
        :param action: 
        :return: 
        """
        try:
            # print("asked", self.q_value[state, action])
            return self.q_value[state, action]
        except KeyError:
            # Initialization
            return self.initialization

    def get_best_actions(self, state):
        """
        Returns the set of the best actions in the given state.
        :param state: 
        :return:      (best Q-value, best actions)
        """
        # We receive no reward after the capture of the whole network
        if state.is_full():
            return 0, set()

        best_q_value = -float("inf")
        best_actions = set()

        for action in self.available_actions(state):
            q_value = self.get_q_value(state, action)

            if q_value > best_q_value:
                best_q_value = q_value
                best_actions = {action}     # This is a set, just saying
            elif q_value == best_q_value:
                best_actions.add(action)

        # print("best", best_q_value)
        return best_q_value, best_actions

    def exploration(self):
        """
        Tries to learn the network, given the current state.
        :return: an action
        """
        return LearningBotnet.exploration(self)  # Random action

    def exploitation(self):
        """
        Chooses a random action between those that maximizes the learned Q.
        :return: an action
        """
        q_value, actions = self.get_best_actions(self.state)
        return random.choice(list(actions))

    def receive_reward(self, action, success, reward):
        """
        Updates the internal Q.
        :param action:  the action attempted by the botnet
        :param success: whether the action was successful (for updating self.state)
        :param reward:  immediate reward for doing action in self.state 
        :return:        None
        """
        if self.shape:
            if success:
                next_state = self.state.add(action)
            else:
                next_state = self.state
            reward += self.gamma * self.potential(next_state) - self.potential(self.state)

        # # TODO: Back-propagation
        # if success:
        #     self.path.append(action)

        new_q_value = reward + self.gamma * self.get_best_actions(self.state)[0]
        old_q_value = self.get_q_value(self.state, action)
        q_value = (1 - self.alpha) * old_q_value + self.alpha * new_q_value

        self.set_q_value(self.state, action, q_value)

        LearningBotnet.receive_reward(self, action, success, reward)  # Updates state, reward...

    def clear(self, all=False):
        """
        Clears internal storage.
        :param all: whether to also clear learned Q-values
        :return: 
        """
        LearningBotnet.clear(self)

        if all:
            self.q_value = dict()
        else:
            # TODO: Back-propagates the max_a Q(s, a) on each state to speed-up learning
            # state = State.full_state(self.size)
            # for action in reversed(self.path):
            #     state.remove(action)
            #     self.set_q_value(state, action, )
            pass

        self.path = []
