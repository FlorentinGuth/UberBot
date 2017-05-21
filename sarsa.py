from qlearning import *


class Sarsa(QLearning):
    def __init__(self, strategy, graph, gamma=0.9, nb_trials=None, alpha=0.01, potential=None, initial_nodes=None):
        """
        Initializes the Q-learning botnet.
        :param strategy:  defining how to resolve exploration vs. exploitation conflict
        :param graph:     the graph of the network
        :param gamma:    
        :param alpha:     learning rate
        :param shape:     whether to use reward shaping
        :param potential: apply the given potential for shaping
        """
        QLearning.__init__(self, strategy, graph, gamma, nb_trials, alpha, potential, initial_nodes)

        self.type = "Sarsa"

    def receive_reward(self, action, success, reward):
        """
        Updates the internal Q.
        :param action:  the action attempted by the botnet
        :param success: whether the action was successful (for updating self.state)
        :param reward:  immediate reward for doing action in self.state 
        :return:        None
        """
        # TODO: Account for shaping if needed
        # if self.shape:
        #     if self.potential is None:
        #         if success:
        #             reward =  self.gamma / (1. - self.gamma) * self.network.get_proselytism(action) - self.network.get_cost(action)
        #         else:
        #             reward -self.network.get_cost(action)
        #     else:
        #         if success:
        #             next_state = self.state.add(action)
        #         else:
        #             next_state = self.state
        #
        #         reward = self.network.immediate_reward(self.state, action) + self.gamma * self.potential(next_state) - self.potential(self.state)

        # # TODO: Back-propagation
        # if success:
        #     self.path.append(action)

        try:
            if success:
                self.state = self.state.add(action)
                next_action = self.choose_action()
                next_q_value = self.get_q_value(self.state, next_action)
                self.state = self.state.remove(action)
            else:
                next_action = self.choose_action()
                next_q_value = self.get_q_value(self.state, next_action)
        except IndexError:  # No next action
            next_q_value = self.get_best_actions(self.state)[0]

        new_q_value = reward + self.gamma * next_q_value
        old_q_value = self.get_q_value(self.state, action)
        q_value = (1 - self.alpha) * old_q_value + self.alpha * new_q_value

        self.set_q_value(self.state, action, q_value)

        LearningBotnet.receive_reward(self, action, success, reward)  # Updates state, reward...