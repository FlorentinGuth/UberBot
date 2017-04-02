from state import State
from botnet import Botnet
import random
from policy import Policy
# TODO Comprendre l'initialisation des valeurs de Q learning
# TODO Detecter les blocages lors de l'apprentissage
# TODO Tester les blocages, essayer d'en déterminer l'origine
# TODO Sparse sampling algorithm ? / variante d'exploration à profondeur fixée ?


class Qlearning(Botnet):
    """
    This class computes an approximation of the exact Q*, by learning it incrementally.
    """

    def __init__(self, network, gamma, alpha=0.01, strat=None, shape=False):
        Botnet.__init__(self, network)

        self.content = dict()                     # Maps (state, action) to its Q value
        self.best_actions = dict()                # Maps a state to a couple (best_q, list of optimal actions)
        self.gamma = gamma
        self.alpha = alpha

        self.strat = strat
        self.type = "Qlearning"
        self.shape = shape

    def clear(self):
        """
        Clears the internal storage
        :return: 
        """
        self.reset()
        self.content = dict()
        self.best_actions = dict()

    def set(self, state, action, value):
        """
        Remembers that doing this action in this state has the given Q value (and updates best_actions)
        :param state: 
        :param action: 
        :param value: 
        :return: 
        """
        self.content[(state, action)] = value

        # Update best_actions
        (best_q, actions) = self.max_line(state)
        if value > best_q:
            self.best_actions[state] = (value, [action])
        elif value == best_q:
            actions.append(action)

    def get(self, state, action):
        """
        Returns the Q value
        :param state: 
        :param action: 
        :return: 
        """
        try:
            return self.content[(state, action)]
        except KeyError:
            # Initialization
            # return self.immediate_reward(state, action) / (1 - self.gamma)
            return -float("inf")

    def exists(self, state, action):
        """
        Returns True if we the Q value has already been computed
        :param state: 
        :param action: 
        :return: 
        """
        return (state.content, action) in self.content

    def max_line(self, state):
        """
        Returns the best Q value of the available actions in the given state, along with the corresponding actions
        :param state: 
        :return:
        """
        try:
            return self.best_actions[state]
        except KeyError:
            return -float("inf"), list(self.network.get_actions(state))

    def update_q_learning(self, si, a, sf, success):
        """
        Updates Q with the current experiment
        :param si:      the initial state
        :param a:       the action
        :param sf:      the final state
        :param success: whether the action was successful
        :return:        None
        """
        reward = self.immediate_reward(si, a, success)
        self.set(si, a, (1 - self.alpha) * self.get(si, a) + self.alpha * (reward + self.gamma * self.max_line(sf)[0]))

    def take_action(self, action):
        """
        Takes the given action in the current state, and updates Q-learning
        :param action: 
        :return: 
        """
        si = self.state.copy()
        res = Botnet.take_action(self, action)

        self.update_q_learning(si, action, self.state, success=res)

        return res

    def random_action(self):
        """
        Returns a random available action
        :return: 
        """
        return random.choice(list(self.network.get_actions(self.state)))

    def policy(self, state=None):
        """
        Computes the best action to take in this state according to the already computed Q.
        :param state: 
        :return: a random choice between the best actions available
        """

        if state is None:
            state = self.state

        best_q = -float("inf")
        best_actions = []

        for action in self.network.get_actions(state):
            if action in state:
                assert False

            new_q = self.get(state, action)

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            assert False

        # print("Expected best value : ", best_q)
        return random.choice(best_actions)

    def compute_policy(self):
        """
        Uses the policy method to computes the optimal policy (Q should already be computed)
        :return: the optimal policy
        """
        n = self.network.size
        state = State(n)
        actions = []

        for _ in range(n):
            a = self.policy(state)
            actions.append(a)
            state.add(a)
        self.reset()
        return Policy(self.network, actions)

    def choose_action(self, tot_nb_invasions, cur_nb):
        """
        :param tot_nb_invasions: total number of invasions
        :param cur_nb: current number of invasions
        :return: chooses an action to perform in current state, according to a variable strategy
        """
        # TODO In progress
        # Exploration
        #   Curious
        #   Random
        #   Progress
        #
        # Exploitation
        #   Best_action
        #   ...

        if self.strat is None:
            return self.policy(self.state)

        return self.strat(self, tot_nb_invasions, cur_nb)

    def immediate_reward(self, state, action, success):
        """
        Computes the immediate reward, corrected with shaping if self.shape is True
        :param state: 
        :param action: 
        :param success: whether the action was successful, needed for shaping (needs final state)
        :return: 
        """
        if not self.shape:
            return self.network.immediate_reward(state, action)

        if success:
            return self.gamma / (1. - self.gamma) * self.network.get_proselytism(action) - self.network.get_cost(action)
        return -self.network.get_cost(action)
