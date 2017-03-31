import random
from policy import *
# TODO Implémenter l'exploration online à profondeur fixée, par calcul exact
# TODO: refactor, along with qlearning.py to provide clearer interface and computes lazily everything (see fast.py for instance)


class Qstar(Botnet):
    """
    This class performs the computation of the Q* function.
    """

    def __init__(self, network, gamma):

        Botnet.__init__(self, network)

        self.content = dict()                     # Maps (state, action) to its Q*-value
        self.best_actions = dict()                # Maps a state to (best value, best action)

        self.gamma = gamma
        self.type = "Qstar"

    def clear(self):
        """
        Clears internal storage of Q*
        :return: 
        """
        self.content = dict()
        self.best_actions = dict()

    def set_q_value(self, state, action, value):
        """
        Updates Q* along with best action
        :param state: 
        :param action: 
        :param value: 
        :return: 
        """
        self.content[(state.content, action)] = value

        if value > self.max_line(state)[0]:
            # Update of the best action for this state.
            self.best_actions[state] = value, action

    def get_q_value(self, state, action):
        """
        Wrapper for content (Q*) with default value
        :param state: 
        :param action: 
        :return: 
        """
        try:
            return self.content[(state.content, action)]
        except KeyError:
            return self.ex_value(state, action)

    def exists(self, state, action):
        """
        Checks whether we know the Q*-value of the pair (state, action)
        :param state: 
        :param action: 
        :return: 
        """
        return (state.content, action) in self.content

    def max_line(self, state):
        """
        Wrapper for best_action with default values, calls ex_value
        :param state: 
        :return: (best value, best action) (beware that best_action can be None!)
        """
        try:
            return self.best_actions[state]
        except KeyError:
            best_a = None
            best_q = -float("inf")
            for a in self.network.get_actions(state):
                q = self.ex_value(state, a)
                if q > best_q:
                    best_q = q
                    best_a = a
            self.best_actions[state] = best_q, best_a
            return best_q, best_a

    def ex_value(self, state, action):
        """
        Computes Q* (or an under-approximation for non-optimal values)
        :param state: 
        :param action: 
        :return: the Q* value
        """

        if self.exists(state, action):
            return self.get_q_value(state, action)

        # The result may be smaller than its real exact value if it isn't the maximum.
        res = self.network.immediate_reward(state, action)

        splusa = State.added(state, action)
        proba_s_to_splusa = self.network.success_probability(action, state)

        max_q = -float("inf")
        max_action = None
        for h in self.network.get_actions(splusa):
            assert h not in splusa

            new_q = self.ex_value(splusa, h)

            if new_q > max_q:
                max_q = new_q

        if max_q == -float("inf"):
            # No more target left.
            # Possibility 1
            # max_q = 0

            # Possibility 2 (infinite horizon)
            max_q = self.network.total_power() / (1. - self.gamma)

        res += self.gamma * proba_s_to_splusa * max_q
        res /= (1 - self.gamma * (1 - proba_s_to_splusa))

        self.set_q_value(state, action, res)
        return res

    def ex_policy(self, state):
        """
        Computes the best action to perform according to Q*
        :param state: 
        :return: 
        """
        return self.max_line(state)[1]

    def compute_policy(self):
        """
        Computes the optimal policy from self.best_actions
        :return: 
        """
        n = self.network.size
        state = State(n)
        actions = []

        for _ in range(n):
            a = self.ex_policy(state)
            actions.append(a)
            state.add(a)

        self.reset()
        return Policy(self.network, actions)

    def choose_action(self, tot_nb_invasions=1, cur_nb_invasions=1):
        """
        Useless, wrapper for ex_policy to conform to the Qlearning interface
        :param tot_nb_invasions: 
        :param cur_nb_invasions: 
        :return: 
        """
        return self.ex_policy(self.state)
