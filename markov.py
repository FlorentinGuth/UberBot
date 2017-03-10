import random
from policy import *
# TODO Implémenter l'exploration online à profondeur fixée, par calcul exact


class Qstar(Botnet):
    """
    This class performs the computation of the Q* function.
    """

    def __init__(self, network, gamma, inf=1000):
        Botnet.__init__(self, network)

        self.content = dict()
        self.best_actions = dict()
        self.actions = list(range(network.size))
        self.gamma = gamma
        self.inf = inf

    def clear(self):
        self.content = dict()

    def set(self, state, action, value):
        self.content[(state.content, action)] = value

        if value > self.max_line(state):
            # Update of the best action for this state.
            self.best_actions[state] = value

    def get(self, state, action):
        try:
            return self.content[(state.content, action)]
        except KeyError:
            return 0.

    def exists(self, state, action):
        return (state.content, action) in self.content

    def max_line(self, state):
        try:
            return self.best_actions[state]
        except KeyError:
            return 0.

    def ex_value(self, state, action):
        # Compute the exact value of Qstar.
        if self.exists(state, action):
            return self.get(state, action)

        # The result may be smaller than its real exact value if it isn't the maximum.
        res = self.network.immediate_reward(state, action)

        splusa = State.added(state, action)
        proba_s_to_splusa = self.network.success_probability(action, state)

        max_q = -self.inf
        for h in self.actions:
            if h in splusa:
                continue

            new_q = self.ex_value(splusa, h)

            if new_q > max_q:
                max_q = new_q

        if max_q == -self.inf:
            # No more target left.
            # Possibility 1
            # max_q = 0

            # Possibility 2
            max_q = self.network.total_power() / (1. - self.gamma)

        res += self.gamma * proba_s_to_splusa * max_q
        res /= (1 - self.gamma * (1 - proba_s_to_splusa))

        self.set(state, action, res)
        return res

    def ex_policy(self, state):
        # Computes the best action to perform according to Qstar.

        best_q = -self.inf
        best_actions = []

        for action in self.actions:
            if action in state:
                continue

            new_q = self.ex_value(state, action)

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            return None

        # print("Expected best value : ", best_q)
        return random.choice(best_actions)

    def compute_policy(self):
        n = self.network.size
        state = State(n)
        actions = []

        for _ in range(n):
            a = self.ex_policy(state)
            actions.append(a)
            state.add(a)

        return Policy(self.network, actions)

    def choose_action(self, tot_nb_invasions=1, cur_nb_invasions=1):
        return self.ex_policy(self.state)
