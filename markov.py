import random
from botnet import *
from policy import *


class Qstar(Botnet):
    """
    This class performs the computation of the Q* function.
    """

    def __init__(self, network, gamma, inf=1000):
        Botnet.__init__(self, network)

        self.content = dict()
        self.actions = list(range(network.size))
        self.gamma = gamma
        self.inf = inf

    def clear(self):
        self.content = dict()

    def set(self, state, action, value):
        self.content[(state.content, action)] = value

    def get(self, state, action):
        try:
            return self.content[(state.content, action)]
        except KeyError:
            return 0

    def exists(self, state, action):
        return (state.content, action) in self.content

    def max_line(self, state):
        # TODO Save the maxima ?
        return max(self.get(state, action) for action in self.actions)

    def update_fix_point(self):
        # TODO Delete this or try to understand what he meant with these fix point iterations
        # Pas du tout bien défini ce que ca devrait faire pour rester n**k (k = nombre itérations)
        states = []
        actions = []

        for s in states:
            for a in actions:
                # Il faut détailler la formule.
                self.set(s, a, self.network.R(s, a) + self.gamma * 0)

    def ex_value(self, state, action):
        # Compute the exact value of Qstar.
        if self.exists(state, action):
            return self.get(state, action)

        # The result may be smaller than its real exact value if it isn't the maximum.
        res = self.network.immediate_reward(state, action)

        splusa = State.added(state, action)
        proba_s_to_splusa = self.network.success_probability(action, state)

        maxQ = -self.inf
        for h in self.actions:
            if h in splusa:
                continue

            newQ = self.ex_value(splusa, h)

            if newQ > maxQ:
                maxQ = newQ

        if maxQ == -self.inf:
            # No more target left.
            # Possibility 1
            # maxQ = 0

            # Possibility 2
            maxQ = self.network.total_power() / (1. - self.gamma)

        res += self.gamma * proba_s_to_splusa * maxQ
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
