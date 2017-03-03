import random
from state import *
from botnet import *
from policy import *


class Qstar(Botnet):
    """
    This class performs the computation of the Q* function.
    """

    def __init__(self, network, gamma, alpha=0., inf=1000):
        Botnet.__init__(self, network)

        self.content = dict()
        self.actions = list(range(network.size))
        self.gamma = gamma
        self.alpha = alpha
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

    def update_q_learning(self, si, a, sf):
        # TODO Nothing to do here if this botnet computes Qstar
        reward = self.immediate_reward(si, a)
        self.set(si, a, (1 - self.alpha) * self.get(si, a) + self.alpha * (reward + self.gamma * self.max_line(sf)))

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
        res = self.network.R(state, action)

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

    def policy(self, state):
        # Function Dstar (en supposant que Qstar a été calculée)
        # TODO Same thing : no reason to be here (but in Q learning ?)
        best_q = -self.inf
        best_actions = []

        for action in self.actions:
            if action in state:
                continue

            new_q = self.get(state, action)

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            return None

        # print("Expected best value : ", best_q)
        return random.choice(best_actions)

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

    def static_info(self, ex=False):
        # TODO Just computes the same thing as in Policy methods ?
        s = State(self.network.size)
        t = 0
        for _ in range(self.network.size):
            if ex:
                a = self.ex_policy(s)
            else:
                a = self.policy(s)

            t += 1.0 / self.network.p(a, s)
            s.add(a)
        return t, self.max_line(State(self.network.size))
