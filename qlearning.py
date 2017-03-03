from state import State
from botnet import Botnet
import random
from policy import Policy
# TODO Group the methods using Q function in a intermediate class ?


class Qlearning(Botnet):
    """
    This class computes an approximation of the exact Qstar, by learning it incrementally.
    """

    def __init__(self, network, gamma, alpha=0., inf=1000):
        Botnet.__init__(self, network)

        self.content = dict()
        self.best_actions = dict()
        self.actions = list(range(network.size))
        self.gamma = gamma
        self.alpha = alpha
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
            return 0

    def exists(self, state, action):
        return (state.content, action) in self.content

    def max_line(self, state):
        try:
            return self.best_actions[state]
        except KeyError:
            return 0.

    def update_q_learning(self, si, a, sf):
        reward = self.immediate_reward(si, a)
        self.set(si, a, (1 - self.alpha) * self.get(si, a) + self.alpha * (reward + self.gamma * self.max_line(sf)))

    def random_action(self):
        return random.choice([a for a in self.actions if a not in self.state])

    def policy(self, state=None):
        # Computes the best action to take in this state according to the already computed Q.

        if state is None:
            state = self.state

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

    def compute_policy(self):
        n = self.network.size
        state = State(n)
        actions = []

        for _ in range(n):
            a = self.policy(state)
            actions.append(a)
            state.add(a)

        return Policy(self.network, actions)
