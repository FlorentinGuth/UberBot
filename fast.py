from botnet import *
from policy import *


class Fast(Botnet):
    """
    Botnet minimizing the average time needed to hijack the whole network (exponential complexity).
    """

    def __init__(self, network):
        Botnet.__init__(self, network)

        self.time = {}      # time[(state, action)] is the time to finish the job from state doing action
        self.min_time = {}  # min_time[state] is the min over all actions of time[(state, action)]
        self.type = "Fast_optimal"

    def compute_time(self, state, action):
        """ Returns time[(state, action)] and computes it if needed """
        if (state, action) in self.time:
            return self.time[state, action]

        if action in state:
            res = float("inf")  # We want to discourage such behavior (never useful)
        else:
            new_state = State.added(state, action)
            res = 1. / self.network.success_probability(action, state) + self.compute_min_time(new_state)

        self.time[state, action] = res
        return res

    def compute_min_time(self, state):
        """ Returns min_time[state] and computes it if needed """
        if state in self.min_time:
            return self.min_time[state]

        if state.is_full():
            res = 0
        else:
            res = min([self.compute_time(state, action) for action in range(self.network.size)])

        self.min_time[state] = res
        return res

    def best_action(self, state):
        best = None
        best_time = float("inf")
        for action in range(self.network.size):
            if self.compute_time(state, action) < best_time:
                best = action
                best_time = self.compute_time(state, action)
        return best

    def compute_policy(self):
        n = self.network.size
        state = State(n)
        actions = []

        for _ in range(n):
            a = self.best_action(state)
            actions.append(a)
            state.add(a)

        return Policy(self.network, actions)
