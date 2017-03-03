import random
from qlearning import Qlearning
from state import State


class Thomson(Qlearning):

    def __init__(self, network, gamma, alpha=0., inf=1000):
        Qlearning.__init__(self, network, gamma, alpha, inf)

        self.p = dict()  # Saves the internal estimates of the success probabilities

    def get_p(self, action, state):
        try:
            success, trials = self.p[(action, state)]
            return success / trials

        except KeyError:
            return 1.0

    def add_trial(self, action, state, result, reward):
        try:
            success, trials = self.p[(action, state)]
        except KeyError:
            success, trials = 1, 1

        trials += 1

        if result:
            success += 1

        self.p[(action, state)] = success, trials

        p = success / trials
        new_q = reward + self.gamma * p * self.max_line(State.added(state, action))  # Redundant computation ?
        new_q /= 1 - self.gamma * (1 - p)
        old_q = self.get(state, action)

        self.set(state, action, (1 - self.alpha) * old_q + self.alpha * new_q)

    def simulate(self, state):
        return [i for i in self.actions if (i not in state) and random.random() < self.get_p(i, state)]

    def thomson_policy(self, state):
        # Computes best action to perform in this state according Thomson Sampling.
        possible_actions = self.simulate(state)

        if len(possible_actions) == 0:
            # No positive simulated action. MODIFY ?
            return self.policy(state)

        best_q = -self.inf
        best_actions = []

        for action in possible_actions:
            # It follows the same strategy, only looking the positive simulated trials (more likely to exploit).
            new_q = self.get(state, action)

            # Other possible way to do : just suppose the simulation was real (more likely to explore).
            # new_q = self.max_line(state.add(action))

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            return None

        # print("Expected best value : ", best_q)
        return random.choice(best_actions)
