from debuts import Qbis
from random import *
__author__ = 'Martin'


class Thomson(Qbis):

    def __init__(self, n, gamma, alpha=0., inf=1000):
        Qbis.__init__(self, n, gamma, alpha, inf)

        self.p = dict()

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
        new_q = reward + self.gamma * p * self.max_line(state.add(action)) # Again computation ?
        new_q /= 1 - self.gamma * (1 - p)
        old_q = self.get(state, action)

        self.set(state, action, (1 - self.alpha) * old_q + self.alpha * new_q)

    def simulate(self, state):
        return [i for i in self.actions if (not i in state) and random() < self.get_p(i, state)]

    def thomson_policy(self, state):
        #Fonction Dstar based on Thomson sampling.
        possible_actions = self.simulate(state)

        if len(possible_actions) == 0:
            #No positive simulated action. MODIFY ?
            return self.policy(state)

        best_q = -self.inf
        best_actions = []

        for action in possible_actions:
            #It follows the same strategy, only looking the positive simulated trials. TO MODIFY
            new_q = self.get(state, action)

            # Other possible way to do : just suppose the simulation was real.
            # new_q = self.max_line(state.add(action))

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            return None

        #print("Expected best value : ", best_q)
        return choice(best_actions)