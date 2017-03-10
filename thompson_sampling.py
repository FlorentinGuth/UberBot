import random
from qlearning import Qlearning
from state import State
from botnet import Botnet

# TODO Ajouter de l'auto-évaluation des stratégies adoptées, s'en servir pour les retenir, et détecter des blocages.

class Thomson(Qlearning):

    def __init__(self, network, gamma, alpha=0., inf=100000):
        Qlearning.__init__(self, network, gamma, alpha, inf)

        self.p = dict()  # Saves the internal estimates of the success probabilities

    def get_p(self, action, state):
        try:
            success, trials = self.p[(action, state)]
            return success / trials

        except KeyError:
            return 1.0

    def get_trials(self, action, state):
        try:
            _, trials = self.p[(action, state)]
            return trials - 1

        except KeyError:
            return 0

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
        # new_q = reward + self.gamma * p * self.max_line(State.added(state, action))
        # new_q /= 1 - self.gamma * (1 - p)
        new_q = reward + self.gamma * self.max_line(State.added(state, action))
        old_q = self.get(state, action)

        self.set(state, action, (1 - self.alpha) * old_q + self.alpha * new_q)

        # print("Old : ", old_q, " new : ", new_q)

    def take_action(self, action):
        si = self.state.copy()
        res = Botnet.take_action(self, action)

        self.add_trial(action, si, res, self.immediate_reward(si, action))

        return res

    def simulate(self, state):
        return [i for i in self.network.get_actions(state) if random.random() < self.get_p(i, state)]

    def thomson_policy(self, state):
        # Computes best action to perform in this state according Thomson Sampling.
        possible_actions = self.simulate(state)

        # if len(possible_actions) == 0:
        if True:
            # No positive simulated action.
            return self.policy(state)

        best_q = -self.inf
        best_actions = []

        for action in possible_actions:
            # It follows the same strategy, only looking the positive simulated trials (more likely to exploit).
            # new_q = self.get(state, action)

            # Other possible way to do : just suppose the simulation was real (more likely to explore).
            # new_q = self.max_line(state.add(action))

            # Third possibility
            beta = 1.
            new_q = beta * self.get(state, action) + (1 - beta) * self.max_line(state.add(action))

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            return None

        # print("Expected best value : ", best_q)
        return random.choice(best_actions)

    def be_curious(self, state):

        min_cur = self.inf
        best_a = []

        for a in self.network.get_actions(state):
            if a not in state:
                x = self.get_p(a, state)
                if x < min_cur:
                    min_cur = x
                    best_a = [a]
                elif x == min_cur:
                    best_a.append(a)

        return int(random.choice(best_a))

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
        #   Thomson sampling
        #   ...

        # Typical strategy
        if cur_nb == tot_nb_invasions - 1:
            return self.policy(self.state)

        if random.random() > float(cur_nb) / tot_nb_invasions:
            return self.random_action()

        return self.thomson_policy(self.state)
