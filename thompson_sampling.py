import random
from qlearning import Qlearning
from botnet import Botnet
from state import State

# TODO Ajouter de l'auto-évaluation des stratégies adoptées, s'en servir pour les retenir, et détecter des blocages.


class Thompson(Qlearning):

    def __init__(self, network, gamma, alpha=0., strat=None, inf=100000):
        Qlearning.__init__(self, network, gamma, alpha, strat, inf)

        self.p = dict()  # Saves the internal estimates of the success probabilities
        self.type = "Thompson Sampling"

    def update_p(self, action, state, result):
        try:
            success, trials = self.p[(action, state)]
        except KeyError:
            success, trials = 1, 1
            # success, trials = 0, 0 TODO

        trials += 1

        if result:
            success += 1

        self.p[(action, state)] = success, trials

    def get_p(self, action, state):
        try:
            success, trials = self.p[(action, state)]
            return success / trials

        except KeyError:
            return 1  # TODO

    def get_trials(self, action, state):
        try:
            _, trials = self.p[(action, state)]
            return trials - 1

        except KeyError:
            return 0

    def add_trial(self, action, state, result):
        self.update_p(action, state, result)
        self.update_q_learning(state, action, self.state)

    def take_action(self, action):
        si = self.state.copy()
        res = Botnet.take_action(self, action)

        self.add_trial(action, si, res)

        return res

    def simulate(self, state):
        return [i for i in self.network.get_actions(state) if random.random() < self.get_p(i, state)]

    def thompson_policy(self, state):
        # Computes best action to perform in this state according Thompson Sampling.
        possible_actions = self.simulate(state)

        if len(possible_actions) == 0:
            # No positive simulated action.
            # return self.policy(state) TODO
            return self.random_action()

        best_q = -self.inf
        best_actions = []

        for action in possible_actions:
            # It follows the same strategy, only looking the positive simulated trials (more likely to exploit).
            # new_q = self.get(State.added(state, action))

            # Other possible way to do : just suppose the simulation was real (more likely to explore).
            # new_q = self.max_line(State.added(state, action))

            # Third possibility
            beta = 1.
            new_q = beta * self.get(state, action) + (1 - beta) * self.max_line(State.added(state, action))

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

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


class ModelBasedThompson(Thompson):

    def __init__(self, network, gamma, alpha=0., strat=None, inf=100000):
        Thompson.__init__(self, network, gamma, alpha, strat, inf)
        self.memory = []
        self.history = []

    def add_trial(self, action, state, result):
        self.update_p(action, state, result)

        # Uses model-based update rule, but in a bottom-up way.

        if not self.state.is_full():
            self.memory.append((state, action, self.immediate_reward(state, action)))
        else:
            # The botnet reached the end of the invasion.
            # It is now time to apply updates from the bottom of the tree.
            # It is also going to evaluate the realized policy.
            policy = []
            previous_state = self.state
            expected_value = self.network.total_power() / (1 - self.gamma)  # Some constant ? TODO

            for (state, action, reward) in reversed(self.memory):
                p = self.get_p(action, state)

                # Basic update rule
                # new_value = (1 - p) * self.max_line(state) + p * self.max_line(State.added(state, action))
                # new_value = reward + self.gamma * new_value

                # Tricky update rule --> A lot better !!
                new_value = reward + self.gamma * p * self.max_line(State.added(state, action))
                new_value /= 1 - self.gamma * (1 - p)

                self.set(state, action, new_value)

                if previous_state != state:
                    # It is a new action !
                    # Update the expected reward
                    expected_value = (reward + self.gamma * p * expected_value) / (1 - self.gamma * (1 - p))

                    # Saves the action in the policy
                    policy.append(action)
                    previous_state = state

            # Clears the memory of the botnet
            self.memory = []

            # Could also try to evaluate this policy ?
            policy.reverse()
            self.history.append((policy, expected_value))


class FullModelBasedThompson(ModelBasedThompson):

    def __init__(self, network, gamma, alpha=0., strat=None, inf=100000):
        ModelBasedThompson.__init__(self, network, gamma, alpha, strat, inf)

    def update_p(self, action, state, result):
        try:
            p, trials = self.p[(action, state)]
        except KeyError:
            p, trials = 1, 0
            # success, trials = 0, 0 TODO

        trials += 1
        p += self.alpha * (result - p)

        self.p[(action, state)] = p, trials

    def get_p(self, action, state):
        try:
            p, _ = self.p[(action, state)]
            return p

        except KeyError:
            return 1  # TODO

    def get_trials(self, action, state):
        try:
            _, trials = self.p[(action, state)]
            return trials

        except KeyError:
            return 0