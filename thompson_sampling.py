import random
from qlearning import QLearning
from botnet import Botnet
from state import State
from learning_botnet import *

# TODO Ajouter de l'auto-évaluation des stratégies adoptées, s'en servir pour les retenir, et détecter des blocages.
# TODO: Initialization of successes/failures


class Thompson(QLearning):
    """
    This class learns the Q-values just as Q-learning, but also learns the transition probabilities.
    Those are used to sample an action with Thompson Sampling.
    """

    def __init__(self, strategy, graph, gamma=0.9, nb_trials=None, alpha=0.01, beta=1, potential=None, initial_nodes=None):
        """
        Initializes the Thompson sampling botnet.
        :param strategy:  defining how to resolve exploration vs. exploitation conflict
        :param graph:     the graph of the network
        :param gamma:     time discount factor
        :param alpha:     learning rate
        :param beta:      a number between 0 and 1:
                          if 0, only look at the successive simulated trials ;
                          if 1, act as if the simulation was real
        :param shape:     whether to use reward shaping
        :param potential: user-specified potential for reward shaping
        """
        QLearning.__init__(self, strategy, graph, gamma, nb_trials, alpha, potential, initial_nodes)

        self.p = dict()     # Saves the internal estimates of the success probabilities: (nb_successes, nb_trials)
        self.beta = beta

        self.type = "ThompsonSampling"

    def update_p(self, state, action, result):
        """
        Updates the internal estimate of the probability of success.
        :param state: 
        :param action: 
        :param result: True if the hijacking succeeded, False otherwise
        :return: 
        """
        try:
            success, trials = self.p[state, action]
        except KeyError:
            success, trials = 1, 1
            # success, trials = 0, 0 TODO Modify initialisation and observe consequences on learning

        trials += 1
        success += result

        self.p[state, action] = success, trials

    def get_p(self, state, action):
        """
        Returns the empirical probability of success.
        :param state: 
        :param action: 
        :return: 
        """
        try:
            success, trials = self.p[state, action]
            return success / trials

        except KeyError:
            return 1  # TODO

    def get_trials(self, state, action):
        """
        Returns the number of time this action was attempted.
        :param state: 
        :param action: 
        :return: 
        """
        try:
            _, trials = self.p[state, action]
            return trials - 1   # TODO Modify this if we change initialisation

        except KeyError:
            return 0

    def simulate(self):
        """
        Simulates an attack on each reachable node.
        :return: the list of the successful attacks
        """
        return [i for i in self.available_actions() if random.random() < self.get_p(self.state, i)]

    def be_curious(self):  # TODO: merge with exploration?
        """
        Chooses the least visited action.
        :return: 
        """
        min_cur = float("inf")
        best_actions = []

        for action in self.available_actions():
            if action not in self.state:
                x = self.get_trials(self.state, action)
                if x < min_cur:
                    min_cur = x
                    best_actions = [action]
                elif x == min_cur:
                    best_actions.append(action)

        return int(random.choice(best_actions))

    def exploitation(self):
        """
        Chooses an action according to the Thompson Sampling policy.
        :return: 
        """
        # Computes best action to perform in this state according Thompson Sampling.
        possible_actions = self.simulate()

        if len(possible_actions) == 0:
            # No positive simulated action.
            # return self.policy(state) TODO
            return self.exploration()  # Random action

        best_q = -float("inf")
        best_actions = []

        for action in possible_actions:
            # It can here look on different Q-values on positive simulated actions :
            #   - Q*(state, action)                 (The simulation was not real)
            #   - max Q*(state+action, action')     (The simulation was real)
            # We chose to use some weighted combination of these both possibilities.

            max_line = self.get_best_actions(self.state.add(action))[0]
            q_value = self.get_q_value(self.state, action)

            new_q = self.beta * q_value + (1 - self.beta) * max_line

            if new_q >= best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        # print("Expected best value : ", best_q)
        return random.choice(best_actions)

    def receive_reward(self, action, success, reward):
        """
        Receives the reward and update the probability.
        :param action: 
        :param success: 
        :param reward: 
        :return: 
        """
        self.update_p(self.state, action, success)
        QLearning.receive_reward(self, action, success, reward)

    def clear(self, all=False):
        """
        Clears internal storage.
        :param all: 
        :return: 
        """
        QLearning.clear(self, all)

        if all:
            self.p = dict()


class ModelBasedThompson(Thompson):
    """
    blah blah
    """
    # TODO: add doc

    def __init__(self, strategy, graph, gamma=0.9, nb_trials=None, alpha=0.01, beta=1, potential=None, initial_nodes=None):
        """
        Initializes the Thompson sampling botnet.
        :param strategy:        defining how to resolve exploration vs. exploitation conflict
        :param graph:           the graph of the network
        :param gamma:           time discount factor
        :param alpha:           learning rate
        :param beta:            a number between 0 and 1:
                                if 0, only look at the successive simulated trials ;
                                if 1, act as if the simulation was real
        :param shape:           whether to use reward shaping
        :param potential:       user-specified potential for reward shaping
        """
        Thompson.__init__(self, strategy, graph, gamma, nb_trials, alpha, beta, potential, initial_nodes)

        self.memory = []   # TODO: add doc
        self.history = []  # TODO: add doc

        self.type = "ModelBasedThompson"

    def receive_reward(self, action, success, reward):
        """
        Bottom-up model-base update rule.
        :param action: 
        :param success: 
        :param reward: 
        :return: 
        """
        # Saves action, result, and corresponding reward.
        self.memory.append((self.state, action, reward))

        # Updates state and probability
        Thompson.receive_reward(self, action, success, reward)

        if self.state.is_full():
            # Uses model-based update rule, but in a bottom-up way.
            # The botnet reached the end of the invasion.
            # It is now time to apply updates from the bottom of the tree.
            # It is also going to evaluate the realized policy.
            policy = []
            previous_state = self.state
            expected_value = 0  # TODO Was it really final_reward ?

            for (state, action, reward) in reversed(self.memory):
                p = self.get_p(state, action)

                # Basic update rule
                # new_value = (1 - p) * self.max_line(state) + p * self.max_line(State.added(state, action))
                # new_value = reward + self.gamma * new_value

                # Tricky update rule --> A lot better !!
                new_value = reward + self.gamma * p * self.get_best_actions(state.add(action))[0]
                new_value /= 1 - self.gamma * (1 - p)

                self.set_q_value(state, action, new_value)

                if previous_state != state:
                    # It is a new action !
                    # Update the expected reward
                    expected_value = (reward + self.gamma * p * expected_value) / (1 - self.gamma * (1 - p))

                    # Saves the action in the policy
                    policy.append(action)
                    previous_state = state

            # Clears the memory of the botnet
            self.memory = []

            # Saves evaluated policy
            policy.reverse()
            self.history.append((policy, expected_value))

    def clear(self, all=False):
        """
        Clears internal storage
        :param all: 
        :return: 
        """
        Thompson.clear(self, all)

        if all:
            self.memory = []
            self.history = []


class FullModelBasedThompson(ModelBasedThompson):
    """
    blah
    """
    # TODO: add doc

    def __init__(self, strategy, graph, gamma=0.9, nb_trials=None, alpha=0.01, beta=1, potential=None, initial_nodes=None, alpha_p=0.05):
        """
        Initializes the Thompson sampling botnet.
        :param strategy:  defining how to resolve exploration vs. exploitation conflict
        :param graph:     the graph of the network
        :param gamma:     
        :param alpha:     learning rate
        :param beta:      a number between 0 and 1:
                          if 0, only look at the successive simulated trials ;
                          if 1, act as if the simulation was real
        :param shape:     whether to use reward shaping
        :param potential: user-specified potential for reward shaping
        :param alpha_p:   probability learning rate
        """
        ModelBasedThompson.__init__(self, strategy, graph, gamma, nb_trials, alpha, beta, potential, initial_nodes)
        self.alpha_p = alpha_p      # Probability learning rate
        self.type = "FullModelBasedThompson"

    def update_p(self, state, action, result):
        """
        Overrides the update rule of Thompson.
        Here, we learn the probability rather than empirically compute it.
        :param state: 
        :param action: 
        :param result: 
        :return: 
        """
        try:
            p, trials = self.p[(action, state)]
        except KeyError:
            p, trials = 1, 0
            # success, trials = 0, 0 TODO

        trials += 1
        p += self.alpha_p * (result - p)

        self.p[(action, state)] = p, trials

    def get_p(self, state, action):
        """
        Reflect the fact that the probability isn't stored the same way.
        :param state: 
        :param action: 
        :return: 
        """
        try:
            p, _ = self.p[(action, state)]
            return p

        except KeyError:
            return 1  # TODO

    def get_trials(self, state, action):
        try:
            _, trials = self.p[(action, state)]
            return trials

        except KeyError:
            return 0
