from botnet import *
from policy import *


class RewardTentative(Botnet):
    """
    Botnet trying to maximize the average reward (suboptimal, O(n * total_power) ~ n²).
    Uses the formula (6) (Suboptimal, greedy algorithm) from the Overleaf.
    """

    def __init__(self, network, gamma=0.9):
        Botnet.__init__(self, network, gamma)

        self.q_tilde = {}  # q_tilde[power, action] is an estimation of the reward
        self.v_tilde = {}  # v_tilde[power] is the max over all actions of q_tilde

        self.total_power = network.total_power

        self.type = "RewardTentative"

    def compute_q_tilde(self, power, action):
        """
        Returns q_tilde[power, action] and computes it if needed.
        """
        if (power, action) in self.q_tilde:
            return self.q_tilde[power, action]

        if power >= self.total_power:
            res = self.network.final_reward(self.gamma)  # We consider we have hijacked the whole network here
        else:
            proba = self.network.success_probability_power(power, action)
            rho = self.network.immediate_reward_power(power, action)
            res = (rho + self.gamma * proba * self.compute_v_tilde(power + self.network.get_proselytism(action))) \
                  / (1. - self.gamma * (1. - proba))

        self.q_tilde[power, action] = res
        return res

    def compute_v_tilde(self, power):
        """
        Returns v_tilde[power] and computes it if needed.
        """
        if power in self.v_tilde:
            return self.v_tilde[power]

        res = min([self.compute_q_tilde(power, action) \
                if self.network.get_proselytism(action) != 0 else 0 \
                for action in range(self.network.size)])

        self.v_tilde[power] = res
        return res

    def best_action(self, state, power):
        """
        Returns the "best" action.
        :param state: 
        :param power: to avoid re-computation
        :return: 
        """
        best = None
        best_reward = float("-inf")
        for action in self.available_actions(state):
            if self.compute_q_tilde(power, action) > best_reward:
                best = action
                best_reward = self.compute_q_tilde(power, action)
        return best

    def exploitation(self):
        """
        Just calls best_action.
        :return: 
        """
        return self.best_action(self.state, self.network.current_power(self.state))

    def clear(self, all=False):
        """
        Clears internal storage.
        :param all: whether to also clear computed times
        :return: 
        """
        Botnet.clear(self)

        if all:
            self.q_tilde = dict()
            self.v_tilde = dict()
            self.total_power = self.network.total_power
