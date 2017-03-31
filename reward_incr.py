from botnet import *
from policy import *

class RewardIncr(Botnet):
    """
    Botnet trying to maximize the average time, by inserting nodes one after the other (sub-optimal, O(n^3)).
    Different orders are possible: sort the nodes by decreasing power or increasing resistance.
    """

    def __init__(self, network, gamma=0.9):
        Botnet.__init__(self, network, gamma)
        self.type = "Reward_incr"
        self.madness = madness

    def one_try(self, nodes):
        actions = []
        for node in nodes:
            best_pos = None
            best_reward = float("-inf")
            for i in range(len(actions)+1):
                actions.insert(i, node) # tests node in position i
                reward = Policy(self.network, actions).value(self.gamma)
                if reward > best_reward:
                    best_reward = reward
                    best_pos = i
                del actions[i]
            actions.insert(best_pos, node)
        return actions



    def compute_policy(self):
        n = self.network.size
        nodes = list(range(n))
        nodes.sort(key=lambda node: self.network.get_resistance(node))
        value = -float("inf")
        while True:
            old_value = value
            old = nodes[:]
            nodes = self.one_try(old)
            value = Policy(self.network, nodes).value(self.gamma)
            if old_value > value:
                nodes = old[:]
                value = old_value
            if old_value == value:
                break
            print(value)
        return Policy(self.network, nodes)
