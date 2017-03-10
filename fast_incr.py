from botnet import *
from policy import *


class Fast(Botnet):
    """
    Botnet trying to minimize the average time, by inserting nodes one after the other (sub-optimal, O(n^3)).
    Different orders are possible: sort the nodes by decreasing power or increasing resistance.
    """

    def __init__(self, network):
        Botnet.__init__(self, network)

    def compute_policy(self):
        n = self.network.size
        actions = []

        nodes = list(range(n))
        nodes.sort(key=lambda node: self.network.get_resistance(node))
        for node in nodes:
            best_pos = None
            best_time = float("inf")
            for i in range(len(actions)+1):
                actions.insert(i, node) # tests node in position i
                time = Policy(self.network, actions).expected_time()
                if time < best_time:
                    best_time = time
                    best_pos = i
                del actions[i]
            actions.insert(best_pos, node)

        return Policy(self.network, actions)
