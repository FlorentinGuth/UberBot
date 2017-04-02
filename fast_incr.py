from botnet import *
from policy import *


class FastIncr(Botnet):
    """
    Botnet trying to minimize the average time, by inserting nodes one after the other (sub-optimal, O(n^3)).
    Different orders are possible: sort the nodes by decreasing power or increasing resistance for instance.
    Please note that this approach does not work with non-complete networks!
    """

    def __init__(self, network, gamma=0.9):
        Botnet.__init__(self, network, gamma)

        self.type = "FastIncr"

    def exploitation(self):
        """
        The method could be implemented, but I let it this way to stress the fact that this botnet don't behave like
        others.
        :return: a random action
        """
        return self.exploration()

    def compute_policy(self):
        """
        Computes a policy by incrementally inserting nodes into it.
        :return: 
        """
        n = self.network.size
        actions = []

        nodes = list(range(n))
        nodes.sort(key=lambda node: self.network.get_resistance(node))
        for node in nodes:
            best_pos = None
            best_time = float("inf")
            for i in range(len(actions)+1):
                actions.insert(i, node)                               # Tests node in position i
                time = Policy(self.network, actions).expected_time()
                if time < best_time:
                    best_time = time
                    best_pos = i
                del actions[i]
            actions.insert(best_pos, node)

        return actions
