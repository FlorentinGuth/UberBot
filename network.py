import random
from state import *


class Network:
    """
    The Network class contains everything about the base data: the nodes, their attributes...
    It also contains the default reward functions (which can be modified by the botnets, by reward shaping for instance)
    """

    def __init__(self, base_power=0):

        self.initial_nodes = []             # List of nodes initially possessed by the botnet.
        self.base_power = base_power        # Some bonus power the botnet always has.
        self.initial_power = base_power     # Total power in initial state.

        self.total_power = base_power       # Power of the whole network.
        self.size = 0                       # Number of nodes in the network.

        # Nodes properties
        self.resistance = []
        self.proselytism = []
        self.action_cost = []
        self.graph = []                     # List of set of neighbors.

    def add_initial_node(self, node):
        """
        Adds the given node to initial state, and increases initial_power with its proselytism.
        """
        self.initial_power += self.get_proselytism(node)
        self.initial_nodes.append(node)

    def get_initial_state(self):
        return State(self.size, self.initial_nodes)

    def add_node(self, resistance, proselytism, cost):
        """
        Adds the given node to the network. This does not add any link between nodes.
        :param resistance: 
        :param proselytism: 
        :param cost: 
        :return: 
        """
        self.action_cost.append(cost)
        self.proselytism.append(proselytism)
        self.resistance.append(resistance)
        self.graph.append(set())

        self.size += 1
        self.total_power += proselytism

    def add_link(self, node1, node2):
        """
        Adds the edge 'node1 <--> node2' to the graph of the network.
        :param node1:
        :param node2:
        :return:
        """
        self.graph[node2].add(node1)
        self.graph[node1].add(node2)

    def set_complete_network(self):
        """
        Sets the network to be the complete graph.
        """
        tot = set(range(self.size))
        for i in range(self.size):
            self.graph[i] = tot  # Every node uses the same alias of tot, to not use too much space.

    def clear_graph(self):
        """
        Removes all edges from the graph
        :return: 
        """
        for i in range(self.size):
            self.graph[i] = set()

    def current_power(self, state):
        """
        Returns the sum of the proselytism of the hijacked nodes, plus the base power
        :param state: 
        :return:  
        """
        return self.base_power + sum(self.get_proselytism(i) for i in state.to_list())

    def get_cost(self, action):
        """
        Returns the cost of the given action
        :param action: 
        :return: 
        """
        return self.action_cost[action]

    def get_proselytism(self, node):
        """
        Returns the proselytism of the given node
        :param node: 
        :return: 
        """
        return self.proselytism[node]

    def get_resistance(self, node):
        """
        Returns the resistance of the given node
        :param node: 
        :return: 
        """
        return self.resistance[node]

    def success_probability_power(self, action, power):
        """
        Returns the probability of success
        :param action: the node to hijack
        :param power:  the available power
        :return:       the probability of success
        """
        # TODO: change to 1 - exp(-C*P/R) with C to adjust?
        if self.get_resistance(action) == 0:
            return 1.
        return min(1., float(power) / self.get_resistance(action))
        # return 1 -  math.exp(-float(power) / self.get_resistance(action))

    def success_probability(self, state, action):
        """
        Same as success_probability_power, but with a state
        :param state:  the current state
        :param action: the node to hijack
        :return:       the probability of success
        """
        return self.success_probability_power(action, self.current_power(state))

    def attempt_hijacking(self, state, action):
        """
        Attempts to hijack the given node.
        :param state: 
        :param action: 
        :return: True if the attack succeeded, False otherwise
        """
        rnd = random.random()
        probability = self.success_probability(state, action)
        return rnd < probability

    def immediate_reward(self, state, action):
        """
        Same as immediate_reward_power, but with a state instead of a power.
        :param state:
        :param action: 
        :return:       the immediate reward (independent on the success) 
        """
        if action in state:
            # Returns some arbitrary negative reward, to avoid to do it again. (But this shouldn't happen.)
            print("Someone tried to compute the reward of a stupid action, this can be optimized!")
            return -self.get_cost(action)

        return self.immediate_reward_power(self.current_power(state), action)

    def immediate_reward_power(self, power, action):
        """
        Computes the immediate reward (for one turn only). Please note this does not depend on the result of the action.
        Please also note that it assumes that action doesn't belong to current state.
        :param power:
        :param action: 
        :return: 
        """
        # TODO Change Probabilities !
        # The current reward is the total power the botnet isn't using while performing current action.
        return max(-self.get_cost(action) + power, 0)

    def final_reward(self, gamma):
        """
        Returns the last term of the reward, when the network has been fully captured.
        :param gamma: 
        :return: 
        """
        return self.total_power / float(1 - gamma)

    def generate_random_connected(self):
        """
        Adds random links on the network. The result is guaranteed to be connected.
        :return: None
        """
        rep = [i for i in range(self.size)]

        # Union-Find with path compression (each operation takes O(log(n)))
        def find(n):
            if rep[n] != n:
                rep[n] = find(rep[n])
            return rep[n]

        def union(a, b):
            A = find(a)
            B = find(b)
            rep[B] = A
            self.add_link(a, b)
            return A != B

        n = self.size
        while n > 1:
            a = random.randint(0, self.size-1)
            b = random.randint(0, self.size-1)
            if a != b and union(a, b):
                n -= 1


def random_network(size, difficulty, big_nodes, complete=True):
    """
    Generates a random network.

    How the network is computed:
     - For each node, we test at random if it will be a big one
     - Big nodes have a proselytism uniformly between 0 and size ** difficulty, small ones between 0 and size
     - The resistance is between a half and the double of the proselytism ** difficulty
     - The cost is a random fraction of the resistance
     - The edges are computed from Network.generate_random_connected()

    :param size:       the number of nodes in the resulting network
    :param difficulty: the resistance of a node is approximately equivalent to its proselytism ** difficulty
    :param big_nodes:  the ratio (between 0 and 1) of hard-to-hijack nodes
    :param complete:   if True, the network is complete, and is randomly connected otherwise
    :return:           the random network
    """
    network = Network(1)
    # TODO Check these parameters are nice for learning
    for _ in range(size):
        big = random.random() < big_nodes

        if big:
            proselytism = random.randint(0, int(size ** difficulty))
        else:
            proselytism = random.randint(0, size)

        resistance = int((3*random.random()+1)/2 * (proselytism ** difficulty))
        cost = int(random.random() * proselytism)

        network.add_node(resistance, proselytism, cost)

    if complete:
        network.set_complete_network()
    else:
        network.generate_random_connected()
    return network
