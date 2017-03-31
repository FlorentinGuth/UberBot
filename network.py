import random
from state import *


class Network:
    """
    The Network class contains everything about the base data: the nodes, their attributes...
    It does not contain a state, and neither the different reward-related functions (those are associated with the
    botnet rather than being a part of the problem)
    """

    def __init__(self, initial_power):
        self.initial_power = initial_power
        self.size = 0

        self.resistance = []
        self.proselytism = []
        self.action_cost = []
        self.graph = []         # List of set of neighbors

    def add(self, resistance, proselytism, cost):
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
        self.size += 1
        self.graph.append(set())

    def add_link(self, node1, node2):
        """
        Adds the edge 'node1 --> node2' to the graph of the network
        :param node1:
        :param node2:
        :return:
        """
        # TODO: un attribut accessibles (ensemble des noeuds accesibles depuis l'etat courant)
        # TODO: Possibilité de modifier le graphe en des ensembles ?
        # TODO: Ajouter une methode get_actions(self, state) ou qqch comme ca
        # TODO: It does not make any sense for the graph to be cyclic

        self.graph[node1].add(node2)


    def set_complete_network(self):
        """
        Sets the network to be the complete graph
        :return: 
        """
        tot = set(range(self.size))
        for i in range(self.size):
            self.graph[i] = tot

    def clear_graph(self):
        """
        Removes all edges from the graph
        :return: 
        """
        for i in range(self.size):
            self.graph[i] = set()

    def get_actions(self, state):
        """
        Returns the available actions in the given state
        :param state: the current state
        :return:      a set of actions
        """
        if state.is_empty():
            # TODO: makes not much sense to be able to choose any node
            # TODO: we could say that the Botnet starts with an already hijacked node instead (gives initial power)
            return set(range(self.size))

        # TODO: implementation feasible in O(n) instead of O(n*ln(n)) (or maybe O(n²)...)
        res = set()
        state = state.to_list()
        for i in state:
            res.update(self.graph[i])  # Adds the neighbours of node i
        for i in state:
            res.discard(i)             # Removes the nodes we already hijacked
        return res

    def current_power(self, state):
        """
        Returns the sum of the proselytism of the hijacked nodes, plus the initial power
        :param state: 
        :return:  
        """
        return self.initial_power + sum(self.proselytism[i] for i in range(self.size) if i in state)

    def total_power(self):
        """
        Total available power (including initial power)
        :return: 
        """
        # TODO: memoize this if used often
        return self.current_power(State.full_state(self.size))

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
        # TODO; change by an exponential?
        if self.get_resistance(action) == 0:
            return 1.
        return min(1., float(power) / self.get_resistance(action))

    def success_probability(self, action, state):
        """
        Same as success_probability_power, but with a state
        :param action: the node to hijack
        :param state:  the current state
        :return:       the probability of success
        """
        power = self.current_power(state)
        return self.success_probability_power(action, power)

    def attempt_hijacking(self, action, state):
        """
        Attempts to hijack the given node
        :param action: 
        :param state: 
        :return:       True if the attack succeeded, False otherwise
        """
        rnd = random.random()
        probability = self.success_probability(action, state)
        return rnd < probability

    def immediate_reward(self, state, action):
        """
        Computes the immediate reward (for one turn only). Please note this depends not on the result of the action.
        :param state: 
        :param action: 
        :return:       the immediate reward (independent on the success) 
        """
        # TODO: Remplacer par une notion de cout agréable, et fusionner les deux fonctions suivantes
        # TODO: Unify with the rest (see learning algorithms), and make it depend on the success?
        if action in state:
            return -self.get_cost(action)
        return -self.get_cost(action) + self.current_power(state)  # No reason to do a max with 0...

    def immediate_reward_power(self, power, action):
        """
        Same as immediate_reward, but with a power instead of a state.
        Please note that this assumes the node was not already part of the botnet!
        :param power: 
        :param action: 
        :return: 
        """
        return -self.get_cost(action) + power

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
            self.add_link(b, a)
            return A != B

        n = self.size
        while n > 1:
            a = random.randint(0, self.size-1)
            b = random.randint(0, self.size-1)
            if a != b and union(a, b):
                n -= 1

    def compute_percolation(self):
        """
        Computes the percolation of the graph of the network.
        :return: the percolation
        """
        perc = [0] * self.size
        visited = [0] * self.size
        prov = [0] * self.size
        count = 1
        for i in range(self.size):
            for j in range(self.size):
                prov[j] = -1

            # We compute with a bfs the last node prov[j]
            # on a shortest path between i and j
            queue = [(i, -1)]
            while len(queue) > 0:
                cur, last = queue.pop()
                if visited[cur] < count:
                    visited[cur] = count
                    prov[cur] = last
                    for v in self.graph[cur]:
                        queue.append((v, cur))

            # For each node j, we follow the shortest path from
            # j to i using the array prov computed previously
            for j in range(self.size):
                if i != j and visited[j] == count:
                    cur = j
                    while cur != -1:
                        perc[cur] += 1
                        cur = prov[cur]
            count += 1
        return perc


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

    for _ in range(size):
        big = random.random() < big_nodes

        if big:
            proselytism = random.randint(0, int(size ** difficulty))
        else:
            proselytism = random.randint(0, size)

        resistance = int((3*random.random()+1)/2 * (proselytism ** difficulty))
        cost = int(random.random() * proselytism)

        network.add(resistance, proselytism, cost)

    if complete:
        network.set_complete_network()
    else:
        network.generate_random_connected()
    return network
