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
        self.graph = []

    def add(self, resistance, proselytism, cost):
        self.action_cost.append(cost)
        self.proselytism.append(proselytism)
        self.resistance.append(resistance)
        self.size += 1
        self.graph.append(set())

    def add_link(self, node1, node2):
        """
        :param node1:
        :param node2:
        :return: adds the edge node1-node2 to the graph of the network
        """
        # un attribut accessibles (ensemble des noeuds accesibles depuis l'etat courant)
        # Possibilité de modifier le graphe en des ensembles ?
        # Ajouter une methode get_actions(self, state) ou qqch comme ca
        self.graph[node1].add(node2)

    def set_complete_network(self):
        tot = set(range(self.size))
        for i in range(self.size):
            self.graph[i] = tot

    def clear_graph(self):
        for i in range(self.size):
            self.graph[i] = set()

    def get_actions(self, state):
        if state.is_empty():
            return set(range(self.size))

        res = set()
        state = state.to_list()
        for i in state:
            res.update(self.graph[i])
        for i in state:
            res.discard(i)
        return res

    def current_power(self, state):
        return self.initial_power + sum(self.proselytism[i] for i in range(self.size) if i in state)

    def total_power(self):
        return self.current_power(State.full_state(self.size))

    def get_cost(self, action):
        return self.action_cost[action]

    def get_proselytism(self, node):
        return self.proselytism[node]

    def get_resistance(self, node):
        return self.resistance[node]

    def success_probability_power(self, action, power):
        if self.resistance[action] == 0:
            return 1.
        return min(1., float(power) / self.resistance[action])

    def success_probability(self, action, state):
        power = self.current_power(state)
        return self.success_probability_power(action, power)

    def attempt_hijacking(self, action, state):
        rnd = random.random()
        probability = self.success_probability(action, state)
        return rnd < probability

    def immediate_reward(self, state, action):
        # TODO Remplacer par une notion de cout agréable
        if action in state:
            return -self.get_cost(action)
        return -self.get_cost(action) + self.current_power(state)

    def immediate_reward_power(self, power, action):
        """ Immediate reward, assumes that the node has not been hijacked yet! """
        return -self.get_cost(action) + power

    def generate_random_connected(self):
        rep = [i for i in range(self.size)]

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
        perc = [0] * self.size
        visited = [0] * self.size
        prov = [0] * self.size
        count = 1
        for i in range(self.size):
            for j in range(self.size):
                prov[j] = -1

            queue = [(i, -1)]
            while len(queue) > 0:
                cur, last = queue.pop()
                if visited[cur] < count:
                    visited[cur] = count
                    prov[cur] = last
                    for v in self.graph[cur]:
                        queue.append((v, cur))

            for j in range(self.size):
                if i != j and visited[j] == count:
                    cur = j
                    while cur != -1:
                        perc[cur] += 1
                        cur = prov[cur]
            count += 1
        return perc


def random_network(size, difficulty, big_nodes):
    """
    Returns a random network with given size.
    For each node, we can expect the resistance to be approximately equivalent to its proselytism ** difficulty.
    Big_nodes is the ratio (between 0 and 1) of hard-to-hijack nodes.

    How the network is computed:
     - For each node, we test at random if it will be a big one
     - Big nodes have a proselytism uniformly between 0 and size ** difficulty, small ones between 0 and size
     - The resistance is between a half and the double of the proselytism ** difficulty
     - The cost is a random fraction of the resistance
     - The edges are computed from Network.generate_random_connected()
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

    # network.generate_random_connected()
    network.set_complete_network()
    return network
