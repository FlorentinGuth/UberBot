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
        self.graph.append([])

    def add_link(self, node1, node2):
        """
        :param node1:
        :param node2:
        :return: adds the edge node1-node2 to the graph of the network
        """
        # TODO Décider si c'est orienté, et propager ces modifications dans le reste, par exemple en ajoutant
        # un attribut accessibles (ensemble des noeuds accesibles depuis l'etat courant)
        # Possibilité de modifier le graphe en des ensembles ?
        # Ajouter une methode get_actions(self, state) ou qqch comme ca
        self.graph[node1].append(node2)

    def current_power(self, state):
        return self.initial_power + sum(self.proselytism[i] for i in range(self.size) if i in state)

    def total_power(self):
        return self.current_power(State.full_state(self.size))

    def get_cost(self, action):
        return self.action_cost[action]

    def get_proselytism(self, node):
        return self.proselytism[node]

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
        if action in state:
            return -self.get_cost(action)
        return -self.get_cost(action) + self.current_power(state)
