__author__ = 'Martin'
import random
from matplotlib import *


class Q:
    def __init__(self):
        self.content = dict()
        self.inf = 1
        self.actions = []
        self.states = []

    def set(self, key, value):
        self.content[key] = value

    def get(self, key):
        try:
            return self.content[key]
        except KeyError:
            return 0

    def max_line(self, line):
        return max(self.content[key] for key in self.content.keys() if key[0] == line)

    def update_q_learning(self, r, w, alpha, si, a, sf):
        self.set((si, a), (1 - alpha) * self.get((si, a)) + alpha * (r(si, a) + w * self.max_line(sf)))

    def update_fix_point(self, r, w, p):
        states = []
        actions = []

        for s in states:
            for a in actions:
                #La somme est en fait sur deux états seulement !
                self.set((s, a), r(s, a) + w * sum(p(s, a, s2) * self.max_line(s2) for s2 in states))

    def policy(self, s):
        #Fonction Dstar

        best_q = -self.inf
        best_actions = []

        for a in self.actions:
            new_q = self.get((s, a))
            if new_q > best_q:
                best_q = new_q
                best_actions = [a]

            elif new_q == best_q:
                best_actions.append(a)

        if len(best_actions) == 0:
            return None

        return random.choice(best_actions)


#A présent, appliquons le au cas particulier ici traité du botnet.
class Qbis:
    def __init__(self, n, gamma, alpha=0., inf=1000):
        self.content = dict()
        self.actions = list(range(n))
        self.gamma = gamma
        self.alpha = alpha
        self.inf = inf

    def clear(self):
        self.content = dict()

    def set(self, state, action, value):
        self.content[(state.get(), action)] = value

    def get(self, state, action):
        try:
            return self.content[(state.get(), action)]
        except KeyError:
            return 0

    def exists(self, state, action):
        return (state.get(), action) in self.content

    def max_line(self, state):
        return max(self.get(state, action) for action in self.actions)

    def update_q_learning(self, network, si, a, sf):
        reward = network.R(si, a)
        self.set(si, a, (1 - self.alpha) * self.get(si, a) + self.alpha * (reward + self.gamma * self.max_line(sf)))

    def update_fix_point(self, network):
        #Pas du tout bien défini ce que ca devrait faire pour rester n**k (k = nombre itérations)
        states = []
        actions = []

        for s in states:
            for a in actions:
                #Il faut détailler la formule.
                self.set(s, a, network.R(s, a) + self.gamma * 0)

    def ex_value(self, state, action, network):
        #On suppose que toutes les valeurs présentes ont été calculées avec ça aussi.
        if self.exists(state, action):
            return self.get(state, action)

        #On ne renvoie la bonne réponse que quand c'est l'action optimale, une minoration sinon.
        res = network.R(state, action)

        splusa = state.add(action)
        proba_s_to_splusa = network.p(action, state)

        maxQ = -self.inf
        for h in self.actions:
            if h in splusa:
                continue

            newQ = self.ex_value(splusa, h, network)

            if newQ > maxQ:
                maxQ = newQ

        if maxQ == -self.inf:
            #Cas où il ne reste aucune cible.
            maxQ = 0

        res += self.gamma * proba_s_to_splusa * maxQ
        res /= (1 - self.gamma * (1 - proba_s_to_splusa))

        self.set(state, action, res)
        return res

    def policy(self, state):
        #Fonction Dstar (en supposant que Qstar a été calculée)

        best_q = -self.inf
        best_actions = []

        for action in self.actions:
            if action in state:
                continue

            new_q = self.get(state, action)

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            return None

        #print("Expected best value : ", best_q)
        return random.choice(best_actions)

    def ex_policy(self, state, network):
        #Fonction Dstar calculée avec le truc exact (exponentiel).

        best_q = -self.inf
        best_actions = []

        for action in self.actions:
            if action in state:
                continue

            new_q = self.ex_value(state, action, network)

            if new_q > best_q:
                best_q = new_q
                best_actions = [action]

            elif new_q == best_q:
                best_actions.append(action)

        if len(best_actions) == 0:
            return None

        print("Expected best value : ", best_q)
        return random.choice(best_actions)


class State:
    def __init__(self, code=0):
        self.content = code

    def add(self, node):
        return State(self.content + (1 << node))

    def __contains__(self, item):
        return (self.content >> item) & 1

    def get(self):
        return self.content


class Network:
    def __init__(self, initial_power):
        self.resistance = []
        self.proselytism = []
        self.action_cost = []
        self.hijacked = State()
        self.size = 0
        self.initial = initial_power
        self.reward = 0
        self.time_factor = 1
        self.gamma = 0.9 #A modifier
        pass

    def add(self, resistance, proselytism, cost):
        self.action_cost.append(cost)
        self.proselytism.append(proselytism)
        self.resistance.append(resistance)
        self.size += 1

    def R(self, state, action):
        if action in state:
            return -self.cost(action)

        return -self.cost(action) + self.r(state)

    def r(self, state):
        return self.initial + sum(self.proselytism[i] for i in range(self.size) if i in state)

    def cost(self, action):
        return self.action_cost[action]

    def success(self, action, state):
        return min(1, 1.0 * self.r(state) / self.resistance[action])

    def take_action(self, action):
        rnd = random.random()
        probability = self.success(action, self.hijacked)

        self.reward += self.time_factor * self.R(self.hijacked, action)
        self.time_factor *= self.gamma

        if rnd < probability:
            self.hijacked = self.hijacked.add(action)
            return True
        else:
            return False

    def reset(self):
        self.hijacked = State()
        self.reward = 0
        self.time_factor = 1

    def get_state(self):
        return self.hijacked

    def p(self, action, state):
        return self.success(action, state)

    def remaining(self):
        return self.size - sum(1 for node in range(self.size) if node in self.hijacked)
