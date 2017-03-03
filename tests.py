from state import State
from markov import Qstar
from thompson_sampling import Thomson
from network import Network
from qlearning import Qlearning

from matplotlib.pyplot import *
import random

# TODO Thompson Sampling
# TODO Comprendre, modifier la fonction de Q learning (et son initialisation / exploration)
# TODO Ranger un peu tout ça dans des fichiers différents et des fonctions propres
# TODO Réécrire tous ces tests pour qu'ils soient fonctionnels ET corrects

"""
#Test State
s = State(0)
print(1 in s)
s = s.add(1)
print(1 in s)
"""


def known(nb, q, affichage=False):
    # Approche où tout est connu
    invasions = []
    durees = []
    rewards = []

    for j in range(nb):
        if affichage:
            print("Invasion N°", j)
        i = 1
        actions = []
        while not (q.state.is_full()):
            si = q.state
            action = q.ex_policy(si)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % q.state.nb_remaining())
                print("Attack %s!" % action)

            if q.take_action(action):
                if affichage:
                    print("Success")
                    print("\n")
                actions.append(action)
            elif affichage:
                print("Failure")
                print("\n")

            i += 1

        if affichage:
            print(actions)

        # print(actions)
        #
        # exp_rew = expected_reward(State(), actions, n, q.gamma)
        # print(exp_rew, q.static_info(n, True)[1])

        rewards.append(q.reward)
        q.reset()
        invasions.append(actions)
        durees.append(i - 1)

    # print(durees)
    # print(q.content.items())
    # print(invasions[-10:])

    return rewards, sum(durees) / len(durees), invasions[-1]


def unknown(nb, q, r=0., alpha=0.01, affichage=False):
    # Approche ou rien n'est connu, avec nb attaques (avec mémoire).

    q.alpha = alpha
    invasions = []
    durees = []
    rewards = []

    for j in range(nb):
        # Essai de modif des alpha WARNING alpha est modifie donc ne sert a rien ci-dessus
        # q.alpha = 0.5 * (1 - (j / nb) ** 1) ** 2
        if affichage:
            print("Invasion N°", j)
        i = 1
        actions = []
        while not q.state.is_full():
            si = q.state

            bound = r * (1 - (j / nb) ** 1)

            if random.random() < bound:
                action = q.random_action()

            else:
                action = q.policy(q.state)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % q.remaining())
                print("Attack %s!" % action)

            if q.take_action(action):
                if affichage:
                    print("Success")
                    print("\n")
                actions.append(action)
            elif affichage:
                print("Failure")
                print("\n")

            # TODO Check that the two states are different ? (In fact, they are the same !)
            q.update_q_learning(si, action, q.state)
            # assert(n.hijacked.content == si.content)

            i += 1
        if affichage:
            print(actions)

        rewards.append(q.reward)
        q.reset()
        invasions.append(actions)
        durees.append(i - 1)

    # print(durees)
    # print(q.content.items())
    # print(invasions[-10:])

    return rewards, sum(durees) / len(durees), invasions[-1]


def unknown_thomson(nb, q, r=0., alpha=0.01, affichage=False):
    # Approche ou rien n'est connu, avec nb attaques (avec mémoire).
    q.alpha = alpha
    invasions = []
    durees = []
    rewards = []

    for j in range(nb):
        # Essai de modif des alpha.. WARNING alpha est modifie donc ne sert a rien ci-dessus
        # q.alpha = 0.5 * (1 - (j / nb) ** 1) ** 2

        if affichage:
            print("Invasion N°", j)
        i = 1
        actions = []
        while not q.state.is_full():
            si = q.state  # TODO Get a real copy or find a trick to Q learn directly.

            bound = r * (1 - (j / nb) ** 1)

            if random.random() < bound:
                action = q.random_action()

            else:
                action = q.thomson_policy(si)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % q.state.remaining())
                print("Attack %s!" % action)

            res_action = q.take_action(action)
            if res_action:
                if affichage:
                    print("Success")
                    print("\n")
                actions.append(action)
            elif affichage:
                print("Failure")
                print("\n")

            # Update through Thomson Sampling method
            q.add_trial(action, si, res_action, q.immediate_reward(si, action))

            i += 1
        if affichage:
            print(actions)

        rewards.append(q.reward)
        q.reset()
        invasions.append(actions)
        durees.append(i - 1)

    # print(durees)
    # print(q.content.items())
    # print(invasions[-10:])

    return rewards, sum(durees) / len(durees), invasions[-1]

# Test
# Premier réseau
"""
n = Network(1)
n.add(2, 1, 1)
n.add(20, 3, 1)
q = Qbis(2, 0.9)
"""

# Deuxieme réseau
k = 12
n = Network(1)
for i in range(k):
    n.add(i**1+1, i+1, i)
q1 = Qstar(n, 0.9)
q2 = Qlearning(n, 0.9, 0.1)
q3 = Thomson(n, 0.9, 0.1)

nb = 1000
y1 = known(nb, q1)[0]
print(y1)

y2 = unknown(nb, q2, 1, 0.001)[0]
y3 = unknown_thomson(nb, q3, 1, 0.01)
plot(range(nb), y1)
moy1 = sum(y1) / nb
moy2 = sum(y2) / nb

plot(range(nb), y2)
plot(range(nb), [moy1] * nb)
plot(range(nb), [moy2] * nb)
show()


def avg_length_known(nb, gammas, n, q):
    res = []

    for gamma in gammas:
        print("Gamma = ", gamma)
        q.gamma = gamma
        res.append(known(nb, n, q)[1])
        q.clear()

    plot(gammas, res)
    show()

    print("Durees avec gamma :", res)

# avg_length_known(200, [0.8 + 0.0025 * i for i in range(80)], n, q)


def avg_length_mieux(gammas, n, q):
    res_t = []
    res_v = []
    for gamma in gammas:
        print(gamma)
        q.gamma = gamma
        (t, v) = q.static_infos(n, True)
        res_t.append(t)
        res_v.append(v)
        q.clear()
    plot(gammas, res_t)
    plot(gammas, res_v)
    show()

# avg_length_mieux([0.8 + 0.0025 * i for i in range(80)], n, q)


def learning(nb, q, rs=None):
    # Est supposé mesurer la force de l'aléa sur l'apprentissage
    if rs is None:
        rs = [0.]
    res_t = []
    res_reward = []

    for r in rs:
        # reward = sum(unknown(nb, n, q, r, 0.01)[0][-10:]) / 10
        actions = unknown(nb, q, r, 0.01)[2]
        print(actions)
        pol = q.compute_policy()
        res_t.append(pol.expected_time())
        res_reward.append(pol.expected_reward(State(q.network.size), q.gamma, 0))

        q.clear()

    # plot(rs, res_t)
    plot(rs, res_reward)

    # The lines below are supposed to be computed according to exact policy.
    # t, reward = ...
    # plot(rs, [t] * len(rs))
    # plot(rs, [reward] * len(rs))

    show()

# learning(1000, n, q, [0.1 * i for i in range(11)])

# y1 = known(1, n, q)[0]
# unknown(1000, n, q, True)
