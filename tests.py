from debuts import Network, Qbis, State, expected_reward
from matplotlib.pyplot import *
import random

#TODO Thompson Sampling
#TODO Comprendre, modifier la fonction de Q learning (et son initialisation / exploration)
#TODO Ranger un peu tout ça dans des fichiers différents et des fonctions propres

"""
#Test State
s = State(0)
print(1 in s)
s = s.add(1)
print(1 in s)
"""


def known(nb, n, q, affichage=False):
    #Approche où tout est connu
    invasions = []
    durees = []
    rewards = []

    for j in range(nb):
        if affichage:
            print("Invasion N°", j)
        i = 1
        actions = []
        while n.remaining() != 0:
            si = n.hijacked
            action = q.ex_policy(si, n)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % n.remaining())
                print("Attack %s!" % action)

            if n.take_action(action):
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
        # print(exp_rew, q.static_infos(n, True)[1])

        rewards.append(n.reward / (i - 1))
        n.reset()
        invasions.append(actions)
        durees.append(i - 1)

    #print(durees)
    #print(q.content.items())
    #print(invasions[-10:])

    return rewards, sum(durees) / len(durees), invasions[-1]


def unknown(nb, n, q, r=0., alpha=0.01, affichage=False):
    #Approche ou rien n'est connu, avec nb attaques (avec mémoire).
    q.alpha = alpha
    invasions = []
    durees = []
    rewards = []

    for j in range(nb):
        #Essai de modif des alpha
        q.alpha = 0.5 * (1 - (j / nb) ** 1) ** 2
        if affichage:
            print("Invasion N°", j)
        i = 1
        actions = []
        while n.remaining() != 0:
            si = n.hijacked

            bound = r * (1 - (j / nb) ** 1)

            if random.random() < bound:
                action = random.choice([a for a in range(n.size) if not a in n.hijacked])

            else:
                action = q.policy(si)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % n.remaining())
                print("Attack %s!" % action)

            q.update_q_learning(n, si, action, n.hijacked)

            if n.take_action(action):
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

        rewards.append(n.reward)
        n.reset()
        invasions.append(actions)
        durees.append(i - 1)

    #print(durees)
    #print(q.content.items())
    #print(invasions[-10:])

    return rewards, sum(durees) / len(durees), invasions[-1]

#Test
#Premier réseau
"""
n = Network(1)
n.add(2, 1, 1)
n.add(20, 3, 1)
q = Qbis(2, 0.9)
"""

#Deuxieme réseau
k = 12
n = Network(1)
for i in range(k):
    n.add(i**1+1, i+1, i)
q = Qbis(k, 0.9, alpha=0.05)

"""
nb = 1000
y1 = known(nb, n, q)[0]
q.clear()
y2 = unknown(nb, n, q, 1, 0.001)[0]
plot(range(nb), y1)
moy1 = sum(y1) / nb
moy2 = sum(y2) / nb

plot(range(nb), y2)
plot(range(nb), [moy1] * nb)
plot(range(nb), [moy2] * nb)
show()
"""


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

#avg_length_known(200, [0.8 + 0.0025 * i for i in range(80)], n, q)


def avg_length_mieux(gammas, n, q):
    T = []
    V = []
    for gamma in gammas:
        print(gamma)
        q.gamma = gamma
        (t, v) = q.static_infos(n, True)
        T.append(t)
        V.append(v)
        q.clear()
    plot(gammas, T)
    plot(gammas, V)
    show()

#avg_length_mieux([0.8 + 0.0025 * i for i in range(80)], n, q)


def learning(nb, n, q, rs=None):
    #Est supposé mesurer la force de l'aléa sur l'apprentissage
    if rs is None:
        rs = [0.]
    res_t = []
    res_reward = []

    for r in rs:
        # reward = sum(unknown(nb, n, q, r, 0.01)[0][-10:]) / 10
        actions = unknown(nb, n, q, r, 0.01)[2]
        print(actions)
        t, _ = q.static_infos(n)
        res_t.append(t)
        res_reward.append(expected_reward(State(), actions, n, q.gamma))

        q.clear()

    t, reward = q.static_infos(n, True)

    # plot(rs, res_t)
    plot(rs, res_reward)
    # plot(rs, [t] * len(rs))
    plot(rs, [reward] * len(rs))

    show()

learning(1000, n, q, [0.1 * i for i in range(11)])

# y1 = known(1, n, q)[0]