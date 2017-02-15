from ISP_Naccache.debuts import Network, Qbis
from matplotlib.pyplot import *

__author__ = 'Martin'

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

        rewards.append(n.reward / (i - 1))
        n.reset()
        invasions.append(actions)
        durees.append(i - 1)

    print(durees)
    #print(q.content.items())
    print(invasions[-10:])

    return rewards


def unknown(nb, n, q, alpha=0.1, affichage=False):
    #Approche ou rien n'est connu, avec nb attaques (avec mémoire).
    q.alpha = alpha
    invasions = []
    durees = []
    rewards = []

    for j in range(nb):
        print("Invasion N°", j)
        i = 1
        actions = []
        while n.remaining() != 0:
            si = n.hijacked
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

        rewards.append(n.reward / (i - 1))
        n.reset()
        invasions.append(actions)
        durees.append(i - 1)

    print(durees)
    #print(q.content.items())
    print(invasions[-10:])

    return rewards

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
q = Qbis(k, 0.9, alpha=0.03)


nb = 10000
y1 = known(nb, n, q)
q.clear()
y2 = unknown(nb, n, q, 0.001)
plot(range(nb), y1)
moy1 = sum(y1) / nb
moy2 = sum(y2) / nb

plot(range(nb), y2)
plot(range(nb), [moy1] * nb)
plot(range(nb), [moy2] * nb)
show()