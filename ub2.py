import random
import math

INF = 10000
F = [0, 1]
resistance = [2, 20]
proselytism = [1, 3]
fixed_cost = [1, 1]
gamma = 0.9
initial_strength = 1

"""
n = 5
F = list(range(n))
resistance = list(range(1, n+1))
proselytism = list(range(n))
fixed_cost = [1] * n
gamma = 0.9
initial_strength = 1"""


def R(st, at):
    if (at, 1) in st:
        return -C(at)
    else:
        return -C(at) + r(st)


def r(st):
    return initial_strength + sum(B(g) for (g, x) in st if x == 1)


def take_action(st, at):
    rnd = random.random()
    thr = A(at, r(st))

    if rnd < thr:
        return st + [(at, 1)]
    else:
        return st + [(at, 0)]


def A(at, strength):
    res = resistance[at]

    success_proba = min(1.0, 1.0*strength/res)
    return success_proba


def B(g):
    return proselytism[g]


def C(g):
    return fixed_cost[g]


def Qstar(s, a):
    res = R(s, a)
    S = 0
    for g in F:
        if (g, 1) in s:
            continue
        proba_s_to_splusg = A(g, r(s))
        splusg = s + [(g, 1)]
        maxQ = -1
        for h in F:
            if (h, 1) in splusg:
                continue
            newQ = Qstar(splusg, h)
            if newQ > maxQ:
                maxQ = newQ
        S += maxQ*proba_s_to_splusg
    res += gamma * S

    return res


def Qstar_bis(s, a):
    #On renvoie la bonne réponse que quand a est l'action optimale, une minoration sinon.
    res = R(s, a)
    S = 0

    proba_s_to_splusa = A(a, r(s))
    splusa = s + [(a, 1)]
    maxQ = -INF
    for h in F:
        if (h, 1) in splusa:
            continue
        newQ = Qstar_bis(splusa, h)

        if newQ > maxQ:
            maxQ = newQ
    if maxQ == -INF:
        #Cas où il ne reste aucune cible.
        maxQ = 0

    res += gamma * proba_s_to_splusa * maxQ

    return res / (1 - gamma * (1 - proba_s_to_splusa))


def Dstar(s):
    #Ne faudrait-il pas mettre 0 ?
    maxQ = -INF
    maxA = None

    for a in F:
        if (a, 1) in s:
            continue
        newQ = Qstar_bis(s, a)
        print(a, newQ)
        if newQ > maxQ:
            maxQ = newQ
            maxA = a

    if maxA is None:
        #Quand est-ce que ça pourrait arriver ? On préfèrerait ne faire aucune action ?
        print("Random action chosen")
        maxA = random.choice([g for g in F if (g, 1) not in s])
    return maxA


def remaining(s):
    return len(F)-sum(x for (g, x) in s)



s = []
for i in range(10000):
    print("Action ", i)
    print("Remaining nodes = %s" % remaining(s))
    newAction = Dstar(s)
    print("Attack %s!" % newAction)
    s = take_action(s, newAction)
    if remaining(s) == 0:
        print("Attack complete.")
        break
