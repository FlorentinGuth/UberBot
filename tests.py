from state import State
from markov import Qstar
from thompson_sampling import Thomson
from network import Network
from qlearning import Qlearning
import fast
import fast_tentative
import fast_incr
import sys

from matplotlib.pyplot import *
import random

# TODO Comprendre, modifier la fonction de Q learning (et son initialisation / exploration)
# TODO Recoder une fonction d'invasion uniquement à partir de la méthode choose_action
# TODO Réorganiser pour n'avoir que des trucs propres, et pareil pour l'affichage

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
            si = q.state.copy()
            action = q.ex_policy(si)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % q.state.nb_remaining())
                print("Attack %s!" % action)
            res_action = q.take_action(action)

            if res_action:
                if affichage:
                    print("Success")
                    print("\n")
            elif affichage:
                print("Failure")
                print("\n")
            actions.append((action, res_action))
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
    print("Opt ", q.compute_policy().expected_reward(State(q.network.size), q.gamma, 0))
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
            si = q.state.copy()

            # bound = r * (1 - (j / nb) ** 2)
            # bound = 1
            # if random.random() < bound:
            #     action = q.random_action()
            #
            # else:
            #     action = q.policy(q.state)

            action = q.choose_action(nb, j)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % q.remaining())
                print("Attack %s!" % action)

            res_action = q.take_action(action)

            if res_action:
                if affichage:
                    print("Success")
                    print("\n")

            elif affichage:
                print("Failure")
                print("\n")

            actions.append((action, res_action))

            # q.update_q_learning(si, action, q.state)

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
    print("Q learn ", q.compute_policy().expected_reward(State(q.network.size), q.gamma, 0))
    return rewards, sum(durees) / len(durees), invasions[-1]


def unknown_thomson(nb, q, r=0., alpha=0.01, affichage=False):
    # Approche ou rien n'est connu, avec nb attaques (avec mémoire).
    q.alpha = alpha
    invasions = []
    durees = []
    rewards = []

    for j in range(nb):
        # Essai de modif des alpha.. WARNING alpha est modifie donc ne sert a rien ci-dessus
        # q.alpha = 0.5 * (1 - (j / nb) ** 2) ** 1

        if affichage:
            print("Invasion N°", j)
        i = 1
        actions = []
        while not q.state.is_full():
            si = q.state.copy()

            # bound = r * (1 - (j / nb) ** 1)
            # WARNING MODIF TODO
            # if random.random() < bound or r == 1:
            #     action = q.random_action()
                # action = q.be_curious(si)
            # else:
            #     if j == nb - 1:
            #         action = q.policy(si)
            #     else:
            #         action = q.thomson_policy(si)
            action = q.choose_action(nb, j)

            if affichage:
                print("Action ", i)
                print("Remaining nodes = %s" % q.state.remaining())
                print("Attack %s!" % action)

            res_action = q.take_action(action)
            if res_action:
                if affichage:
                    print("Success")
                    print("\n")

            elif affichage:
                print("Failure")
                print("\n")

            # Update through Thomson Sampling method
            # q.add_trial(action, si, res_action, q.immediate_reward(si, action))

            actions.append((action, res_action))
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
    print("Thompson ", q.compute_policy().expected_reward(State(q.network.size), q.gamma, 0))

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
    n.add(i**1.+1, i+1, i)

q1 = Qstar(n, 0.9)
q2 = Qlearning(n, 0.9, 0.1)
q3 = Thomson(n, 0.9, 0.1)

# nb = 1000
# print("1...")
# y1 = known(nb, q1)[0]
# # print(y1)
# print("2...")
# y2 = unknown(nb, q2, 1, 0.01)[0]
# print("3...")
# y3 = unknown_thomson(nb, q3, 1, 0.01)[0]
# plot(range(nb), y1)
#
# moy1 = sum(y1) / nb
# moy2 = sum(y2) / nb
#
# plot(range(nb), y2)
# print(len(y3))
# plot(range(nb), y3)
#
# plot(range(nb), [moy1] * nb)
# # plot(range(nb), [moy2] * nb)
# show()


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


def liozoub(nb, q, r=1., alpha=0.01, affichage=False):
    res = None
    if isinstance(q, Thomson):
        res = unknown_thomson(nb, q, r, 0.01, affichage)[-1]
    elif isinstance(q, Qlearning):
        res = unknown(nb, q, r, alpha, affichage)[-1]
    elif isinstance(q, Qstar):
        res = known(nb, q)[-1]

    return q.network.size, res


def results(nb, q, r=1., alpha=0.01, affichage=False):
    res = None
    if isinstance(q, Thomson):
        res = unknown_thomson(nb, q, r, 0.01, affichage)[0]
    elif isinstance(q, Qlearning):
        res = unknown(nb, q, r, alpha, affichage)[0]
    elif isinstance(q, Qstar):
        res = known(nb, q)[0]

    return res


def soft(points, window_size):
    n = len(points)
    soft_points = []
    cur_sum = 0
    for i in range(window_size):
        cur_sum += points[i]
        soft_points.append(cur_sum / (i + 1))

    for i in range(window_size, n):
        cur_sum += points[i] - points[i-window_size]
        soft_points.append(cur_sum / window_size)

    return soft_points


def plot_perf(points, window_size=1):
    soft_points = soft(points, window_size)
    plot(range(len(points)), soft_points)


def test_fast():
    sys.setrecursionlimit(999999)

    network = Network(1)
    size = []
    f_time = []
    ft_time = []
    fi_time = []
    for n in range(1, 11):
        print(n)
        network.add(n**7, n, 0)
        size.append(n)

        pf = fast.Fast(network).compute_policy()
        print(pf.actions)
        f_time.append(pf.expected_time())

        pft = fast_tentative.Fast(network).compute_policy()
        ft_time.append(pft.expected_time())

        pfi = fast_incr.Fast(network).compute_policy()
        fi_time.append(pfi.expected_time())
    plot(size, f_time, color="green")
    plot(size, ft_time, color="blue")
    plot(size, fi_time, color="red")
    show()

def test_incr(trials, size):
    for _ in range(trials):
        nw = Network(1)
        for _ in range(size):
            nw.add(random.randint(0, size**2), random.randint(0, size), 0)
        p_opti = fast.Fast(nw).compute_policy()
        t_opti = p_opti.expected_time()
        p_incr = fast_incr.Fast(nw).compute_policy()
        t_incr = p_incr.expected_time()
        if t_opti != t_incr:
            print(nw.resistance, nw.proselytism)
            print(p_opti.actions, t_opti)
            print(p_incr.actions, t_incr)
            return


test_incr(200, 10)


# print(liozoub(100, q1))
# print(liozoub(100, q2))
# print(liozoub(1000, q3))
#
# r1 = results(1000, q1, 1, 0.01)
# r3 = results(1000, q3, 0.9, 0.01)
# print(r3)
# plot_perf(results(1000, q2, 0.99, 0.01), 50)
# plot_perf(r3, 50)
# plot_perf(r1, 1000)
# print([x for x in q3.content.items() if x[0][0] == 0])
# show()
