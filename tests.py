from state import State
from markov import Qstar
from thompson_sampling import Thompson
from network import Network
from qlearning import Qlearning
from strategy import *
import fast
import fast_incr
import fast_tentative
import sys

from matplotlib.pyplot import *
import random

# TODO Comprendre, modifier la fonction de Q learning (et son initialisation / exploration)
# TODO Recoder une fonction d'invasion uniquement à partir de la méthode choose_action
# TODO Réorganiser pour n'avoir que des trucs propres, et pareil pour l'affichage


def try_invasions(nb, q, printing=False):
    """
    :param nb: number of invasions
    :param q: botnet to test
    :param printing: Print all the details about the invasions.
    :return:
    """

    invasions = []
    durations = []
    rewards = []

    for j in range(nb):
        if printing:
            print("Invasion N°", j)

        i = 1
        actions = []
        while not q.state.is_full():
            action = q.choose_action(nb, j)

            if printing:
                print("Action ", i)
                print("Remaining nodes = %s" % q.state.remaining())
                print("Attack %s!" % action)

            res_action = q.take_action(action)

            if res_action:
                if printing:
                    print("Success")
                    print("\n")

            elif printing:
                print("Failure")
                print("\n")

            actions.append((action, res_action))
            i += 1

        if printing:
            print(actions)

        rewards.append(q.reward)
        q.reset()
        invasions.append(actions)
        durations.append(i - 1)

    print(q.type, q.compute_policy().value(q.gamma))

    return rewards, sum(durations) / len(durations), invasions[-1]

# Test
# Premier réseau
"""
n = Network(1)
n.set_complete_network()
n.add(2, 1, 1)
n.add(20, 3, 1)
q = Qbis(2, 0.9)
"""

# Deuxieme réseau
k = 12
n = Network(1)

for i in range(k):
    n.add(i**1.4+1, i+1, i)
n.set_complete_network()

q1 = Qstar(n, 0.9)
q2 = Qlearning(n, 0.9, 0.01, strat=full_random)
q3 = Thompson(n, 0.9, 0.01, strat=thompson_standard)


def gamma_influence(gammas, nb, q):
    """
    :param gammas: set of values of gamma
    :param nb: number of trials for each gamma
    :param q: botnet to test
    :return: the expected time and reward for each learnt policy for each gamma.
    """
    res_t = []
    res_v = []

    for gamma in gammas:
        q.gamma = gamma
        _ = try_invasions(nb, q)
        pol = q.compute_policy()
        t = pol.expected_time()
        v = pol.expected_reward(State(_), gamma, 0)

        res_t.append(t)
        res_v.append(v)
        q.clear()

    plot(gammas, res_t)
    plot(gammas, res_v)
    show()

    return res_t, res_v


def alpha_influence(alphas, nb, q):
    """
    :param alphas: set of values of alpha
    :param nb: number of trials for each alpha
    :param q: botnet to test
    :return: the expected time and reward for each learnt policy for each alpha.
    """
    res_t = []
    res_v = []

    for alpha in alphas:
        q.alpha = alpha
        _ = try_invasions(nb, q)
        pol = q.compute_policy()
        t = pol.expected_time()
        v = pol.expected_reward(State(_), alpha, 0)

        res_t.append(t)
        res_v.append(v)
        q.clear()

    plot(alphas, res_t)
    plot(alphas, res_v)
    show()

    return res_t, res_v


def get_last_invasion(nb, q):
    return try_invasions(nb, q, printing=False)[-1]


def test_incr(trials, size):
    for _ in range(trials):
        nw = Network(1)
        for __ in range(size):
            nw.add(random.randint(0, size**2), random.randint(0, size), 0)
        p_optimal = fast.Fast(nw).compute_policy()
        t_optimal = p_optimal.expected_time()
        p_incr = fast_incr.Fast(nw).compute_policy()
        t_incr = p_incr.expected_time()
        if t_optimal != t_incr:
            print(nw.resistance, nw.proselytism)
            print(p_optimal.actions, t_optimal)
            print(p_incr.actions, t_incr)
            return


def results(nb, q, printing=False):
    return try_invasions(nb, q, printing)[0]


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
    for n in range(1, 11):
        print(n)
        network.add(n, n**2, 0)
        size.append(n)

        pf = fast.Fast(network).compute_policy()
        f_time.append(pf.expected_time())

        pft = fast_tentative.Fast(network).compute_policy()
        ft_time.append(pft.expected_time())
    plot(size, f_time, color="blue")
    plot(size, ft_time, color="red")
    show()


# print(liozoub(100, q1))
# print(liozoub(100, q2))
# print(liozoub(1000, q3))

# r1 = results(1000, q1)
# r2 = results(1000, q2)
# r3 = results(1000, q3)
#
# plot_perf(r1, 1000)
# plot_perf(r2, 50)
# plot_perf(r3, 50)
#
# show()
