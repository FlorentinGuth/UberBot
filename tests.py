from state import State
from network import Network
import fast
import fast_incr
import fast_tentative
from qlearning import Qlearning
# import sys

from matplotlib.pyplot import *
import random

# TODO Fonction de génération de graphes aléatoires
# TODO Fonction de génération d'ordinateurs aléatoires
# TODO Reprendre les algorithmes, les structures de données, les optimiser !


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
                print("Remaining nodes = %s" % q.state.nb_remaining())
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
    """
    :param nb: number of invasions to lead
    :param q: botnet to train
    :return: actions of the botnet according to its final policy
    """
    return try_invasions(nb, q, printing=False)[-1]


def get_rewards(nb, q, printing=False):
    """
    :param nb: number of invasions to lead
    :param q: botnet to train
    :param printing: enables information printing
    :return: successive rewards obtained at the end of each invasion
    """
    if isinstance(q, Qlearning):
        return try_invasions(nb, q, printing)[0]
    return [q.compute_policy().value(q.gamma)]*nb


def soft(points, window_size):
    """
    :param points: signal
    :param window_size: width of the window used to compute the mean
    :return: list of points of the local mean of the input signal
    """
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


def plot_perf(points, window_size=1, name=None):
    """
    :param points: signal to plot
    :param window_size: size of the window used to call soft function
    :return: plots the obtained local mean
    """
    soft_points = soft(points, window_size)
    plot(range(len(points)), soft_points, label=name)
