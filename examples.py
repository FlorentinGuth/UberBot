from markov import QStar
from thompson_sampling import Thompson, ModelBasedThompson, FullModelBasedThompson
from qlearning import QLearning
from strategy import *
from tests import *
from fast import *
from fast_incr import *
from fast_tentative import *
from reward_incr import *
from math import *
from network import *
from matplotlib.font_manager import FontProperties
# TODO FastIncr and RewardIncr don't take into account the starting nodes, nor the network !
fontP = FontProperties()
fontP.set_size('small')

# Martin's pet network (that's cute)
size = 13
delta = 2.

n_martin = Network(1)
for i in range(size):
    n_martin.add_node(i ** delta + 1, i + 1, i ** delta)
n_martin.set_complete_network()
n_martin.add_initial_node(12)


def botnets(network):
    """
    :param network:
    :return: The list of all botnets parametrized with the given network
    """
    qs = [
        Fast(network),
        FastIncr(network),
        FastTentative(network),

        RewardIncr(network),
        QStar(network, 0.9),

        QLearning(full_exploration, network.graph, initial_nodes=network.initial_nodes),
        Thompson(thompson_standard, network.graph, nb_trials=200, initial_nodes=network.initial_nodes),
        ModelBasedThompson(thompson_standard, network.graph, nb_trials=200, initial_nodes=network.initial_nodes),
        FullModelBasedThompson(thompson_standard, network.graph, nb_trials=200, initial_nodes=network.initial_nodes),
    ]
    return qs
botnet_names = [q.type for q in botnets(Network(0))]


def learning_botnets(network):
    return filter(lambda q: not isinstance(q, Botnet), botnets(network))

learning_botnet_names = [q.type for q in learning_botnets(Network(0))]


def non_learning_botnets(network):
    return filter(lambda q: isinstance(q, Botnet), botnets(network))

non_learning_botnet_names = [q.type for q in non_learning_botnets(Network(0))]


def plot_learning(nb_trials, window, network):
    """
    Plots the performance of the qs, with respect to the trainings.
    Be aware that the results that counts are the final ones, which are the printed ones.
    :return: None
    """

    qs = botnets(network)

    for q in qs:

        test_botnet(q, network, nb_trials, window)

        # if isinstance(q, FullModelBasedThompson):
        #     # Plots the internal estimates of this botnet
        #     estimates = [x[1] for x in q.history]
        #     plot_perf(estimates, window, "Estimates of FullModelBasedThompson")

    show_with_legend()


def plot_immediate(max_size, nb_trials, difficulty):
    """
    Plots the performance of the non-learning botnets, with respect to the size of the network
    :return: None
    """

    nb_botnets = len(non_learning_botnet_names)

    sizes = list(range(1, max_size + 1))
    perfs = []

    for size in sizes:
        print(size)

        trials = []
        for _ in range(nb_trials):
            network = random_network(size, difficulty, big_nodes=log(size) / float(size))
            network.set_complete_network()

            perf = []
            for q in non_learning_botnets(network):
                perf.append(q.compute_policy().expected_reward(q.gamma))

            trials.append(perf)

        average = [sum(trials[k][q] for k in range(nb_trials)) for q in range(nb_botnets)]
        perfs.append(average)

    for i in range(nb_botnets):
        name = non_learning_botnet_names[i]
        perf_i = [perfs[k][i] for k in range(len(sizes))]
        plot(sizes, perf_i, label=name)

    legend(loc="lower right")
    show()


# plot_immediate(10, 20, 2)
print(Policy(n_martin, [4, 5, 3, 6, 2, 7, 1, 0, 8, 9, 10, 11]).expected_reward(0.9))
print(Policy(n_martin, [2, 3, 5, 4, 1, 6, 0, 7, 8, 9, 10, 11]).expected_reward(0.9))
print(Policy(n_martin, [0, 1, 2, 4, 5, 6, 3, 7, 8, 9, 11, 10]).expected_reward(0.9))

plot_learning(200, 10, n_martin)
