from markov import QStar
from thompson_sampling import Thompson, ModelBasedThompson, FullModelBasedThompson
from qlearning import QLearning
from oriented_exploration import OrientedExploration
from sarsa import Sarsa
from strategy import *
from tests import *
from fast_tentative import *
from reward_tentative import *
from math import *
from network import *
from matplotlib.font_manager import FontProperties
from shaping import immediate_shaping_potential

fontP = FontProperties()
fontP.set_size('medium')

# Martin's pet network (that's cute)
size = 13
delta = 2.

n_martin = Network(1)
for i in range(size):
    n_martin.add_node(i ** delta + 1, i + 1, i ** delta)
n_martin.set_complete_network()
n_martin.add_initial_node(12)


def botnets(network, gamma, nb_trials=200):
    """
    :param network:
    :return: The list of all botnets parametrized with the given network
    """
    potential = immediate_shaping_potential(network, gamma)
    qs = [
        # Fast(network),
        FastTentative(network),

        # QStar(network, gamma),
        Sarsa(full_exploration, network.graph, gamma=gamma, initial_nodes=network.initial_nodes),
        RewardTentative(network, gamma),

        QLearning(full_exploration, network.graph, gamma=gamma, initial_nodes=network.initial_nodes),
        OrientedExploration(full_exploration, network.graph, gamma=gamma, initial_nodes=network.initial_nodes),
        QLearning(full_exploration, network.graph, gamma=gamma, initial_nodes=network.initial_nodes, potential=potential),
        Thompson(thompson_standard, network.graph, gamma=gamma, nb_trials=nb_trials, initial_nodes=network.initial_nodes),
        ModelBasedThompson(thompson_standard, network.graph, gamma=gamma, nb_trials=nb_trials, initial_nodes=network.initial_nodes),
        FullModelBasedThompson(thompson_standard, network.graph, gamma=gamma, nb_trials=nb_trials, initial_nodes=network.initial_nodes, alpha_p=0.05),
    ]
    return qs
botnet_names = [q.type for q in botnets(Network(0), 0.9)]


def learning_botnets(network, gamma=0.9):
    return filter(lambda q: not isinstance(q, Botnet), botnets(network, gamma))

learning_botnet_names = [q.type for q in learning_botnets(Network(0), 0.9)]


def non_learning_botnets(network, gamma=0.9):
    return filter(lambda q: isinstance(q, Botnet), botnets(network, gamma))

non_learning_botnet_names = [q.type for q in non_learning_botnets(Network(0), 0.9)]


def plot_learning(nb_trials, window, network, gamma=0.9):
    """
    Plots the performance of the qs, with respect to the trainings.
    Be aware that the results that counts are the final ones, which are the printed ones.
    :return: None
    """

    qs = botnets(network, gamma)

    for q in qs:

        test_botnet(q, network, nb_trials, window)

        if isinstance(q, FullModelBasedThompson):
            # Plots the internal estimates of this botnet
            estimates = [x[1] for x in q.history]
            plot_perf(estimates, window, "Estimates of FullModelBasedThompson")

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
                perf.append(Policy(network, q.compute_policy()).expected_reward(q.gamma))

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
network = network_from_file("graphs/W08atk.gr")[0]
plot_learning(200, 10, network)
# size = 20
# difficulty = 2
# network = random_network(size, difficulty, big_nodes=log(size) / float(size), complete=False)
# plot_learning(200, 10, network)
