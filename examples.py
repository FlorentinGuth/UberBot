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

fontP = FontProperties()
fontP.set_size('small')

# Martin's pet network (that's cute)
size = 13
delta = 2

n_martin = Network(1)
for i in range(size):
    n_martin.add_node(i ** delta + 1, i + 1, i ** delta)
n_martin.set_complete_network()


# def botnets(network):
#     """
#     :param network:
#     :return: The list of all botnets parametrized with the given network
#     """
#     # TODO: would be more practical if we had a class of list: [Fast, Fast_incr, QLearning...] (no need to give the botnet)
#     qs = [
#         fast.Fast(network),
#         fast_incr.FastIncr(network),
#         fast_tentative.FastTentative(network),
#
#         RewardIncr(network),
#         QStar(network, 0.9),
#
#         QLearning(network, 0.9, 0.01, strat=full_random, shape=False),
#         # Thompson(network, 0.9, 0.01, strat=curious_standard),
#         # ModelBasedThompson(network, 0.9, 0.01, strat=thompson_standard),
#         # FullModelBasedThompson(network, 0.9, 0.1, strat=thompson_standard),
#     ]
#     return qs
# botnet_names = [q.type for q in botnets(Network(0))]
#
#
# def learning_botnets(network):
#     return filter(lambda q: isinstance(q, QLearning), botnets(network))
#
# learning_botnet_names = [q.type for q in learning_botnets(Network(0))]
#
#
# def non_learning_botnets(network):
#     return filter(lambda q: not isinstance(q, QLearning), botnets(network))
#
# non_learning_botnet_names = [q.type for q in non_learning_botnets(Network(0))]
#
#
# def plot_learning(nb_trials, window, network):
#     """
#     Plots the performance of the qs, with respect to the trainings.
#     Be aware that the results that counts are the final ones, which are the printed ones.
#     :return: None
#     """
#
#     qs = botnets(network)
#
#     for q in qs:
#
#         if isinstance(q, QLearning):
#             r = get_rewards(nb_trials, q)
#
#         else:
#             pol = q.compute_policy()
#             r = pol.expected_reward(q.gamma)
#
#             print(q.type, r, pol.expected_time(), sep='\t')
#             r = [r] * nb_trials
#
#         plot_perf(r, window, q.type, )
#
#         if isinstance(q, FullModelBasedThompson):
#             # Plots the internal estimates of this botnet
#             estimates = [x[1] for x in q.history]
#             plot_perf(estimates, window, "Estimates of FullModelBasedThompson")
#
#     # legend(loc="lower right")
#     legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=fontP).draggable()
#
#     show()
#
#
# def plot_immediate(max_size, nb_trials, difficulty):
#     """
#     Plots the performance of the non-learning botnets, with respect to the size of the network
#     :return: None
#     """
#
#     nb_botnets = len(non_learning_botnet_names)
#
#     sizes = list(range(1, max_size + 1))
#     perfs = []
#
#     for size in sizes:
#         print(size)
#
#         trials = []
#         for _ in range(nb_trials):
#             network = random_network(size, difficulty, big_nodes=log(size) / float(size))
#             network.set_complete_network()
#
#             perf = []
#             for q in non_learning_botnets(network):
#                 perf.append(q.compute_policy().expected_reward(q.gamma))
#
#             trials.append(perf)
#
#         average = [sum(trials[k][q] for k in range(nb_trials)) for q in range(nb_botnets)]
#         perfs.append(average)
#
#     for i in range(nb_botnets):
#         name = non_learning_botnet_names[i]
#         perf_i = [perfs[k][i] for k in range(len(sizes))]
#         plot(sizes, perf_i, label=name)
#
#     legend(loc="lower right")
#     show()


# plot_immediate(10, 20, 2)
# plot_learning(100, 10, n_martin)

for B in [QStar, Fast, FastIncr, FastTentative, RewardIncr]:
    q = B(n_martin, 0.9)
    pol = Policy(n_martin, q.compute_policy())
    print(q.type, pol.expected_time(), pol.expected_reward(0.9), sep='\t')

test_botnet(QLearning(full_exploration, n_martin.graph, alpha=0.01), n_martin, 1000, 1, True)
