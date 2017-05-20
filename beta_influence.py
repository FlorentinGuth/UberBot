import tests
from network import network_from_file
import numpy as np
from thompson_sampling import Thompson
from strategy import thompson_standard

networks = ["W08atk.gr"]

hyper_parameter = "beta"
values = np.linspace(0, 1, 21)
redundancy = 5


def launch_tests():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        botnet = Thompson(thompson_standard, network.graph, gamma=0.9, nb_trials=200, initial_nodes=network.initial_nodes)
        tests.hyper_parameter_influence(botnet, network, 200, hyper_parameter, values, redundancy, is_log=False)


def show_results():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        b = Thompson(thompson_standard, network.graph, gamma=0.9, nb_trials=200, initial_nodes=network.initial_nodes)
        [actions, times, rewards, values] = tests.load_file("results/beta_%d_%s.out"%(network.size, b.type))
        tests.plot_with_legend(values, times, "Time")
        tests.plot_with_legend(values, rewards, "Reward")
        tests.show_with_legend()

"""
Generates files :
beta_*_*, Saved object [actions, times, rewards, alpha_values]
action is size len(alpha_values) * redundancy
[actions, times, rewards, alpha_values] = load_file("results/beta_network_nameBotnet")
"""
show_results()
