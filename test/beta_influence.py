import tests
from network import network_from_file
import numpy as np
from thompson_sampling import Thompson
from strategy import thompson_standard
import os
os.chdir("..")

networks = ["W08atk.gr"]

hyper_parameter = "beta"
values = np.linspace(0, 1, 21)
redundancy = 5
nb_trials = 200


def launch_tests():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        botnet = Thompson(thompson_standard, network.graph, gamma=0.9, nb_trials=nb_trials, initial_nodes=network.initial_nodes)
        tests.hyper_parameter_influence(botnet, network, nb_trials, hyper_parameter, values, redundancy, is_log=False)


def show_results():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        b = "ThompsonSampling"
        [actions, times, rewards, values] = tests.load_file("results/beta_%d_%s_%d.out"%(network.size, b, nb_trials))
        tests.plot_with_legend(values, times, "Time")
        tests.plot_with_legend(values, rewards, "Reward")
        tests.show_with_legend()

# launch_tests()
show_results()
