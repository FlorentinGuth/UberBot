import tests
from network import network_from_file
import numpy as np
from thompson_sampling import FullModelBasedThompson
from strategy import thompson_standard
import math
import os
os.chdir("..")

networks = ["W08atk.gr"]

hyper_parameter = "alpha_p"
values = 10**(np.linspace(-3, 0, 19))
redundancy = 5
nb_trials = 200


def launch_tests():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        botnet = FullModelBasedThompson(thompson_standard, network.graph, gamma=0.9, nb_trials=nb_trials, initial_nodes=network.initial_nodes)
        tests.hyper_parameter_influence(botnet, network, 200, hyper_parameter, values, redundancy, is_log=True)


def show_results():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        b = FullModelBasedThompson(thompson_standard, network.graph, gamma=0.9, nb_trials=nb_trials, initial_nodes=network.initial_nodes)
        [actions, times, rewards, values] = tests.load_file("results/alpha_p_%d_%s_%d.out" % (network.size, b.type, nb_trials))
        values = [math.log(v, 10) for v in values]
        tests.plot_with_legend(values, times, "Time")
        tests.plot_with_legend(values, rewards, "Reward")
        tests.show_with_legend()

# launch_tests()
show_results()
