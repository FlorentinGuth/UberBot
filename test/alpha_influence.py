import tests
import examples
from network import network_from_file
import numpy as np
import math
import os
os.chdir("..")

networks = ["W08atk.gr"]

hyper_parameter = "alpha"
values = 10**(np.linspace(-3, 0, 2))
redundancy = 1
nb_trials = 10


def launch_tests():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        for b in examples.botnets(network, 0.9, nb_trials):
            tests.hyper_parameter_influence(b, network, nb_trials, hyper_parameter, values, redundancy, is_log=True)


def show_results():
    botnets_names = ["Sarsa", "QLearning", "QLearning - Potential", "ThompsonSampling"]
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        print(name)
        for b in botnets_names:
            print(b)
            [actions, times, rewards, values] = tests.load_file("results/alpha_%d_%s_%d.out"%(network.size, b, nb_trials))
            values = [math.log(v, 10) for v in values]
            tests.plot_with_legend(values, times, "Time")
            tests.plot_with_legend(values, rewards, "Reward")
            tests.show_with_legend()

launch_tests()
# show_results()