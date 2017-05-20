import tests
import examples
from network import network_from_file
import numpy as np
networks = ["W08atk.gr"]
import math

hyper_parameter = "gamma"
values = np.linspace(0., 0.999, 21)
redundancy = 1
nb_trials = 200


def launch_tests():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        for b in examples.botnets(network, 0.9, nb_trials):
            tests.hyper_parameter_influence(b, network, nb_trials, hyper_parameter, values, redundancy)


def show_results():
    botnets_names = ["RewardTentative", "Sarsa", "QLearning", "QLearning - Potential", "ThompsonSampling",
                     "ModelBasedThompson", "FullModelBasedThompson"]
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        print(name)
        for b in botnets_names:
            print(b)
            [actions, times, rewards, values] = tests.load_file("results/gamma_%d_%s.out"%(network.size, b))
            tests.plot_with_legend(values, times, b)
        tests.show_with_legend()

"""
Generates files :
gamma_*_*, Saved object [actions, times, rewards, gamma_values]
action is size len(gamma_values) * redundancy
[actions, times, rewards, alpha_values] = load_file("results/gamma_network_nameBotnet")
"""
# launch_tests()
show_results()