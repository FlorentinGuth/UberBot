import tests
import examples
from network import network_from_file
import numpy as np
import os
os.chdir("..")

networks = ["W08atk.gr"]

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
            [actions, times, rewards, values] = tests.load_file("results/gamma_%d_%s_%d.out"%(network.size, b, nb_trials))
            tests.plot_with_legend(values, times, b)
        tests.show_with_legend()

# launch_tests()
show_results()