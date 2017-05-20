import tests
import examples
from network import network_from_file
import numpy as np
networks = ["simpleatk.gr", "W08atk.gr"]
import math

# hyper_parameter_influence(botnet, network, nb_trials, hyper_param, values)

hyper_parameter = "alpha"
values = 10**(np.linspace(-3, 0, 19))
redundancy = 5


def launch_tests():
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        for b in examples.botnets(network, 0.9):
            tests.hyper_parameter_influence(b, network, 200, hyper_parameter, values, redundancy, is_log=True)


def show_results():
    botnets_names = ["Sarsa", "QLearning", "QLearning - Potential", "ThompsonSampling"]
    for name in networks:
        network = network_from_file("graphs//" + name)[0]
        print(name)
        for b in botnets_names:
            print(b)
            [actions, times, rewards, values] = tests.load_file("results/alpha_%d_%s.out"%(network.size, b))
            values = [math.log(v, 10) for v in values]
            tests.plot_with_legend(values, times, "Time")
            tests.plot_with_legend(values, rewards, "Reward")
            tests.show_with_legend()

"""
Generates files :
alpha_*_*, Saved object [actions, times, rewards, alpha_values]
action is size len(alpha_values) * redundancy
[actions, times, rewards, alpha_values] = load_file("results/alpha_network_nameBotnet")
"""
launch_tests()