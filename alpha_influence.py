import tests
import examples
from network import network_from_file

networks = ["simpleatk.gr", "W08atk.gr"]

# hyper_parameter_influence(botnet, network, nb_trials, hyper_param, values)

hyper_parameter = "alpha"
values = [1., 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.001]
redundancy = 5

for name in networks:
    network = network_from_file("graphs//" + name)[0]
    for b in examples.botnets(network, 0.9):
        tests.hyper_parameter_influence(b, network, 200, hyper_parameter, values, redundancy, is_log=True)
