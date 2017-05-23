from state import *
from network import Network


def immediate_shaping_potential(network, gamma):
    return lambda s: 1 / (1 - gamma) * network.current_power(s)