import random


# TODO: Delete this file (the smart code is in the exploration, not in the strategy)
def strategy(botnet):
    """
    Prototype of a strategy. A strategy dictates how exploration and exploitation should be balanced during the training.
    You do not need to deal with the last training case, this is handled in test.py which uses compute_policy instead.
    :param botnet: learning botnet
    :return:       action to take in the q.state according to this strategy
    """
    return 0


def full_exploration(botnet):
    """
     Full exploration strategy available for every learning botnet.
     :param botnet:
     :return:
    """
    return botnet.exploration()


def full_exploitation(botnet):
    """
    Full exploitation strategy available for every learning botnet.
    :param botnet: 
    :return: 
    """
    return botnet.exploitation()


def thompson_standard(botnet):
    """
    Uses random at the beginning and then uses more and more Thompson policy.
    This strategy is available if the number of trials is not None.
    :param botnet: 
    :return: 
    """

    phi = lambda x: 1.*x  # Controls the decrease of random use : phi = 1 means no random, 0 means full random.

    if random.random() > phi(botnet.completed_trials / float(botnet.nb_trials)):
        return botnet.exploration()

    return botnet.exploitation()


def curious_standard(botnet):
    """
    Same as thompson_standard but uses curiosity instead. Available on Thompson Botnets.
    :param botnet: a thompson botnet
    :return: 
    """

    phi = lambda x: x  # Controls the decrease of curiosity. phi = 1 means no curiosity, 0 means full curiosity.

    if random.random() > phi(botnet.completed_trials / float(botnet.nb_trials)):
        return botnet.be_curious()

    return botnet.exploitation()
