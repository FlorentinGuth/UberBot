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


def full_exploration(q):
    """
     Full exploration strategy available for every learning botnet.
    """
    return q.exploration()


def thompson_standard(q, nb_tot, i):
    """
     Uses random at the beginning and then uses more and more Thompson policy.
    """

    phi = lambda x: 1.*x  # Controls the decrease of random use : phi = 1 means no random, 0 means full random.

    if i == nb_tot - 1:
        return q.policy(q.state)

    if random.random() > phi(i / nb_tot):
        return q.random_action()

    return q.thompson_policy(q.state)


def greedy(q, nb_tot, i):
    """
     Greedy policy, only exploits current approximation.
    """
    return q.policy(q.state)


def curious_standard(q, nb_tot, i):
    """
     Same as random_standard but uses curiosity instead. Available on Thompson policy.
    """

    phi = lambda x: x  # Controls the decrease of curiosity. phi = 1 means no curiosity, 0 means full curiosity.

    if i == nb_tot - 1:
        return q.policy(q.state)

    if random.random() > phi(i / nb_tot):
        return q.be_curious(q.state)

    return q.thompson_policy(q.state)
