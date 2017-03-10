import random


def strategy(q, nb_tot, i):
    """
    :param q: botnet
    :param nb_tot: total number of invasions
    :param i: current number of the invasion
    :return: action to take in the q.state according to this strategy
    """
    return 0


def full_random(q, nb_tot, i):
    """
     Full random strategy, except in the last round, available for every Qlearning botnet.
    """
    if i == nb_tot - 1:
        return q.policy(q.state)

    return q.random_action()


def thompson_standard(q, nb_tot, i):
    """
     Uses random at the beginning and then uses more and more Thompson policy.
    """

    phi = lambda x: x  # Controls the decrease of random use : phi = 1 means no random, 0 means full random.

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
