from state import State
from network import Network
from qlearning import QLearning
from policy import Policy

# import sys
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
import random
import pickle


def plot_with_legend(x_axis, y_axis, legend):
    """
    Plots the given function using a legend
    :param x_axis: 
    :param y_axis: 
    :param legend: 
    :return: 
    """
    plot(x_axis, y_axis, label=legend)


def show_with_legend():
    """
    Shows the current plot, along with its legend
    :return: 
    """
    font_p = FontProperties()
    font_p.set_size('small')

    legend(loc='lower right', bbox_to_anchor=(1, 0.5), prop=font_p).draggable()
    show()


def invade(botnet, network, printing=False):
    """
    Let the botnet invade once the network.
    :param botnet:   a (learning) botnet, set up on this network, and which must have been cleared beforehand
    :param network:  the network to invade
    :param printing: if True, prints all the details about the invasions
    :return:         a list of all (action, result), the total reward received, and the expected reward of the induced 
                     policy (which is constructed by taking the successful actions)
    """
    actions = []
    reward = 0
    policy = []

    t = 0
    while not botnet.state.is_full():
        action = botnet.choose_action()

        if printing:
            print("Action ", t)
            print("Remaining nodes = %s" % botnet.state.nb_remaining())
            print("Attack %s!" % action)

        success = network.attempt_hijacking(botnet.state, action)
        if success:
            policy.append(action)
            if printing:
                print("Success\n")
        else:
            if printing:
                print("Failure\n")

        immediate_reward = network.immediate_reward(botnet.state, action)
        if success and botnet.state.add(action).is_full():
            immediate_reward += botnet.gamma * network.final_reward(botnet.gamma)
        reward += botnet.time_factor * immediate_reward
        botnet.receive_reward(action, success, immediate_reward)

        actions.append((action, success))
        t += 1

    return actions, reward, Policy(network, policy).expected_reward(botnet.gamma)


def train(botnet, network, nb_trials, printing=False):
    """
    Trains the botnet by doing multiple invasions on the same network.
    Please note that this function does not modify the nb_trials parameter of the botnet, neither takes it into account!
    :param botnet:    a (learning) botnet, set up on this network, and which must have been cleared beforehand
    :param network: 
    :param nb_trials: number of invasions to do, not necessarily equal to botnet.nb_trials
    :param printing: 
    :return:          the list of the real rewards,
                      the list of the expected rewards of the induced policy (see invade documentation),
                      the list of the expected rewards of the computed policy
    """
    real_rewards = []
    expected_rewards = []
    policy_rewards = []
    for _ in range(nb_trials):
        _actions, reward, expected_reward = invade(botnet, network, printing)
        real_rewards.append(reward)
        expected_rewards.append(expected_reward)
        policy_rewards.append(Policy(network, botnet.compute_policy()).expected_reward(botnet.gamma))
        botnet.clear()
    return real_rewards, expected_rewards, policy_rewards


def test_botnet(botnet, network, nb_trials, window_size=1, real_rewards=False, induced_rewards=False, policy_rewards=True, show=False):
    """
    Plots the expected reward of the induced policy over trainings, and prints the expected reward of the computed policy.
    :param botnet: 
    :param network: 
    :param nb_trials: 
    :param window_size:
    :param real_rewards:    whether to plot the real rewards received during the training
    :param induced_rewards: whether to plot the expected rewards of the induced policy
    :param policy_rewards:  whether to plot the expected rewards of the full-exploitation policy
    :param show:            if True, shows the results
    :return: 
    """
    rewards, expected, policy = train(botnet, network, nb_trials)

    if real_rewards:
        plot_with_legend(list(range(nb_trials)), soften(rewards,  window_size), legend=botnet.type+" real")
    if induced_rewards:
        plot_with_legend(list(range(nb_trials)), soften(expected, window_size), legend=botnet.type+" induced")
    if policy_rewards:
        plot_with_legend(list(range(nb_trials)), soften(policy,   window_size), legend=botnet.type+" policy")
    print(botnet.compute_policy())
    print(botnet.type, Policy(network, botnet.compute_policy()).expected_reward(botnet.gamma), sep='\t')
    if show:
        show_with_legend()


def hyper_parameter_influence(botnet, network, nb_trials, hyper_param, values):
    """
    Plots expected time and reward of the given botnet with respect to the hyper parameter.
    :param botnet: 
    :param network: 
    :param nb_trials: 
    :param hyper_param: the name of the hyper parameter
    :param values:      the set of values to test
    :return: 
    """
    times = []
    rewards = []

    for value in values:
        botnet.__setattr__(hyper_param, value)
        _ = train(botnet, network, nb_trials)
        policy = botnet.compute_policy()
        botnet.clear(all=True)

        times.append(policy.expected_time())
        rewards.append(policy.expected_reward(botnet.gamma))

    plot_with_legend(values, times, "Time")
    plot_with_legend(values, rewards, "Reward")
    show_with_legend()


def soften(points, window_size):
    """
    :param points:      signal
    :param window_size: width of the window used to compute the mean
    :return:            list of points of the local mean of the input signal
    """
    n = len(points)
    soft_points = []
    cur_sum = 0
    for i in range(window_size):
        cur_sum += points[i]
        soft_points.append(cur_sum / (i + 1))

    for i in range(window_size, n):
        cur_sum += points[i] - points[i-window_size]
        soft_points.append(cur_sum / window_size)

    return soft_points


def plot_perf(points, window_size=1, name=None):
    """
    :param points:      signal to plot
    :param window_size: size of the window used to call soften function
    :return:            plots the obtained local mean
    """
    soft_points = soften(points, window_size)
    plot(range(len(points)), soft_points, label=name)


def sample_optimal(botnet, network):
    """
    """
    policy = botnet.compute_policy()
    actions = []
    for action in policy:
        success = False
        while not success:
            success = network.attempt_hijacking(botnet.state, action)
            actions.append((action, success))

        immediate_reward = network.immediate_reward(botnet.state, action)
        if success and botnet.state.add(action).is_full():  # TODO: include all this in network somehow
            immediate_reward += botnet.gamma * network.final_reward(botnet.gamma)
        botnet.receive_reward(action, success, immediate_reward)

    return actions


def dump_actions(test_name, network_name, botnet_name, actions):
    """
    Dump the sequence of actions in a file
    :param test_name:    The name of the test, e.g. alpha_influence
    :param network_name: "iotatk", "simpleatk", "W08atk"...
    :param botnet_name:  Botnet.type
    :param actions:      A list of trials, which are the list of (action, success)
    :return: 
    """
    filename = "results/" + test_name + "_" + network_name + "_" + botnet_name + ".out"
    with open(filename, 'w') as f:
        pickle.dump(actions, f)

def retrieve_actions(test_name, network_name, botnet_name):
    """
    Retrieves the history of actions from a result file
    :param test_name: 
    :param network_name: 
    :param botnet_name: 
    :return: what was dumped in the file
    """
    filename = "results/" + test_name + "_" + network_name + "_" + botnet_name + ".out"
    with open(filename, 'r') as f:
        actions = pickle.load(f)
    return actions