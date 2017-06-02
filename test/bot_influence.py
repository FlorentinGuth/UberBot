import examples
from network import network_from_file
import tests
import numpy as np
from policy import Policy
from matplotlib.pyplot import *
import os
os.chdir("..")

networks = ["W08atk.gr"]
redundancy = 1
nb_trials = 200


def launch_tests():
    for name in networks:
        print(name)
        network = network_from_file("graphs//" + name)[0]
        results = []
        for b in examples.botnets(network, 0.9, nb_trials):
            print(b.type)
            real = []
            exp = []
            policy = []
            action = []
            perf = []
            time = []
            for i in range(redundancy):
                print("Trial ", i)
                real_rewards, expected_rewards, policy_rewards = tests.train(b, network, nb_trials)
                real.append(real_rewards)
                exp.append(expected_rewards)
                policy.append(policy_rewards)
                action.append(b.compute_policy())
                p = Policy(network, b.compute_policy())
                perf.append(p.expected_reward(b.gamma))
                time.append(p.expected_time())
                b.clear(all=True)
            results.append((b.type, real, exp, policy, action, perf, time))
        tests.dump_actions("botnet_Compare", name, "all", results, nb_trials)


def show_results(nb):
    window_size = int(nb / 20)
    for name in networks:
        res = tests.load_file("results/botnet_Compare_%s_all_%d.out" % (name, nb))
        network = network_from_file("graphs//" + name)[0]
        real_res = []
        exp_res = []
        pol_res = []
        final_perf = []
        for x in res:
            (b, real_rewards, expected_rewards, policy_rewards, actions, perf, time) = x
            real_rewards = np.array(real_rewards)
            expected_rewards = np.array(expected_rewards)
            policy_rewards = np.array(policy_rewards)
            perf = np.array(perf)
            time = np.array(time)
            real_rewards = np.mean(real_rewards, axis=0)
            expected_rewards = np.mean(expected_rewards, axis=0)
            policy_rewards = np.mean(policy_rewards, axis=0)
            real_res.append((b, real_rewards))
            exp_res.append((b, expected_rewards))
            pol_res.append((b, policy_rewards))
            final_perf.append((b, np.mean(perf), np.max(perf), np.mean(time), np.min(time), actions))

        f = figure(1)
        for (name, perf) in real_res:
            tests.plot_perf(perf, window_size, name)
        ylabel("Real rewards")
        tests.show_with_legend(f)
        f = figure(2)
        for (name, perf) in exp_res:
            tests.plot_perf(perf, window_size, name)
        ylabel("Expected rewards")
        tests.show_with_legend(f)
        f = figure(3)
        for (name, perf) in pol_res:
            tests.plot_perf(perf, window_size, name)
        ylabel("Policy rewards")
        tests.show_with_legend(f)

        for (b_name, mean_p, max_p, mean_t, min_t, actions) in final_perf:
            print(b_name, "Mean perf:", mean_p, "Max perf:", max_p, "Mean time:", mean_t, "Min time:", min_t, "Actions:", actions, sep='\t')
        x = input("Enter something to exit :")

"""
We have chosen  gamma = 0.9, nb_trials =  50, redundancy = 100;
                gamma = 0.9, nb_trials = 100, redundancy = 50;
                gamma = 0.9, nb_trials = 200, redundancy = 20;
                gamma = 0.9, nb_trials = 500, redundancy =  5;
"""

# launch_tests()
# show_results(50)
# show_results(100)
# show_results(200)
# show_results(500)
