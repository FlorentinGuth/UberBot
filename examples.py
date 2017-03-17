from markov import Qstar
from thompson_sampling import Thompson, ModelBasedThompson, FullModelBasedThompson
from qlearning import Qlearning
from strategy import *
from tests import *

# Bleu Vert Rouge BleuCiel

# Premier réseau
"""
n = Network(1)
n.set_complete_network()
n.add(2, 1, 1)
n.add(20, 3, 1)
q = Qbis(2, 0.9)
"""

# Deuxieme réseau
k = 12
n = Network(1)
delta = 1.

for i in range(k):
    n.add(i**delta+1, i+1, i**delta)
n.generate_random_connected()
# n.set_complete_network()

qs =
q1 = Qstar(n, 0.9)
q2 = Qlearning(n, 0.9, 0.01, strat=full_random)
q3 = Thompson(n, 0.9, 0.01, strat=thompson_standard)
q4 = ModelBasedThompson(n, 0.9, 0.01, strat=thompson_standard)
q5 = FullModelBasedThompson(n, 0.9, 0.1, strat=thompson_standard)

# nb = 100
# r1 = get_rewards(nb, q1)
# r2 = get_rewards(nb, q2)
# r3 = get_rewards(nb, q3)
# r4 = get_rewards(nb, q4)
# r5 = get_rewards(nb, q5)
#
# window = nb // 10
# plot_perf(r1, nb)
# plot_perf(r2, window)
# plot_perf(r3, window)
# plot_perf(r4, window)
# plot_perf(r5, window)
#
# estimated_rewards = [x[1] for x in q5.history]
# plot_perf(estimated_rewards, window)
# show()

