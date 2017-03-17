from markov import Qstar
from thompson_sampling import Thompson, ModelBasedThompson, FullModelBasedThompson
from qlearning import Qlearning
from strategy import *
from tests import *
import fast
import fast_incr
import fast_tentative
from math import *
from network import *


# Parameters
nb_trials = 1000
window = nb_trials // 10
size = 12
difficulty = 2
big_ratio = log(size) / size

n = random_network(size, difficulty, big_ratio)

# n = Network(1)
# delta = 0.
#
# for i in range(size):
#     n.add(i ** delta + 1, i + 1, i ** delta)
# n.set_complete_network()

qs = [fast.Fast(n),
      fast_incr.Fast(n),
      fast_tentative.Fast(n),
      Qstar(n, 0.9),
      Qlearning(n, 0.9, 0.01, strat=full_random, shape=False),
      Thompson(n, 0.9, 0.01, strat=curious_standard),
      ModelBasedThompson(n, 0.9, 0.01, strat=thompson_standard),
      FullModelBasedThompson(n, 0.9, 0.1, strat=thompson_standard)]

# for q in qs:
#
#     if isinstance(q, Qlearning):
#         r = get_rewards(nb_trials, q)
#
#     else:
#         r = q.compute_policy().value(q.gamma)
#         print(q.type, r)
#         r = [r] * nb_trials
#     plot_perf(r, window, q.type)
#
# legend(loc="lower right")
# show()

