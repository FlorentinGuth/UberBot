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
nb_trials = 10
window = nb_trials // 10
size = 12
difficulty = 2
big_ratio = log(size) / size

n = random_network(size, difficulty, big_ratio)

qs = [fast.Fast(n),
      fast_incr.Fast(n),
      fast_tentative.Fast(n),
      Qstar(n, 0.9),
      Qlearning(n, 0.9, 0.01, strat=full_random),
      Thompson(n, 0.9, 0.01, strat=thompson_standard),
      ModelBasedThompson(n, 0.9, 0.01, strat=thompson_standard),
      FullModelBasedThompson(n, 0.9, 0.1, strat=thompson_standard)]

for q in qs:
    print(q.type)
    if isinstance(q, Qlearning):
        nb = nb_trials
        wd = window
    else:
        nb = 1
        wd = 1
    r = get_rewards(nb, q)
    plot_perf(r, wd, q.type)

legend()
show()

