from math import exp
from matplotlib import pyplot
"""
horizon = 30
nb = 10000
times = [k * horizon / float(nb) for k in range(nb)]
lambdas = [0.1, 0.5, 1, 2]
points = [[exp(-l*t) * l for t in times] for l in lambdas]

for l in range(len(lambdas)):
    pyplot.plot(times, points[l], label=str(lambdas[l]))
pyplot.legend()
pyplot.show()
"""

def space(n):
    binom = 1
    sum = 0
    for k in range(n):
        sum += binom * k ** (n - k)
        binom = (binom * (n - k)) // (k + 1)
    return sum

def strange_sum(n):
    sum = 1
    k_fact = 1
    for k in range(1, n ** 2):
        k_fact *= k
        sum += (k ** n) / float(k_fact)
    return sum


from math import log

x = []
y = []
z = []
for n in range(10):
    x.append(n)
    y.append(space(n))
    z.append(strange_sum(n))
pyplot.plot(x, y, 'g')
pyplot.plot(x, z, 'b')
pyplot.show()
