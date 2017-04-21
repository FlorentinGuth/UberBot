
import matplotlib.pyplot as plt
import tests
import network
import strategy
import thompson_sampling as thom

n = 11

net = network.Network(1)

for i in range(n):
    net.add_node(i**2, i, i)

net.generate_random_connected()

q = thom.Thompson(strategy.thompson_standard, net.graph, 0.9, 0.01)

tests.train(q, net, 100)
q.clear()
l = tests.invade(q, net)[0]

net.viz_save()
for (i, r) in l:
    if r:
        net.succeed_hijack(i)
    else:
        net.fail_hijack(i)
    net.viz_save()
    if r:
        net.done_hijack(i)
    else:
        net.clear_hijack(i)
net.viz_save()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()
plt.show()

for i in range(net.vtime):
    fname = "Images/" + str(i) + ".png"
    im = plt.imread(fname)
    img = ax.imshow(im)
    plt.draw()
    accept = input('OK? ')
