
from math import log
import tkinter as tk
import matplotlib.pyplot as plt
import tests
import network
import strategy
import thompson_sampling as thom
import time
from qlearning import QLearning

#n = 16
#difficulty = 2
##
##net = network.Network(1)
##
##for i in range(n):
##    net.add_node(i**1.5, i, i)
##
##net.generate_random_connected()
#net = network.random_network(n, difficulty, big_nodes=log(n)/float(n), complete=False)
net, _ = network.network_from_file("./graphs/simpleatk.gr")

q = thom.Thompson(strategy.thompson_standard, net.graph, 0.9, 0.01)
#q = QLearning(strategy.full_exploration, net.graph, nb_trials=200)

tests.train(q, net, 100)
q.clear()
l = tests.sample_optimal(q, net)

net.viz.layout()

#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.ion()
#plt.show()
#
#for i in range(net.vtime):
#    fname = "Images/" + str(i) + ".png"
#    im = plt.imread(fname)
#    img = ax.imshow(im)
#    plt.draw()
#    accept = input('OK? ')

coord = [net.viz.get_node(i).attr['pos'].split(',') for i in range(net.size)]
X = [(float(a)) for a, _ in coord]
X = [(x - min(X)) / max(X) * 1000 + 50 for x in X]
Y = [int(float(b)) for _, b in coord]
Y = [(y - min(Y)) / max(Y) * 1000 + 50 for y in Y]

class GraphGUI(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        self.graph = tk.Canvas(root, width=max(X)+50, height=max(Y)+50)
        self.nodes = []
        self.edges = []
        self.map = {}
        self.draw()

        self.graph.pack()

        master.bind('<Left>',  lambda x : self.backward())
        master.bind('<Right>', lambda x : self.forward())

        self.curT = 0
        self.last = -1

        # drag and drop (experimental)
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.last_draw = int(round(time.time() * 1000))

    def draw(self):
        self.graph.delete("all")
        self.nodes = []
        self.edges = []
        self.map = {}
        for i in range(net.size):
            for j in net.graph[i]:
                if j > i:
                    self.edges.append(self.graph.create_line(X[i], Y[i], X[j], Y[j]))
            if net.resistance[i] >= 10*1000:
                self.nodes.append(self.graph.create_oval(X[i]-40, Y[i]-40, X[i]+40, Y[i]+40, fill="#93a1a1", tags="node"))
            elif net.resistance[i] >= 1000:
                self.nodes.append(self.graph.create_oval(X[i]-35, Y[i]-35, X[i]+35, Y[i]+35, fill="#93a1a1", tags="node"))
            elif net.resistance[i] >= 100:
                self.nodes.append(self.graph.create_oval(X[i]-30, Y[i]-30, X[i]+30, Y[i]+30, fill="#93a1a1", tags="node"))
            else:
                self.nodes.append(self.graph.create_oval(X[i]-20, Y[i]-20, X[i]+20, Y[i]+20, fill="#93a1a1", tags="node"))
            self.map[self.nodes[-1]] = i
            self.graph.create_text(X[i], Y[i], text=str(int(net.resistance[i]))+"/"+str(int(net.proselytism[i])))
        self.graph.tag_bind("node", "<ButtonPress-1>", self.on_token_press)
        self.graph.tag_bind("node", "<ButtonRelease-1>", self.on_token_release)
        self.graph.tag_bind("node", "<B1-Motion>", self.on_token_motion)

    # dnd code, from SO
    def on_token_press(self, event):
        """
        Begining drag of an object
        """
        # record the item and its location
        self._drag_data["item"] = self.map[self.graph.find_closest(event.x, event.y)[0]]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_token_release(self, event):
        """
        End drag of an object
        """
        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        self.draw()

    def on_token_motion(self, event):
        """
        Handle dragging of an object
        """
        # compute how much the mouse has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        # move the object the appropriate amount
        self.graph.move(self.nodes[self._drag_data["item"]], delta_x, delta_y)
        X[self._drag_data["item"]] += delta_x
        Y[self._drag_data["item"]] += delta_y
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def clear_hijack(self, node):
        self.graph.itemconfigure(self.nodes[node], fill="#93a1a1")

    def fail_hijack(self, node):
        self.graph.itemconfigure(self.nodes[node], fill="blue")

    def succeed_hijack(self, node):
        self.graph.itemconfigure(self.nodes[node], fill="#ff9400")

    def done_hijack(self, node):
        self.graph.itemconfigure(self.nodes[node], fill="#cc2222")

    def forward(self):
        if self.curT < len(l):
            if self.curT > 0:
                i, r = l[self.curT-1]
                if r:
                    self.done_hijack(i)
                else:
                    self.clear_hijack(i)
            self.last = 1
            i, r = l[self.curT]
            self.curT += 1
            if r:
                self.succeed_hijack(i)
            else:
                self.fail_hijack(i)

    def backward(self):
        if self.curT > 0:
            self.curT -= 1
            i, r = l[self.curT]
            self.clear_hijack(i)
            if self.last == 1:
                self.last = -1
                self.backward()
            else:
                self.forward()


root = tk.Tk()
gg = GraphGUI(root)
gg.mainloop()
