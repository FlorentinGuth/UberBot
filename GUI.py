# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:46:02 2017
@author: liozou
"""

#!/usr/bin/env python

import network
import thompson_sampling as thom
import markov
import qlearning
import strategy
import tests

import tkinter as tk
import math
import random as rd
import time

def select(l):
    return l[rd.randint(0,len(l)-1)]

def wait():
    for i in range(100000):
        pass

policies = [("Thompson", thom.Thompson, strategy.thompson_standard),
            ("Qstar", markov.Qstar, None),
            ("Qlearning", qlearning.Qlearning, strategy.full_random)]

class MainGUI(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        
        self.nbP = 3
        self.nbN = 49

        self.running = False

        self.grid()
        index = []
        self.l = []

        self.width = 200
        self.height = 20

        index.append((0, 0))
        c = tk.Canvas(self, width=self.width, height=self.height)
        c.create_rectangle(0, 0, self.width, self.height, tags=str(len(index)))
        c.create_text(self.width//2,self.height//2,text=str(self.nbP) + " policies")
        c.grid(row=index[-1][0], column=index[-1][1],columnspan=2)

        self.c = c

        index.append((1, 0))
        self.p_dec = tk.Button(self, text="Decrease", command=self.decreaseP)
        self.p_dec.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)
        
        index.append((1, 1))
        self.p_inc = tk.Button(self, text="Increase", command=self.increaseP)
        self.p_inc.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)

        index.append((2, 0))
        d = tk.Canvas(self, width=self.width, height=self.height)
        d.create_rectangle(0, 0, self.width, self.height, tags=str(len(index)))
        d.create_text(self.width//2,self.height//2,text=str(self.nbN) + " nodes")
        d.grid(row=index[-1][0], column=index[-1][1],columnspan=2)

        self.d = d

        index.append((3, 0))
        self.p_dec = tk.Button(self, text="Decrease", command=self.decreaseN)
        self.p_dec.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)
        
        index.append((3, 1))
        self.p_inc = tk.Button(self, text="Increase", command=self.increaseN)
        self.p_inc.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)

        index.append((4, 0))
        self.p_dec = tk.Button(self, text="-10", command=self.decreaseN10)
        self.p_dec.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)
        
        index.append((4, 1))
        self.p_inc = tk.Button(self, text="+10", command=self.increaseN10)
        self.p_inc.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)

        tk.Label(self, text="Gamma").grid(row=5, sticky=tk.W)

        self.e1 = tk.Entry(self, width=15)
        self.e1.insert(tk.END, "0.9")

        self.e1.grid(row=5, column=0, columnspan=2, sticky=tk.E)

        for _ in range(self.nbP):
            self.add_policy()
        
        index.append((1000, 0))
        self.start = tk.Button(self, text="Start", command=self.startf)
        self.start.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W, columnspan=2)

        self.index = index

        #self.grid()
        #index = []

        #width = 100
        #height = 20

        #for (a, b, c) in policies:
        #    index.append((0, len(index)))
        #    c = tk.Canvas(self, width=width, height=height)
        #    c.create_rectangle(0, 0, width, height, tags=str(len(index)))
        #    c.create_text(width//2,height//2,text=a)
        #    c.grid(column=0, row=len(index)-1)
        #
        #self.index = index

    def add_policy(self):
        c = tk.Canvas(self, width=self.width, height=self.height)
        c.create_rectangle(0, 0, self.width, self.height, tags=str(len(self.l)))
        c.create_text(self.width//2, self.height//2, text="Policy n°"+str(len(self.l)+1))
        c.grid(row=6+3*len(self.l), column=0, columnspan=2)
        v = tk.StringVar(self)
        v.set(policies[0][0])
        l = tk.OptionMenu(self, v, *[a for a,_,_ in policies])
        l.grid(row=7+3*len(self.l), column=0, columnspan=2,sticky=tk.N+tk.S+tk.E+tk.W)

        el = tk.Label(self, text="Train #")
        el.grid(row=8+3*len(self.l), sticky=tk.W)

        e = tk.Entry(self, width=15)
        e.insert(tk.END, "100")

        e.grid(row=8+3*len(self.l), column=0, columnspan=2, sticky=tk.E)

        self.l.append((c, l, el, v, e))
        self.winfo_toplevel().wm_geometry("")

    def remove_policy(self):
        c, l, el, _, e = self.l.pop()
        c.grid_forget()
        l.grid_forget()
        el.grid_forget()
        e.grid_forget()

    def decreaseP(self):
        if self.nbP > 1:
            self.nbP -= 1
            self.remove_policy()
            self.c.itemconfigure(str(2), text=str(self.nbP) + " policies")
            if self.nbP == 1:
                self.c.itemconfigure(str(2), text=str(self.nbP) + " policy")
            self.c.update()

    def increaseP(self):
        self.nbP += 1
        self.add_policy()
        self.c.itemconfigure(str(2), text=str(self.nbP) + " policies")
        self.c.update()

    def decreaseN(self):
        if self.nbN > 1:
            self.nbN -= 1
            self.d.itemconfigure(str(2), text=str(self.nbN) + " nodes")
            if self.nbN == 1:
                self.d.itemconfigure(str(2), text=str(self.nbN) + " node")
            self.d.update()

    def increaseN(self):
        self.nbN += 1
        self.d.itemconfigure(str(2), text=str(self.nbN) + " nodes")
        self.d.update()

    def decreaseN10(self):
        self.nbN = max(self.nbN - 10, 1)
        self.d.itemconfigure(str(2), text=str(self.nbN) + " nodes")
        if self.nbN == 1:
            self.d.itemconfigure(str(2), text=str(self.nbN) + " node")
        self.d.update()

    def increaseN10(self):
        self.nbN += 10
        self.d.itemconfigure(str(2), text=str(self.nbN) + " nodes")
        self.d.update()

    def startf(self):
        if self.running == False:
            self.running = True
            net = network.Network(1)
            for i in range(self.nbN):
                net.add(i**2, i+1, i)
            net.set_complete_network()

            s = []
            gamma = float(self.e1.get())

            for i in range(self.nbP):
                q = None
                pb, pc = None, None
                for a, b, c in policies:
                    if a == self.l[i][3].get():
                        pb, pc = b, c
                        break
                if pc == None:
                    q = pb(net, gamma)
                else:
                    q = pb(net, gamma, 0.1, strat=pc)
                s.append((self.l[i][3].get(), tests.get_last_invasion(int(self.l[i][4].get()), q)))

            n = GUI(self.nbN, s)
            n.mainloop()
            self.running = False


class GUI(tk.Tk):

    def __init__(self, n, sims, master=None):
        tk.Tk.__init__(self,master)
        self.master = master
        self.n = n # number of devices in the network
        self.nbs = len(sims)
        self.names = [a for a, _ in sims]
        self.initialize()
        self.sims = [b for _, b in sims]
        self.last = max(map(len, self.sims))

    def initialize(self):
        self.grid()

        self.update = [None] * self.nbs

        self.currPos = 0
        self.running = False

        m = max(math.floor(math.sqrt(self.n)),3)

        index = []
        for k in range(self.n):
            index.append((k//m,k%m))
        self.index=index

        refSize = 200//m
        if m<6:
            refSize -= refSize//4
        if m<3:
            refSize -= refSize//4
        width = 5*refSize
        height = 3*refSize

        l = []
        count = 0
        for (i,j) in index:
            c = tk.Canvas(self,width=width,height=height)
            c.create_rectangle(0,0,width,20,tags=str(count),fill='#6c71c4')
            for k in range(self.nbs):
                c.create_rectangle(k*width/self.nbs,20,(k+1)*width/self.nbs,height,tags=str(count),fill='#93a1a1')
                c.create_text(k*width/self.nbs+width/2/self.nbs,10+height/2,width=width/self.nbs,text=str(self.names[k]))
            c.create_text(width//2,10,text=str(count))
            l.append(c)
            c.grid(column=j, row=i)
            count+=1

        self.l = l

        button_pos = max(m//7,1)

        self.b_backward = tk.Button(self, text="Previous", command=self.backward)
        self.b_backward.grid(column=0,row=index[-1][0]+1, columnspan=button_pos)

        self.b_launch = tk.Button(self, text="Launch", command=self.launch)
        self.b_launch.grid(column=button_pos,row=index[-1][0]+1, columnspan = m-2*button_pos)

        self.b_forward = tk.Button(self, text="Next", command=self.forward)
        self.b_forward.grid(column=m-button_pos,row=index[-1][0]+1, columnspan=button_pos)

        self.bind('<Right>',lambda x:self.forward())
        self.bind('<Return>',lambda x:self.forward())
        self.bind('<Left>',lambda x:self.backward())
        self.bind('<BackSpace>',lambda x:self.backward())
        self.bind('<space>',lambda x:self.launch())


    def attacked(self,n,i):
        self.l[n].itemconfigure(str(2*i+2), fill="#ff9400")
        self.l[n].update()

    def resisted(self,n,i):
        self.l[n].itemconfigure(str(2*i+2), fill="blue")
        self.l[n].update()

    def unharmed(self,n,i):
        self.l[n].itemconfigure(str(2*i+2), fill="#93a1a1")
        self.l[n].update()

    def compromized(self,n,i):
        self.l[n].itemconfigure(str(2*i+2), fill="#cc2222")
        self.l[n].update()

    def act(self,iAct):
        for i, sim in enumerate(self.sims):
            if iAct < len(sim):
                action = sim[iAct]
                if self.update[i] is not None:
                    if self.update[i][1]:
                        self.compromized(self.update[i][0], i)
                    else:
                        self.unharmed(self.update[i][0], i)
                if action[1]:
                    self.attacked(action[0], i)
                else:
                    self.resisted(action[0], i)
                self.update[i] = action
        wait()

    def reboot(self):
        self.update = [None] * self.nbs
        #for i in range(self.n):
        #    self.l[i].itemconfigure("1", fill="black")
        for i in range(self.n):
            for j in range(self.nbs):
                self.unharmed(i, j)
        self.currPos = 0

    def launch(self):
        if not self.running:
            self.running = True
            if self.currPos == self.last:
                self.reboot()
            cur = 0
            sims = [sim for sim in self.sims]
            sims.sort(key=len)
            while self.currPos >= len(sims[cur]):
                cur += 1
            for i in range(self.currPos,self.last):
                if i == len(sims[cur]):
                    break
                self.act(i)
                time.sleep(.1)
                self.currPos += 1
            self.running = False

    def forward(self):
        if self.currPos == self.last:
            self.reboot()
            self.currPos = 0
            return
        self.act(self.currPos)
        self.currPos += 1

    def backward(self):
        self.update = [None] * self.nbs
        if self.currPos == 0:
            for i in range(self.n):
                for j in range(self.nbs):
                    self.l[i].itemconfigure(str(j+2), fill="#cc2222")
            self.currPos = self.last
        self.currPos -=1
        for i in range(self.nbs):
            if self.currPos < len(self.sims[i]):
                action = self.sims[i][self.currPos]
                self.attacked(action[0], i)
                self.unharmed(action[0], i)
        wait()


    #def animate(self):
    #    available = [i for i in range(self.n)]
    #    while available!=[]:
    #        x = select(available)
    #        self.attacked(x, 0)
    #        wait()
    #        att = rd.random()
    #        if att > 0.7:
    #            self.compromized(x, 0)
    #            available.remove(x)
    #        else:
    #            self.unharmed(x, 0)


# class GUI(botnet.Botnet):
# 
#     def __init__(self,initial_power):
#         net = network.Network(initial_power)
#         botnet.Botnet.__init__(self, net)
#         self.GUI = None
# 
#     def display(self,mode):
#         def exe():
#             mode(10,self)
# 
#         self.GUI = MainGUI(self.network.size,exe)
#         self.GUI.mainloop()
# 
#     def take_action(self,action):
#         if self.GUI != None:
#             self.GUI.attacked(action, 0)
#             wait()
#             success = botnet.Botnet.take_action(self,action)
#             wait()
#             if success:
#                 self.GUI.compromized(action, 0)
#             else:
#                 self.GUI.unharmed(action, 0)
#             wait()
#             return success
#         else:
#             return botnet.Botnet.take_action(self,action)


#n = GUI(1)
#
#k = 16
#for i in range(k):
#    n.network.add(i**1.8+1, i+1, i)
#q = thom.Thompson(n.network, 0.9)
#
#
#n.display(tests.unknown_thompson,q)

#k = 12
#n = network.Network(1)
#for i in range(k):
#    n.add(i**2, i+1, i)
#n.set_complete_network()
#gamma = 0.9
#q1 = thom.Thompson(n, gamma, 0.1, strat=strategy.thompson_standard)
#q2 = thom.Thompson(n, gamma, 0.1, strat=strategy.thompson_standard)
#qs = markov.Qstar(n, gamma)
#ql = qlearning.Qlearning(n, gamma, 0.1, strat=strategy.full_random)
#
#nbt = 500
#s = [tests.get_last_invasion(1, qs), tests.get_last_invasion(nbt, q2), tests.get_last_invasion(nbt, ql)]
#n = MainGUI(k, s)
#n.mainloop()


root = tk.Tk()
root.columnconfigure(0, weight=1)
n = MainGUI(root)
n.mainloop()
