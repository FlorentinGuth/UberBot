# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:46:02 2017
@author: liozou
"""

#!/usr/bin/env python

import botnet
import network
import tests
import thompson_sampling as thom
import markov
import qlearning
import strategy

import tkinter as tk
import math
import random as rd

def select(l):
    return l[rd.randint(0,len(l)-1)]

def wait():
    for i in range(100000):
        pass

class MainGUI(tk.Tk):

    def __init__(self, n, sims, master=None):
        tk.Tk.__init__(self,master)
        self.master = master
        self.n = n # number of devices in the network
        self.nbs = len(sims)
        self.initialize()
        self.sims = sims
        self.last = max(map(len, self.sims))

    def initialize(self):
        self.grid()

        self.update = [None] * self.nbs

        self.currPos = 0

        m = max(math.floor(math.sqrt(self.n)),3)

        index = []
        for k in range(self.n):
            index.append((m+k//m,k%m))
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
        self.l[n].itemconfigure(str(i+2), fill="#ff9400")
        self.l[n].update()

    def resisted(self,n,i):
        self.l[n].itemconfigure(str(i+2), fill="blue")
        self.l[n].update()

    def unharmed(self,n,i):
        self.l[n].itemconfigure(str(i+2), fill="#93a1a1")
        self.l[n].update()

    def compromized(self,n,i):
        self.l[n].itemconfigure(str(i+2), fill="#cc2222")
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
        if self.currPos == self.last:
            self.reboot()
        for i in range(self.currPos,self.last):
            stop = False
            for sim in self.sims:
                if i == len(sim):
                    stop = True
                    break
            if stop:
                break
            self.act(i)
        self.currPos = self.last

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


class GUI(botnet.Botnet):

    def __init__(self,initial_power):
        net = network.Network(initial_power)
        botnet.Botnet.__init__(self, net)
        self.GUI = None

    def display(self,mode):
        def exe():
            mode(10,self)

        self.GUI = MainGUI(self.network.size,exe)
        self.GUI.mainloop()

    def take_action(self,action):
        if self.GUI != None:
            self.GUI.attacked(action, 0)
            wait()
            success = botnet.Botnet.take_action(self,action)
            wait()
            if success:
                self.GUI.compromized(action, 0)
            else:
                self.GUI.unharmed(action, 0)
            wait()
            return success
        else:
            return botnet.Botnet.take_action(self,action)


#n = GUI(1)
#
#k = 16
#for i in range(k):
#    n.network.add(i**1.8+1, i+1, i)
#q = thom.Thompson(n.network, 0.9)
#
#
#n.display(tests.unknown_thompson,q)

k = 49
n = network.Network(1)
for i in range(k):
    n.add(i**2, i+1, i)
n.set_complete_network()
gamma = 0.9
q1 = thom.Thompson(n, gamma, 0.1, strat=strategy.thompson_standard)
q2 = thom.Thompson(n, gamma, 0.1, strat=strategy.thompson_standard)
qs = markov.Qstar(n, gamma)
ql = qlearning.Qlearning(n, gamma, 0.1, strat=strategy.full_random)

nbt = 500
s = [tests.get_last_invasion(10, q1), tests.get_last_invasion(nbt, q2), tests.get_last_invasion(nbt, ql)]
n = MainGUI(k, s)
n.mainloop()

