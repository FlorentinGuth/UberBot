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

import tkinter as tk
import math
import random as rd

def select(l):
    return l[rd.randint(0,len(l)-1)]

def wait():
    for i in range(100000):
        pass

class MainGUI(tk.Tk):
    
    def __init__(self, res, master=None):
        tk.Tk.__init__(self,master)
        self.master = master
        self.n = res[0] # number of devices in the network
        self.initialize()
        self.actions = res[1]
        self.last = len(self.actions)
        
    def initialize(self):
        self.grid()
        
        self.update = None
        
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
            c.create_rectangle(0,0,width,height,tags=str(count),fill='green')
            c.create_text(width//2,height-10,text=str(count))
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
        
    
    def attacked(self,n):
        self.l[n].itemconfigure("1", fill="orange")
        self.l[n].update()
    
    def resisted(self,n):
        self.l[n].itemconfigure("1", fill="blue")
        self.l[n].update()
    
    def unharmed(self,n):
        self.l[n].itemconfigure("1", fill="green")
        self.l[n].update()
    
    def compromized(self,n):
        self.l[n].itemconfigure("1", fill="red")
        self.l[n].update()
    
    def act(self,action):
        if self.update != None:
            if self.update[1]:
                self.compromized(self.update[0])
            else:
                self.unharmed(self.update[0])
        wait()
        if action[1]:
            self.attacked(action[0])
        else:
            self.resisted(action[0])
        self.update = action
        wait()
    
    def reboot(self):
        self.update = None
        for i in range(self.n):
            self.l[i].itemconfigure("1", fill="black")
        for i in range(self.n):
            self.unharmed(i)
        self.currPos = 0
    
    def launch(self):
        if self.currPos == self.last:
            self.reboot()
        for i in range(self.currPos,self.last):
            self.act(self.actions[i])
        self.currPos = self.last
    
    def forward(self):
        if self.currPos == self.last:
            self.reboot()
            self.currPos = 0
            return
        self.act(self.actions[self.currPos])
        self.currPos += 1

    def backward(self):
        self.update = None
        if self.currPos == 0:
            for i in range(self.n):
                self.l[i].itemconfigure("1", fill="red")
            self.currPos = self.last
        action = self.actions[self.currPos-1]
        self.attacked(action[0])
        wait()
        self.unharmed(action[0])
        wait()
        self.currPos -=1
        

    def animate(self):
        available = [i for i in range(self.n)]
        while available!=[]:
            x = select(available)
            self.attacked(x)
            wait()
            att = rd.random()
            if att > 0.7:
                self.compromized(x)
                available.remove(x)
            else:
                self.unharmed(x)


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
            self.GUI.attacked(action)
            wait()
            success = botnet.Botnet.take_action(self,action)
            wait()
            if success:
                self.GUI.compromized(action)
            else:
                self.GUI.unharmed(action)
            wait()
            return success
        else:
            return botnet.Botnet.take_action(self,action)


#n = GUI(1)
#
#k = 16
#for i in range(k):
#    n.network.add(i**1.8+1, i+1, i)
#q = thom.Thomson(n.network, 0.9)
#
#
#n.display(tests.unknown_thomson,q)

k = 49
n = network.Network(1)
for i in range(k):
    n.add(i**1.5, i+1, i)

q = thom.Thomson(n, 0.9, 0.1)

n = MainGUI(tests.liozoub(500, q))
n.mainloop()

