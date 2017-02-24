# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:19:29 2017

@author: liozou
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:46:02 2017

@author: liozou
"""

#!/usr/bin/env python

import debuts
import time

import tkinter as tk
import math
import random as rd

def select(l):
    return l[rd.randint(0,len(l)-1)]

def wait():
    for i in range(100000):
        pass

class MainGUI(tk.Tk):
    
    def __init__(self, n, master=None):
        tk.Tk.__init__(self,master)
        self.master = master
        self.n = n # number of devices in the network
        self.initialize()
        
    def initialize(self):
        self.grid()
        
        m = math.floor(math.sqrt(self.n))
        
        index = []
        for k in range(self.n):
            index.append((m+k//m,k%m))
        self.index=index
        
        refSize = 200/m
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
        
        self.button = tk.Button(self, text="Animate", command=self.onButtonClick)
        self.button.grid(column=1,row=index[-1][0]+1, columnspan=m)
        
        
#        self.entry = tk.Entry(self)
#        self.entry.grid(column=0,row=0,sticky = 'NS')
#        
#        
#        self.label = tk.Label(self, anchor = 'w', fg="white", bg = "blue")
#        self.label.grid(column=1, row=1, columnspan=2, sticky='WE')
#        
#        self.grid_columnconfigure(1,weight=1)
#        self.grid_rowconfigure(0,weight=1)
    
    def onButtonClick(self):
        self.animate()
        
    def attacked(self,n):
        self.l[n].itemconfigure("1", fill="blue")
        self.l[n].update()
    
    def resisted(self,n):
        self.l[n].itemconfigure("1", fill="green")
        self.l[n].update()
    
    def compromized(self,n):
        self.l[n].itemconfigure("1", fill="red")
        self.l[n].update()

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
                self.resisted(x)

if __name__ == "__main__":
    app = MainGUI(256)
    app.title("TestMain")
    app.mainloop()
    
    del app

