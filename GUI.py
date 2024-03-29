# -*- coding: utf-8 -*-

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

policies = [("Thompson", thom.Thompson, strategy.thompson_standard),
            ("QStar", markov.QStar, None),
            ("QLearning", qlearning.QLearning, strategy.full_random)]

class MainGUI(tk.Frame):

    def __init__(self, master=None):
        tk.Frame.__init__(self, master)
        self.master = master
        
        self.nbP = 3
        self.comp = True

        self.running = False

        self.grid()
        index = []
        self.l = []

        self.width = 200
        self.height = 20

        tk.Label(self, text="# nodes").grid(row=0, sticky=tk.W)

        self.enodes = tk.Entry(self, width=15)
        self.enodes.insert(tk.END, "49")

        self.enodes.grid(row=0, column=0, columnspan=2, sticky=tk.E)

        tk.Label(self, text="Difficulty").grid(row=1, sticky=tk.W)

        self.ediff = tk.Entry(self, width=15)
        self.ediff.insert(tk.END, "1.5")

        self.ediff.grid(row=1, column=0, columnspan=2, sticky=tk.E)

        tk.Label(self, text="Big nodes %").grid(row=2, sticky=tk.W)

        self.bignodes = tk.Scale(self, from_=0, to=100, orient=tk.HORIZONTAL)
        self.bignodes.grid(row=2, column=0, columnspan=2, sticky=tk.E)
        self.bignodes.set(20)

        self.complete = tk.Button(self, text="Complete", command=self.toggleComplete)
        self.complete.grid(row=3, column=0, columnspan=2, sticky=tk.E+tk.N+tk.S+tk.W)

        index.append((4, 0))
        c = tk.Canvas(self, width=self.width, height=self.height)
        c.create_rectangle(0, 0, self.width, self.height, tags=str(len(index)))
        c.create_text(self.width//2,self.height//2,text=str(self.nbP) + " policies")
        c.grid(row=index[-1][0], column=index[-1][1],columnspan=2)

        self.c = c

        index.append((5, 0))
        self.p_dec = tk.Button(self, text="Decrease", command=self.decreaseP)
        self.p_dec.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)
        
        index.append((5, 1))
        self.p_inc = tk.Button(self, text="Increase", command=self.increaseP)
        self.p_inc.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W)

        tk.Label(self, text="Gamma").grid(row=6, sticky=tk.W)

        self.e1 = tk.Entry(self, width=15)
        self.e1.insert(tk.END, "0.9")

        self.e1.grid(row=6, column=0, columnspan=2, sticky=tk.E)

        for _ in range(self.nbP):
            self.add_policy()
        
        index.append((1000, 0))
        self.start = tk.Button(self, text="Start", command=self.startf)
        self.start.grid(row=index[-1][0],column=index[-1][1],sticky=tk.N+tk.S+tk.E+tk.W, columnspan=2)

        self.index = index

    def toggleComplete(self):
        if self.comp:
            self.complete["text"] = "Not complete"
        else:
            self.complete["text"] = "Complete"
        self.comp = not self.comp

    def add_policy(self):
        c = tk.Canvas(self, width=self.width, height=self.height)
        c.create_rectangle(0, 0, self.width, self.height, tags=str(len(self.l)))
        c.create_text(self.width//2, self.height//2, text="Policy n°"+str(len(self.l)+1))
        c.grid(row=7+3*len(self.l), column=0, columnspan=2)
        v = tk.StringVar(self)
        v.set(policies[0][0])
        l = tk.OptionMenu(self, v, *[a for a,_,_ in policies])
        l.grid(row=8+3*len(self.l), column=0, columnspan=2,sticky=tk.N+tk.S+tk.E+tk.W)

        el = tk.Label(self, text="Train #")
        el.grid(row=9+3*len(self.l), sticky=tk.W)

        e = tk.Entry(self, width=15)
        e.insert(tk.END, "100")

        e.grid(row=9+3*len(self.l), column=0, columnspan=2, sticky=tk.E)

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

    def startf(self):
        if not self.running:
            self.running = True
            net = network.random_network(int(self.enodes.get()), float(self.ediff.get()), self.bignodes.get()/100, self.comp)

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
                    q = pb(net, gamma, 0.01, strat=pc)
                s.append((self.l[i][3].get(), tests.get_last_invasion(int(self.l[i][4].get()), q)))

            n = GUI(int(self.enodes.get()), s)
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
        self.stop = False
        self.reset = False

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
        self.b_launch.grid(column=button_pos-1,row=index[-1][0]+1, columnspan = m-2*button_pos)

        self.b_reset = tk.Button(self, text="Reset", command=self.reboot)
        self.b_reset.grid(column=button_pos+1,row=index[-1][0]+1, columnspan = m-2*button_pos)

        self.b_forward = tk.Button(self, text="Next", command=self.forward)
        self.b_forward.grid(column=m-button_pos,row=index[-1][0]+1, columnspan=button_pos)

        self.bind('<Right>',lambda x:self.forward())
        self.bind('<Return>',lambda x:self.forward())
        self.bind('<Left>',lambda x:self.backward())
        self.bind('<BackSpace>',lambda x:self.backward())
        self.bind('<space>',lambda x:self.launch())
        self.bind('<Escape>',lambda x:self.reboot())


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

    def reboot(self):
        if self.running:
            self.reset = True
            self.stop = True
        else:
            self.update = [None] * self.nbs
            for i in range(self.n):
                for j in range(self.nbs):
                    self.unharmed(i, j)
            self.currPos = 0
            self.reset = False

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
                if self.stop or i == len(sims[cur]):
                    break
                self.act(i)
                time.sleep(.1)
                self.currPos += 1
            self.stop = False
            self.running = False
            if self.reset:
                self.reboot()
        else:
            self.stop = True

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


root = tk.Tk()
root.columnconfigure(0, weight=1)
n = MainGUI(root)
n.mainloop()
