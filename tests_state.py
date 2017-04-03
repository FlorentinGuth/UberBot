from state import State


d = {}

def fill():
    s = State(5)
    for i in range(5):
        d[s] = i
        s = s.add(i)

def show():
    t = State(5)
    for i in range(5):
        print(d[t])
        t = t.add(i)

fill()
show()