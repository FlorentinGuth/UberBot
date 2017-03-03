import copy
class State:
    """
    The State class is actually a set of integers, represented by an integer.
    WARNING: add is now mutable, use State.added(state, node) instead
    """

    def __init__(self, size, nodes=[]):
        self.content = 0
        self.size = size    # Keeps track of the size of the network, useful

        for node in nodes:
            self.add(node)

    def __contains__(self, item):
        return (self.content >> item) & 1

    def is_empty(self):
        return self.content == 0

    def is_full(self):
        return self.to_list() == list(range(self.size))

    @staticmethod
    def full_state(size):
        return State(size, list(range(size)))

    def to_list(self):
        l = []
        for node in range(self.size):
            if node in self:
                l.append(node)
        return l

    def cardinality(self):
        return len(self.to_list())

    def add(self, node):
        if self not in node:
            self.content += 1 << node

    @staticmethod
    def added(state, node):
        new_state = copy.copy(state)
        new_state.add(node)