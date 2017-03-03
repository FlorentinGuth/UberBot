import copy


class State:
    """
    The State class is actually a set of integers, represented by an integer.
    WARNING: add is now mutable, use State.added(state, node) instead
    """

    def __init__(self, size, nodes=None):
        self.content = 0
        self.size = size    # Keeps track of the size of the network, useful
        self.nb_hikacked = 0

        if nodes is not None:
            for node in nodes:
                self.add(node)

    def __contains__(self, item):
        return (self.content >> item) & 1

    def is_empty(self):
        return self.content == 0

    def is_full(self):
        return ((self.content + 1) >> self.size) & 1

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
        return self.nb_hikacked

    def nb_remaining(self):
        # TODO Ca va pas plaire a Bodohautin
        return self.size - self.cardinality()

    def add(self, node):
        if not (0 <= node < self.size):
            raise ValueError("Node %d out of bounds" % node)

        if node not in self:
            self.content += 1 << node
            self.nb_hikacked += 1

    @staticmethod
    def added(state, node):
        new_state = copy.copy(state)
        new_state.add(node)
        return new_state

    def copy(self):
        return copy.copy(self)
