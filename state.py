import copy


class State:
    """
    The State class is actually a set of integers, represented by an integer.
    This class is immutable, so it can be used as dictionary keys.
    """

    def __init__(self, size, nodes=None):
        self.content = 0
        self.size = size    # Keeps track of the size of the network, useful
        self.nb_hijacked = 0

        if nodes is not None:
            for node in nodes:
                self.content += 1 << node
                self.nb_hijacked += 1

    def __contains__(self, item):
        return (self.content >> item) & 1

    def __len__(self):
        return self.cardinality()

    def __eq__(self, other):
        return self.content == other.content    # Should be sufficient

    def __hash__(self):
        return self.content

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
        return self.nb_hijacked

    def nb_remaining(self):
        return self.size - self.cardinality()

    def add(self, node):
        """
        Returns a new state with the node added. This has no effect if the node was already present.
        :param node: 
        :return: 
        """
        new_state = self.copy()

        if not (0 <= node < new_state.size):
            raise ValueError("Node %d out of bounds" % node)

        if node not in new_state:
            new_state.content += 1 << node
            new_state.nb_hijacked += 1

        return new_state

    def remove(self, node):
        """
        Returns a new state, with the node removed. This has no effect if the node was already not present.
        :param node: 
        :return: 
        """
        new_state = self.copy()

        if not (0 <= node < new_state.size):
            raise ValueError("Node %d out of bounds" % node)

        if node in new_state:
            new_state.content -= 1 << node
            new_state.nb_hijacked -= 1

        return new_state

    def copy(self):
        return copy.copy(self)
