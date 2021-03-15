import numpy as np
import sys
import gc


class Tree(object):

    def __init__(self, parent=None):
        """
        Monte Carlo Tree
        :param parent: Monte Carlo Tree parent Node
        """
        self.parent = parent
        self.children = {}
        self.values = []
        self.checkpoint = False

    def add_child(self, move):
        self.children[move] = Tree(parent=self)

    def get_root(self):
        """
        Get the root node
        :return: the Tree root
        """
        root = self
        while root.parent:
            root = root.parent
        return root

    def get_checkpoint(self):
        node = self
        while not node.checkpoint:
            node = node.parent
        return node

    def get_value(self, sampler_fn):
        return sampler_fn(self.values)

    def update(self, Returns=None):
        """
        Update a node with observed Returns
        Args:
            Returns: Future returns

        Returns:

        """
        if Returns:
            self.values.append(Returns)

    def get_up(self):
        return self.parent

    def get_down(self, move):
        return self.children[move]

    def clean(self, maxsize=1e9, color = 1):
        worst_move = None
        worst_score = 1e3
        size = sys.getsizeof(self)
        if size > maxsize:
            for move, child in self.children.items():
                values = color * np.array(child.values)
                if np.max(values) < worst_score:
                    worst_score = np.max(values)
                    worst_move = move
            print("cleaning worst node", worst_move)
            print("prior size", sys.getsizeof(self))
            del self.children[worst_move]
            gc.collect()
            print("new size", sys.getsizeof(self))
