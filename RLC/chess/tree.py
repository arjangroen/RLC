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

    def get_value(self, sampler_fn):
        return sampler_fn(self.values)

    @staticmethod
    def _thompson_sampler(values):
        return np.random.choice(values)

    @staticmethod
    def _greedy_sampler(values):
        return np.max(values)

    def select(self, sampler_fn):
        """
        Select a child node according to the sampler function
        :param sampler_fn: gives a value for a node
        :return: best_move: key of selected child node
        """
        if not self.children:
            return None
        else:
            best_move = None
            best_value = sampler_fn(self.values)
            for move, child in self.children.items():
                sampler_fn[move]
                value = sampler_fn(child.values)
                if value > best_value:
                    best_move = move
                    best_value = value
            return best_move

    def get_up(self):
        return self.parent

    def get_down(self, move):
        return self.children[move]

    def clean(self, maxsize=1e9):
        worst_move = None
        worst_score = 1e3
        size = sys.getsizeof(self)
        if size > maxsize:
            for move, child in self.children.items():
                if np.max(child.values) < worst_score:
                    worst_score = np.max(child.values)
                    worst_move = move
            print("cleaning worst node", worst_move)
            print("prior size", sys.getsizeof(self))
            del self.children[worst_move]
            gc.collect()
            print("new size", sys.getsizeof(self))
