import numpy as np
import sys
import gc
import torch


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.9):
        """
        Game Node for Monte Carlo Tree Search
        Args:
            board: the chess board
            parent: the parent node
            gamma: the discount factor
        """
        self.children = {}  # Child nodes
        self.board = board  # Chess board
        self.parent = parent
        self.values = []  # reward + Returns
        self.gamma = gamma
        self.starting_value = 0

    def clean(self, maxsize=1e9):
        worst_move = None
        worst_score = 1e3
        if sys.getsizeof(self) > maxsize:
            for move, child in self.children.items():
                if np.max(child.values) < worst_score:
                    worst_score = np.max(child.values)
                    worst_move = move
            print("cleaning worst node", worst_move)
            print("prior size", sys.getsizeof(self))
            del self.children[worst_move]
            gc.collect()
            print("new size", sys.getsizeof(self))

    def get_root(self):
        root = self
        while root.parent:
            root = root.parent
        return root

    def update_child(self, move, Returns):
        """
        Update a child with a simulation result
        Args:
            move: The move that leads to the child
            Returns: the reward of the move and subsequent returns

        Returns:

        """
        child = self.children[move]
        child.values.append(Returns)

    def update(self, Returns):
        """
        Update a node with observed Returns
        Args:
            Returns: Future returns

        Returns:

        """
        self.values.append(Returns)

    def select(self, color=1):
        """
        Use Thompson sampling to select the best child node
        Args:
            color: Whether to select for white or black

        Returns:
            (node, move)
            node: the selected node
            move: the selected move
        """
        assert color == 1 or color == -1, "color has to be white (1) or black (-1)"
        if self.children:
            max_sample = np.random.choice(self.values) * color
            max_move = None
            for move, child in self.children.items():
                child_sample = np.random.choice(child.values) * color
                if child_sample > max_sample:
                    max_sample = child_sample
                    max_move = move
            if max_move:
                return self.children[max_move], max_move
            else:
                return self, None
        else:
            return self, None

    def add_child(self, move):
        self.children[move] = Node(parent=self)

    def get_up(self):
        return self.parent

    def get_down(self, move):
        return self.children[move]

    def simulate(self, fixed_agent, env, depth=0, max_depth=5):
        """
        Recursive Monte Carlo Playout
        Args:
            model: The model used for bootstrap estimation
            env: the chess environment
            depth: The recursion depth
            max_depth: How deep to search
            temperature: softmax temperature

        Returns:
            Playout result.
        """

        move, move_proba = fixed_agent.select_action(env)
        episode_end, reward = env.step(move)

        if episode_end:
            Returns = reward
        elif depth >= max_depth:  # Bootstrap the Monte Carlo Playout
            if env.board.turn:
                Returns = reward + self.gamma * fixed_agent(
                    torch.from_numpy(np.expand_dims(env.layer_board, axis=0)).float()
                )[1].max()
            else:
                Returns = reward + self.gamma * fixed_agent(
                    torch.from_numpy(np.expand_dims(env.layer_board, axis=0)).float()
                )[1].min()
            Returns = Returns.detach().numpy()
        else:  # Recursively continue
            Returns = reward + self.gamma * self.simulate(fixed_agent, env, depth=depth + 1)

        env.reverse()

        if depth == 0:
            gc.collect()
            return Returns, move
        else:
            noise = np.random.randn() / 1e6
            return Returns + noise
