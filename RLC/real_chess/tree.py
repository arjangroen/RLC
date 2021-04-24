import numpy as np
import sys
import gc
import torch


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.8):
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

    def get_ucb(self, move, c, color=1, q=None):
        """
        Calculate the UCB
        :return: UCB
        """
        Na = 1
        if move in self.children.keys():
            q = np.mean(np.array(self.children[move].values) * color)
            Na = len(self.children[move].values)
        confidence_interval = c * np.sqrt(np.log(len(self.values)) / Na)
        return q + confidence_interval

    def select(self, color=1, legal_moves=None, q_values=None):
        max_value = np.NINF
        max_move = None
        for move in legal_moves:
            if move in self.children.keys():
                child_value = self.get_ucb(move, 1, color)
            else:
                child_value = self.get_ucb(move, 1,  color=color, q=q_values[0, move.from_square, move.to_square].detach().numpy())
            if child_value > max_value:
                max_move = move
                max_value = child_value
        if max_move not in self.children.keys():
            self.add_child(max_move)
        return self.children[max_move], max_move

    def add_child(self, move):
        self.children[move] = Node(parent=self)

    def get_up(self):
        return self.parent

    def get_down(self, move):
        return self.children[move]

    def simulate(self, fixed_agent, env, depth=0, max_depth=10, eps=.25):
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

        explore = np.random.uniform(0, 1) < eps

        if explore:
            legal_moves = [x for x in env.board.generate_legal_moves()]
            move = np.random.choice(legal_moves)
            move_proba = torch.Tensor([1 / len(legal_moves)]).float()
        else:
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
                )[1].min()  # Assuming black chooses lowest Q value
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
