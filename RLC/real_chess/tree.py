import numpy as np
import sys
import gc
import torch
from RLC.real_chess.hyperparams import GAMMA


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


class Node(object):

    def __init__(self, board=None, parent=None):
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

    def ucb1(self, Qsa, Ns, Nsa, C=1.4):
        """Calculate the UCB1 value for a given state-action pair."""
        return Qsa + C * torch.sqrt(torch.log(Ns) / Nsa)

    def select(self, color=1, legal_moves=None, q_values=None):
        max_value = np.NINF
        max_move = None
        Ns = torch.tensor(len(self.values)) + 1
        for move in legal_moves:
            Qsa = q_values[0, move.to_square]
            if move in self.children.keys():
                Nsa = torch.tensor(len(self.children[move].values)) + 1
            else:
                Nsa = torch.tensor(1)
            child_value = self.ucb1(Qsa, Ns, Nsa)
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

    def simulate(self, fixed_agent, env, depth=0, max_depth=4, eps=.01):
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
        else:
            move, _ = fixed_agent.select_action(env)
        episode_active, reward = env.step(move)

        if episode_active < 1.:
            Returns = reward
        elif depth >= max_depth:  # Bootstrap the Monte Carlo Playout

            bootstrap_q = fixed_agent.get_q_values(env)
            action_probas = fixed_agent.get_action_probabilities(env)
            state_value = torch.inner(action_probas, bootstrap_q).squeeze()
            Returns = reward + GAMMA * state_value

        else:  # Recursively continue
            Returns = reward + GAMMA * \
                self.simulate(fixed_agent, env, depth=depth + 1)

        env.reverse()

        if depth == 0:
            gc.collect()
            return Returns, move
        else:
            return Returns
