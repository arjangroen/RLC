import torch

import chess
import numpy as np
from RLC.real_chess.tree import Node
import warnings
from RLC.real_chess.hyperparams import GAMMA

mapper = {}
mapper["p"] = 0
mapper["r"] = 1
mapper["n"] = 2
mapper["b"] = 3
mapper["q"] = 4
mapper["k"] = 5
mapper["P"] = 0
mapper["R"] = 1
mapper["N"] = 2
mapper["B"] = 3
mapper["Q"] = 4
mapper["K"] = 5

moves_mirror = np.array([i for i in range(64)]).reshape(8, 8)[::-1].reshape(64)


class ChessEnv(object):

    def __init__(self):
        """
        Chess Board Environment
        """
        self.board = chess.Board()
        self.layer_board = torch.zeros(size=(1, 8, 8, 8))
        self.init_layer_board()
        self.node = Node()

    def init_layer_board(self):
        """
        Initalize the numerical representation of the environment
        Returns:

        """
        self.layer_board = torch.zeros(size=(1, 8, 8, 8))
        for i in range(64):
            row = i // 8
            col = i % 8
            piece = self.board.piece_at(i)
            if piece == None:
                continue
            elif piece.symbol().isupper():
                sign = 1
            else:
                sign = -1
            layer = mapper[piece.symbol()]
            self.layer_board[0, layer, row, col] = sign
        self.layer_board[0, 6, :, :] = 1. if self.board.turn else -1.

    @property
    def layer_board_mirror(self):
        layer_board_mirror = self.layer_board.flip(dims=(2,)).clone()
        layer_board_mirror = layer_board_mirror * -1.
        return layer_board_mirror

    def update_layer_board(self, move=None):
        self._prev_layer_board = self.layer_board.clone()
        self.init_layer_board()

    def pop_layer_board(self):
        self.layer_board = self._prev_layer_board.clone()
        self._prev_layer_board = None

    def step(self, move):
        """
        Run a step
        Args:
            move: python chess move
        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: float
                Difference in material value after the move
        """
        piece_balance_before = self.get_material_value()
        self.board.push(move)
        if not self.board.is_valid():
            warnings.warn("Invalid position.")
        self.update_layer_board(move)
        piece_balance_after = self.get_material_value()
        auxiliary_reward = (piece_balance_after - piece_balance_before)
        result = self.board.result()
        if result == "*":
            reward = torch.tensor(0.)
            episode_active = torch.tensor(1.)
        elif result == "1-0":
            reward = torch.tensor(10.)
            episode_active = torch.tensor(0.)
        elif result == "0-1":
            reward = torch.tensor(-10.)
            episode_active = torch.tensor(0.)
        elif result == "1/2-1/2":
            reward = torch.tensor(0.)
            episode_active = torch.tensor(0.)

        if reward == 0:
            reward = reward + auxiliary_reward

        return episode_active, reward

    def project_legal_moves(self):
        """
        Create a mask of legal actions
        Returns: np.ndarray with shape (64,64)
        """
        self.action_space = torch.zeros(size=(1, 64))
        moves = set([x.to_square for x in self.board.generate_legal_moves()])
        for move in moves:
            self.action_space[0, move] = 1.
        return self.action_space

    def get_material_value(self):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * self.layer_board[0, 0, :, :].sum()
        rooks = 5 * self.layer_board[0, 1, :, :].sum()
        minor = 3 * self.layer_board[0, 2:4, :, :].sum()
        queen = 9 * self.layer_board[0, 4, :, :].sum()
        return pawns + rooks + minor + queen

    def reset(self):
        """
        Reset the environment
        Returns:

        """
        self.board = chess.Board()
        self.init_layer_board()
        self.node = self.node.get_root()
        self.node.children = {}

    def reverse(self):
        self.board.pop()
        self.init_layer_board()
