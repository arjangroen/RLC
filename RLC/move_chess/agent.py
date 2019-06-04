import numpy as np
import pprint


class Piece(object):

    def __init__(self,piece='king', k_max=32):
        self.piece = piece
        self.k_max = k_max
        self.synchronous = synchronous
        self.init_actionspace()
        self.value_function = np.zeros(shape=env.reward_space.shape)
        self.N = np.zeros(self.value_function.shape)
        self.Returns = {}
        self.action_function = np.zeros(shape=(env.reward_space.shape[0],
                                               env.reward_space.shape[1],
                                               len(self.action_space)))
        self.E = np.zeros(shape=self.action_function.shape)
        self.policy = np.zeros(shape=self.action_function.shape)
        self.policy_old = self.policy.copy()

    def apply_policy(self,state,epsilon):
        greedy_action_value = np.max(self.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.policy[state[0], state[1], :]) if
                       a == greedy_action_value]
        action_index = np.random.choice(greedy_indices)
        if np.random.uniform(0, 1) < epsilon:
            action_index = np.random.choice(range(len(self.action_space)))
        return action_index

    def compare_policies(self):
        return np.sum(np.abs(self.policy - self.policy_old))


    def init_actionspace(self):
        assert self.piece in ["king", "rook", "bishop",
                              "knight"], f"{self.piece} is not a supported piece try another one"
        if self.piece == 'king':
            self.action_space = [(-1, 0),  # north
                                 (-1, 1),  # north-west
                                 (0, 1),  # west
                                 (1, 1),  # south-west
                                 (1, 0),  # south
                                 (1, -1),  # south-east
                                 (0, -1),  # east
                                 (-1, -1),  # north-east
                                 ]
        elif self.piece == 'rook':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, 0))  # north
                self.action_space.append((0, amplitude))  # east
                self.action_space.append((amplitude, 0))  # south
                self.action_space.append((0, -amplitude))  # west
        elif self.piece == 'knight':
            self.action_space = [(-2, 1),  # north-north-west
                                 (-1, 2),  # n-w-w
                                 (1, 2),  # s-w-w
                                 (2, 1),  # s-s-w
                                 (2, -1),  # s-s-e
                                 (1, -2),  # s-e-e
                                 (-1, -2),  # n-e-e
                                 (-2, -1)]  # n-n-e
        elif self.piece == 'bishop':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, amplitude))  # north-west
                self.action_space.append((amplitude, amplitude))  # south-west
                self.action_space.append((amplitude, -amplitude))  # south-east
                self.action_space.append((-amplitude, -amplitude))  # north

    def evaluate_policy(self):
        self.value_function_old = self.value_function.copy()  # For synchronous updates
        for row in range(self.value_function.shape[0]):
            for col in range(self.value_function.shape[1]):
                self.value_function[row, col] = self.evaluate_state((row, col))



