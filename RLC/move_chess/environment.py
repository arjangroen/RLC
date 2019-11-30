import pprint
import numpy as np


class Board(object):

    def __init__(self):
        self.state = (0, 0)
        self.reward_space = np.zeros(shape=(8, 8)) - 1
        self.terminal_state = (7, 5)

    def step(self, action):
        reward = self.reward_space[self.state[0], self.state[1]]
        if self.state == self.terminal_state:
            episode_end = True
            return 0, episode_end
        else:
            episode_end = False
            old_state = self.state
            new_state = (self.state[0] + action[0], self.state[1] + action[1])  # step
            self.state = old_state if np.min(new_state) < 0 or np.max(new_state) > 7 else new_state
            return reward, episode_end

    def render(self):
        visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())
        visual_board[self.state[0]][self.state[1]] = "[S]"
        visual_board[self.terminal_state[0]][self.terminal_state[1]] = "[F]"
        self.visual_board = visual_board
