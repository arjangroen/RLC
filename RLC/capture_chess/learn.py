from RLC.capture_chess.agent import Agent
from RLC.capture_chess.environment import Board
import numpy as np
from chess.pgn import Game
import pandas as pd

class Reinforce(object):

    def __init__(self,agent,env,memsize=10000):
        self.agent = agent
        self.env = env
        self.memory = []
        self.memsize=memsize
        self.reward_trace = []


    def learn(self,iters=100,c=3):
        for k in range(iters):
            print(k)
            if k % c == 0:
                self.agent.fix_model()
            self.play_game(k)
            self.env.reset()

        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return Game.from_board(self.env.board)

    def play_game(self,k):
        """
        make
        :return:
        """
        maxiter=100
        episode_end = False
        turncount = 0
        eps = max(0.1,1/(1+(k/10)))
        while not episode_end:
            state = self.env.layer_board
            explore = np.random.uniform(0,1) < eps
            if explore:
                move = self.env.get_random_action()
                move_from = move.from_square
                move_to = move.to_square
            else:
                action_values = self.agent.get_action_values(np.expand_dims(state,axis=0))
                action_values = np.reshape(np.squeeze(action_values),(64,64))
                action_space = self.env.project_legal_moves()
                action_values = np.multiply(action_values,action_space)
                move_from = np.argmax(action_values,axis=None) // 64
                move_to = np.argmax(action_values,axis=None) % 64
                moves = [x for x in self.env.board.generate_legal_moves() if\
                        x.from_square == move_from and x.to_square == move_to]
                if len(moves) == 0:
                    print(np.max(action_values))
                    move = self.env.get_random_action()
                    move_from = move.from_square
                    move_to = move.to_square
                else:
                    move = moves[0]

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
            turncount += 1
            if turncount > maxiter:
                # After more than maxiter moves, we take the piece balance as the result
                episode_end = True
                reward = self.env.get_material_value()
            self.memory.append([state,(move_from, move_to),reward,new_state])
            self.update_agent()

        self.reward_trace.append(reward)

        return self.env.board

    def sample_memory(self):
        minibatch = []
        indices = np.random.choice(range(len(self.memory)),min(64,len(self.memory)),replace=False)
        for i in indices:
            minibatch.append(self.memory[i])

        return minibatch

    def update_agent(self):
        minibatch = self.sample_memory()
        self.agent.network_update(minibatch)








