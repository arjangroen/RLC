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
        self.sampling_probs = []


    def learn(self,iters=100,c=3):
        for k in range(iters):
            print(k)
            if k % c == 0:
                self.agent.fix_model()
            self.play_game(k)
            pgn = Game.from_board(self.env.board)
            self.env.reset()


        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return pgn

    def play_game(self,k,maxiter=25):
        """
        make
        :return:
        """
        episode_end = False
        turncount = 0
        eps = max(0.05,1/(1+(k/250)))
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
                reward = 0
            if episode_end:
                new_state = new_state * 0
            self.memory.append([state,(move_from, move_to),reward,new_state])
            self.sampling_probs.append(1)


            self.reward_trace.append(reward)

            self.update_agent(turncount)

        return self.env.board

    def sample_memory(self,turncount):
        minibatch = []
        memory = self.memory[:-turncount]
        probs = self.sampling_probs[:-turncount]
        sample_probs = [probs[n]/np.sum(probs) for n in range(len(probs))]
        indices = np.random.choice(range(len(memory)),min(64,len(memory)),replace=False,p=sample_probs)
        for i in indices:
            minibatch.append(memory[i])

        return minibatch, indices

    def update_agent(self,turncount):
        if turncount < len(self.memory):
            minibatch,indices = self.sample_memory(turncount)
            td_errors = self.agent.network_update(minibatch)
            print(np.min(td_errors),np.max(td_errors),np.mean(td_errors))
            for n,i in enumerate(indices):
                self.sampling_probs[i] = np.abs(td_errors[n])











