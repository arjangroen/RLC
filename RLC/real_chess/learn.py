import numpy as np
import time
from RLC.real_chess.tree import Node
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class TD_search(object):

    def __init__(self, env, agent, lamb=0.9, gamma=0.9, search_time=1):
        self.env = env
        self.agent = agent
        self.tree = Node(self.env)
        self.lamb = lamb
        self.gamma = gamma
        self.memory = []
        self.memsize = 10000
        self.batch_size = 1024
        self.result_trace = []
        self.piece_balance_trace = []
        self.ready = False
        self.search_time = search_time

    def learn(self,iters=40,c=5,timelimit_seconds=3600,maxiter=51):
        starttime = time.time()
        
        for k in range(iters):
            self.env.reset()
            if k % c == 0:
                self.agent.fix_model()
                print("iter",k)
            if k > 3:
                self.ready=True
            self.play_game(k,maxiter=maxiter)
            if starttime + timelimit_seconds < time.time():
                break
        return self.env.board


    def play_game(self,k,maxiter=80):
        """
        Play a game of capture chess
        Args:
            maxiter: int
                Maximum amount of steps per game

        Returns:
        """
        episode_end = False
        turncount = 0
        tree = Node(self.env.board,gamma=self.gamma)

        # Play a game of chess
        while not episode_end:
            state = self.env.layer_board.copy()
            state_value = self.agent.predict(np.expand_dims(state,axis=0))


            # White's turn
            if self.env.board.turn:
                x = (turncount/maxiter - 0.6)*10
                timelimit = self.search_time * sigmoid(x)
                tree = self.mcts(tree,state_value,timelimit, remaining_depth=maxiter-turncount)
                self.env.init_layer_board()
                # Step the best move
                max_move = None
                max_value = np.NINF
                for move, child in tree.children.items():
                    # Thompson
                    sampled_value = np.random.choice(child.values)
                    if sampled_value > max_value:
                        max_value = sampled_value
                        max_move = move

            # Black's turn
            else:
                max_move = None
                max_value = np.NINF
                for move in self.env.board.generate_legal_moves():
                    self.env.step(move)
                    if self.env.board.result() == "0-1":
                        max_move = move
                        self.env.board.pop()
                        self.env.pop_layer_board()
                        break
                    successor_state_value_opponent = self.env.opposing_agent.predict(np.expand_dims(self.env.layer_board,axis=0))
                    if successor_state_value_opponent > max_value:
                        max_move = move
                        max_value = successor_state_value_opponent

                    self.env.board.pop()
                    self.env.pop_layer_board()

            episode_end, reward = self.env.step(max_move)

            if max_move not in tree.children.keys():
                tree.children[max_move] = Node(self.env.board, parent=None)

            tree = tree.children[max_move]
            tree.parent = None

            new_state_value = self.agent.predict(np.expand_dims(self.env.layer_board,axis=0))
            error = reward + self.gamma*new_state_value - state_value
            error = np.float(np.squeeze(error))

            # construct training sample state, prediction, error
            self.memory.append([state.copy(),reward,self.env.layer_board.copy(),np.min([error,1e-3])])

            if len(self.memory) > self.memsize:
                self.memory.pop(0)
            turncount += 1
            if turncount > maxiter and not episode_end:
                episode_end = True

                # before k steps, use material balance as end result, after k steps, bootstrap from model.
                if len(self.memory) < self.memsize:
                    reward = np.clip(self.env.get_material_value()/40,-1,1)
                else:
                    reward = np.squeeze(self.agent.predict(np.expand_dims(self.env.layer_board,axis=0)))

            self.update_agent()

        self.result_trace.append(reward * self.gamma**turncount)
        piece_balance = self.env.get_material_value()
        self.piece_balance_trace.append(piece_balance)
        print("game ended with result",reward, "and material balance",piece_balance, "in",turncount,"halfmoves")

        return self.env.board

    def update_agent(self):
        if self.ready:
            choice_indices, minibatch = self.get_minibatch()

            td_errors = np.squeeze(self.agent.network_update(minibatch,gamma=self.gamma))
            for index, error in zip(choice_indices,td_errors):
                self.memory[index][3] = error

    def get_minibatch(self):
        if len(self.memory) == 1:
            return [0], [self.memory[0]]
        else:
            sampling_priorities = np.abs(np.array([xp[3] for xp in self.memory]))
            sampling_probs = sampling_priorities / np.sum(sampling_priorities)
            sample_indices = [x for x in range(len(self.memory))]
            choice_indices = np.random.choice(sample_indices,
                                              min(len(self.memory),
                                                  self.batch_size),
                                              p=np.squeeze(sampling_probs),
                                              replace=False
                                              )
            minibatch = [self.memory[idx] for idx in choice_indices]
        return choice_indices, minibatch


    def mcts(self,node,statevalue,timelimit,remaining_depth=3):
        """
        Return best node
        :param node:
        :return:
        """
        starttime = time.time()
        sim_count = 0
        while starttime + timelimit > time.time() or sim_count < 3:
            depth = 0
            while node.children:
                node, move = node.select()
                if not move:
                    break
                else:
                    depth += 1
                    # A best node is selected
                    self.env.step(move)
                    # Check best node is terminal
                    if self.env.board.result() == "1-0" or self.env.board.result(claim_draw=False) == "1/2-1/2":

                        # if so, restore and return root node
                        while node.parent:
                            node = node.parent
                            self.env.board.pop()
                            try:
                                self.env.pop_layer_board()
                            except:
                                self.env.init_layer_board()
                        return node

            # Expand the game tree with a simulation
            result, move = node.simulate(self.agent.model,self.env,max(1,remaining_depth-depth),depth=0)
            error = result * self.gamma**depth - statevalue

            # Add the result to memory
            self.memory.append([self.env.layer_board.copy(), result, None, np.float(np.squeeze(error))])

            if move not in node.children.keys():
                node.children[move] = Node(self.env.board,parent=node)

            node.update_child(move,result)

            node = node.children[move]

            # Return to root node
            while node.parent:
                node.backprop(result)
                node = node.parent
                node.update()
                if node.parent:
                    self.env.board.pop()
                    try:
                        self.env.pop_layer_board()
                    except:
                        self.env.init_layer_board()
            sim_count+=1
        return node
