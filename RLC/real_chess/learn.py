import numpy as np
import time
from RLC.real_chess.tree import Node
import math


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TD_search(object):

    def __init__(self, env, agent, lamb=0.9, gamma=0.9, search_time=1, search_balance=0.0):
        self.env = env
        self.agent = agent
        self.tree = Node(self.env)
        self.lamb = lamb
        self.gamma = gamma
        self.memsize = 10000
        self.batch_size = 1024
        self.reward_trace = []
        self.piece_balance_trace = []
        self.ready = False
        self.search_time = search_time
        self.search_balance = search_balance

        self.mem_state = np.zeros(shape=(1, 8, 8, 8))
        self.mem_sucstate = np.zeros(shape=(1, 8, 8, 8))
        self.mem_reward = np.zeros(shape=(1))
        self.mem_error = np.zeros(shape=(1))

        self.mc_state = np.zeros(shape=(1, 8, 8, 8))
        self.mc_state_result = np.zeros(shape=(1))
        self.mc_state_error = np.zeros(shape=(1))
        self.mc_state_probs = np.zeros(shape=(1))

    def learn(self, iters=40, c=5, timelimit_seconds=3600, maxiter=51):
        starttime = time.time()

        for k in range(iters):
            self.env.reset()
            if k % c == 0:
                self.agent.fix_model()
                print("iter", k)
            if k > 3:
                self.ready = True
            self.play_game(k, maxiter=maxiter)
            if starttime + timelimit_seconds < time.time():
                break
        return self.env.board

    def play_game(self, k, maxiter=80):
        """
        Play a game of capture chess
        Args:
            maxiter: int
                Maximum amount of steps per game

        Returns:
        """
        episode_end = False
        turncount = 0
        tree = Node(self.env.board, gamma=self.gamma)
        tree.values = [0]

        # Play a game of chess
        while not episode_end:
            state = np.expand_dims(self.env.layer_board.copy(), axis=0)
            state_value = self.agent.predict([state,np.zeros((1,1))])

            # White's turn
            if self.env.board.turn:

                # Search longer at end than begin
                x = (turncount / maxiter - self.search_balance) * 10
                timelimit = self.search_time * sigmoid(x)

                # Do a Monte Carlo Tree Search
                tree = self.mcts(tree, state_value, timelimit, remaining_depth=maxiter - turncount)
                # Step the best move
                max_move = None
                max_value = np.NINF
                for move, child in tree.children.items():
                    # optimistic
                    sampled_value = np.mean(child.values)
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
                    successor_state_value_opponent = self.env.opposing_agent.predict(
                        np.expand_dims(self.env.layer_board, axis=0))
                    if successor_state_value_opponent > max_value:
                        max_move = move
                        max_value = successor_state_value_opponent

                    self.env.board.pop()
                    self.env.pop_layer_board()

            episode_end, reward = self.env.step(max_move)

            # Move up the tree
            if max_move not in tree.children.keys():
                tree.children[max_move] = Node(self.env.board, parent=None)

            tree = tree.children[max_move]
            tree.parent = None

            #print(tree.values)

            sucstate = np.expand_dims(self.env.layer_board, axis=0)
            new_state_value = self.agent.predict([sucstate,np.zeros((1,1))])

            #print(new_state_value.item())

            if not tree.values:
                tree.values.append(new_state_value.item())

            #print(tree.values)

            error = reward + self.gamma * new_state_value - state_value
            error = np.float(np.squeeze(error))

            # construct training sample state, prediction, error
            self.mem_state = np.append(self.mem_state, state, axis=0)
            self.mem_reward = np.append(self.mem_reward, reward)
            self.mem_sucstate = np.append(self.mem_sucstate, sucstate, axis=0)
            self.mem_error = np.append(self.mem_error, error)
            self.reward_trace = np.append(self.reward_trace, reward)

            if self.mem_state.shape[0] > self.memsize:
                self.mem_state = self.mem_state[1:]
                self.mem_reward = self.mem_reward[1:]
                self.mem_sucstate = self.mem_sucstate[1:]
                self.mem_error = self.mem_error[1:]

            turncount += 1
            if turncount > maxiter and not episode_end:
                episode_end = True

                # Bootstrap and end episode
                reward = np.squeeze(self.agent.predict([np.expand_dims(self.env.layer_board, axis=0),np.zeros((1,1))]))

            #self.update_agent(mc=False)
            #self.update_agent(mc=True)
            self.reinforce_agent()

        piece_balance = self.env.get_material_value()
        self.piece_balance_trace.append(piece_balance)
        print("game ended with result", reward, "and material balance", piece_balance, "in", turncount, "halfmoves")
        if np.abs(reward) == 1:
            print(self.env.board)
            print(self.env.layer_board[1])

        return self.env.board

    def reinforce_agent(self,batch_size=128):
        """

        Returns:

        """
        self.agent.model.fit(x=[self.mc_state[-batch_size:], self.mc_state_probs[-batch_size:]],
                             y=np.ones((batch_size,1)))


    def update_agent(self,mc=False):
        if self.ready:

            if mc:
                choice_indices, states, results = self.get_mc_minibatch(prioritized=True)
                td_errors = self.agent.MC_update(states, results)
                self.mc_state_error[choice_indices.tolist()] = td_errors

            else:
                choice_indices, states, rewards, sucstates = self.get_minibatch()

                td_errors = self.agent.TD_update(states, rewards, sucstates, gamma=self.gamma)
                self.mem_error[choice_indices.tolist()] = td_errors

    def get_minibatch(self, prioritized=True):
        if prioritized:
            sampling_priorities = np.abs(self.mem_error) + 1e-9
        else:
            sampling_priorities = np.ones(shape=self.mem_error.shape)
        sampling_probs = sampling_priorities / np.sum(sampling_priorities)
        sample_indices = [x for x in range(self.mem_state.shape[0])]
        choice_indices = np.random.choice(sample_indices,
                                          min(self.mem_state.shape[0],
                                              self.batch_size),
                                          p=np.squeeze(sampling_probs),
                                          replace=False
                                          )
        states = self.mem_state[choice_indices]
        rewards = self.mem_reward[choice_indices]
        sucstates = self.mem_sucstate[choice_indices]

        return choice_indices, states, rewards, sucstates

    def get_mc_minibatch(self, prioritized=True):
        if prioritized:
            sampling_priorities = np.abs(self.mc_state_error) + 1e-9
        else:
            sampling_priorities = np.ones(shape=self.mc_state_error.shape)
        sampling_probs = sampling_priorities / np.sum(sampling_priorities)
        sample_indices = [x for x in range(self.mc_state.shape[0])]
        choice_indices = np.random.choice(sample_indices,
                                          min(self.mc_state.shape[0],
                                              self.batch_size),
                                          p=np.squeeze(sampling_probs),
                                          replace=False
                                          )
        states = self.mc_state[choice_indices]
        results = self.mc_state_result[choice_indices]

        return choice_indices, states, results

    def mcts(self, node, statevalue, timelimit, remaining_depth=3):
        """
        Return best node
        :param node:
        :return:
        """
        starttime = time.time()
        sim_count = 0
        while starttime + timelimit > time.time() or sim_count < 3:
            depth = 0
            color = 1
            while node.children:
                node, move = node.select(color=color)
                if not move:
                    break
                else:
                    depth += 1
                    color = color * -1  # switch color
                    # A best node is selected
                    self.env.step(move)
                    # Check best node is terminal
                    if self.env.board.result() == "1-0" or self.env.board.result(claim_draw=False) == "1/2-1/2" or \
                            self.env.board.result() == "0-1":

                        # if so, restore and return root node
                        while node.parent:
                            node = node.parent
                            self.env.board.pop()
                        self.env.init_layer_board()
                        return node

            # Expand the game tree with a simulation
            result, move, value_grads, target_index = node.simulate(self.agent.model,
                                         self.env,
                                         np.max([
                                             1,
                                             remaining_depth - depth
                                         ]),
                                         depth=0)
            self.env.init_layer_board()
            error = result * self.gamma ** depth - statevalue

            self.env.step(move)

            ## Add the result to memory
            self.mc_state = np.append(self.mc_state, np.expand_dims(self.env.layer_board.copy(), axis=0), axis=0)
            self.mc_state_result = np.append(self.mc_state_result, result)
            self.mc_state_probs = np.append(self.mc_state_probs, prob)
            self.mc_state_error = np.append(self.mc_state_error, error)

            if self.mc_state.shape[0] > self.memsize:
                self.mc_state = self.mc_state[1:]
                self.mc_state_result = self.mc_state_result[1:]
                self.mc_state_error = self.mc_state_error[1:]

            if move not in node.children.keys():
                node.children[move] = Node(self.env.board, parent=node)

            node.update_child(move, result)

            node = node.children[move]
            depth += 1

            # Return to root node
            while node.parent:
                node = node.parent
                result = self.gamma * result
                node.update(result)
                if node.parent:
                    self.env.board.pop()

            self.env.init_layer_board()
            sim_count += 1
        return node
