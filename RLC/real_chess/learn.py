import numpy as np
import time
from RLC.real_chess.tree import Node
import math
import gc
import torch


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class ReinforcementLearning(object):

    def __init__(self, env, agent, gamma=0.9, search_time=1, memsize=2000, batch_size=256, temperature=1):
        """
        Chess algorithm that combines bootstrapped monte carlo tree search with Q Learning
        Args:
            env: RLC chess environment
            agent: RLC chess agent
            gamma: discount factor
            search_time: maximum time spent doing tree search
            memsize: Amount of training samples to keep in-memory
            batch_size: Size of the training batches
            temperature: softmax temperature for mcts
        """
        self.env = env
        self.agent = agent
        self.fixed_agent = type(agent)()
        self.fixed_agent.load_state_dict(self.agent.state_dict())
        self.gamma = gamma
        self.memsize = memsize
        self.batch_size = batch_size
        self.temperature = temperature
        self.reward_trace = []  # Keeps track of the rewards
        self.piece_balance_trace = []  # Keep track of the material value on the board
        self.ready = False  # Whether to start training
        self.search_time = search_time
        self.min_sim_count = 10

        self.episode_memory = []

    def learn(self, iters=40, c=5, timelimit_seconds=3600, maxiter=80):
        """
        Start Reinforcement Learning Algorithm
        Args:
            iters: maximum amount of iterations to train
            c: model update rate (once every C games)
            timelimit_seconds: maximum training time
            maxiter: Maximum duration of a game, in halfmoves
        Returns:

        """
        starttime = time.time()
        for k in range(iters):
            self.env.reset()
            if k % c == 0:
                self.fixed_agent.load_state_dict(self.agent.state_dict())
                print("iter", k)
            if k > c:
                self.ready = True
            self.play_game(k, maxiter=maxiter)
            self.update_agent()
            if starttime + timelimit_seconds < time.time():
                break
        return self.env.board

    def play_game(self, k, maxiter=80):
        """
        Play a chess game and learn from it
        Args:
            k: the play iteration number
            maxiter: maximum duration of the game (halfmoves)

        Returns:
            board: Chess environment on terminal state
        """

        # INITIALIZE GAME STATE
        memory_sar = []  # state, action, reward
        episode_end = False
        turncount = 0
        color = 1
        tree = self.env.tree.get_root()  # Initialize the game tree
        self.env.board.reset()
        # Play a game of chess

        while not episode_end:
            state = torch.from_numpy(np.expand_dims(self.env.layer_board, axis=0)).float()

            # Do a Monte Carlo Tree Search after game iteration k
            tree = self.mcts(tree, color)
            max_move = None
            max_value = np.NINF
            for move, child in tree.children.items():
                sampled_value = np.max(child.values) * color
                if sampled_value > max_value:
                    max_value = sampled_value
                    max_move = move

            # Execute the best move
            episode_end, reward = self.env.step(max_move)
            color = color * -1

            tree = tree.children[max_move]
            memory_sar.append([state, max_move, reward])

            gc.collect()

            turncount += 1
            if turncount > maxiter and not episode_end:
                episode_end = True

        self.episode_memory.append(memory_sar)

        piece_balance = self.env.get_material_value()
        self.piece_balance_trace.append(piece_balance)
        print("game ended with result", reward, "and material balance", piece_balance, "in", turncount, "halfmoves")

        return self.env.board

    def update_agent(self):
        """
        Update the Agent with TD learning
        Returns:
            None
        """
        minibatch = self.get_minibatch()
        for event in minibatch:
            self.agent.network_update(self.fixed_agent, event[0], event[1], event[2], event[3], event[4], event[5])

        if len(self.episode_memory) > self.memsize:
            self.episode_memory = self.episode_memory[1:]

    def get_minibatch(self):
        """
        Get a mini batch of experience
        Args:
            prioritized:

        Returns:

        """
        minibatch = []
        for episode in self.episode_memory:
            episode_len = len(episode)
            learn_event_index = np.random.choice(range(episode_len-1))
            episode_end = True if learn_event_index == episode_len-1 else False
            state, action, reward = episode[learn_event_index][0], episode[learn_event_index][1], \
                                    episode[learn_event_index][2]
            successor_state, successor_action, _ = episode[learn_event_index+1][0], episode[learn_event_index+1][1], \
                                    episode[learn_event_index+1][2]
            minibatch.append([episode_end, state, action, reward, successor_state, successor_action])
        return minibatch

    def mcts(self, color):
        """
        Run Monte Carlo Tree Search
        Args:
            node: A game state node object

        Returns:
            the node with playout sims

        """

        starttime = time.time()
        sim_count = 0
        board_in = self.env.board.fen()

        if not self.env.tree.values:
            self.env.tree.values = [0]

        while starttime + self.search_time > time.time() or sim_count < self.min_sim_count:
            depth = 0
            Returns = 0
            episode_end = False
            node_rewards = []

            # 1. Select the best node from where to start MCTS
            while node.children:
                node, move = self.env.tree.select(color=color)
                if not move:
                    # No move means that the node selects itself, not a child node.
                    break
                else:
                    depth += 1
                    color = color * -1  # switch color
                    episode_end, reward = self.env.step(move)  # Update the environment to reflect the node
                    node_rewards.append(reward)

            # 2. Expand the game tree with a new move
            if not episode_end:
                move = None
                loop_max = 20
                loop = 0
                while move in node.children.keys() or not move:
                    move, _ = self.agent.select_action(self.env)
                    loop += 1
                    if loop > loop_max:
                        break
                episode_end, reward = self.env.step(move)
                depth += 1
                node_rewards.append(reward)

            # 3. Monte Carlo Simulation to make a proxy node value
            if not episode_end:
                Returns, move = node.simulate(self.fixed_agent,
                                              self.env,
                                              temperature=self.temperature,
                                              depth=0)
                node.update(Returns)
            else:
                node.update(reward)

            # 4. Backpropagate Returns
            while depth > 0:
                node = node.parent
                latest_reward = node_rewards.pop(-1)
                Returns = latest_reward + self.gamma * Returns
                node.update(Returns)

                self.env.board.pop()
                self.env.init_layer_board()
                depth -= 1

            sim_count += 1

        board_out = self.env.board.fen()
        assert board_in == board_out

        return node
