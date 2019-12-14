import numpy as np
import time
from RLC.real_chess.tree import Node
import math
import gc


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TD_search(object):

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
        self.tree = Node(self.env)
        self.gamma = gamma
        self.memsize = memsize
        self.batch_size = batch_size
        self.temperature = temperature
        self.reward_trace = []  # Keeps track of the rewards
        self.piece_balance_trace = []  # Keep track of the material value on the board
        self.ready = False  # Whether to start training
        self.search_time = search_time
        self.min_sim_count = 10

        self.mem_state = np.zeros(shape=(1, 8, 8, 8))
        self.mem_sucstate = np.zeros(shape=(1, 8, 8, 8))
        self.mem_reward = np.zeros(shape=(1))
        self.mem_error = np.zeros(shape=(1))
        self.mem_episode_active = np.ones(shape=(1))

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
                self.agent.fix_model()
                print("iter", k)
            if k > c:
                self.ready = True
            self.play_game(k, maxiter=maxiter)
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
        episode_end = False
        turncount = 0
        tree = Node(self.env.board, gamma=self.gamma)  # Initialize the game tree

        # Play a game of chess
        while not episode_end:
            state = np.expand_dims(self.env.layer_board.copy(), axis=0)
            state_value = self.agent.predict(state)

            # White's turn involves tree-search
            if self.env.board.turn:

                # Do a Monte Carlo Tree Search after game iteration k
                start_mcts_after = -1
                if k > start_mcts_after:
                    tree = self.mcts(tree)
                    # Step the best move
                    max_move = None
                    max_value = np.NINF
                    for move, child in tree.children.items():
                        sampled_value = np.mean(child.values)
                        if sampled_value > max_value:
                            max_value = sampled_value
                            max_move = move
                else:
                    max_move = np.random.choice([move for move in self.env.board.generate_legal_moves()])

            # Black's turn is myopic
            else:
                max_move = None
                max_value = np.NINF
                for move in self.env.board.generate_legal_moves():
                    self.env.step(move)
                    if self.env.board.result() == "0-1":
                        max_move = move
                        self.env.board.pop()
                        self.env.init_layer_board()
                        break
                    successor_state_value_opponent = self.env.opposing_agent.predict(
                        np.expand_dims(self.env.layer_board, axis=0))
                    if successor_state_value_opponent > max_value:
                        max_move = move
                        max_value = successor_state_value_opponent

                    self.env.board.pop()
                    self.env.init_layer_board()

            if not (self.env.board.turn and max_move not in tree.children.keys()) or not k > start_mcts_after:
                tree.children[max_move] = Node(gamma=0.9, parent=tree)

            episode_end, reward = self.env.step(max_move)

            tree = tree.children[max_move]
            tree.parent = None
            gc.collect()

            sucstate = np.expand_dims(self.env.layer_board, axis=0)
            new_state_value = self.agent.predict(sucstate)

            error = reward + self.gamma * new_state_value - state_value
            error = np.float(np.squeeze(error))

            turncount += 1
            if turncount > maxiter and not episode_end:
                episode_end = True

            episode_active = 0 if episode_end else 1

            # construct training sample state, prediction, error
            self.mem_state = np.append(self.mem_state, state, axis=0)
            self.mem_reward = np.append(self.mem_reward, reward)
            self.mem_sucstate = np.append(self.mem_sucstate, sucstate, axis=0)
            self.mem_error = np.append(self.mem_error, error)
            self.reward_trace = np.append(self.reward_trace, reward)
            self.mem_episode_active = np.append(self.mem_episode_active, episode_active)

            if self.mem_state.shape[0] > self.memsize:
                self.mem_state = self.mem_state[1:]
                self.mem_reward = self.mem_reward[1:]
                self.mem_sucstate = self.mem_sucstate[1:]
                self.mem_error = self.mem_error[1:]
                self.mem_episode_active = self.mem_episode_active[1:]
                gc.collect()

            if turncount % 10 == 0:
                self.update_agent()

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
        if self.ready:
            choice_indices, states, rewards, sucstates, episode_active = self.get_minibatch()
            td_errors = self.agent.TD_update(states, rewards, sucstates, episode_active, gamma=self.gamma)
            self.mem_error[choice_indices.tolist()] = td_errors

    def get_minibatch(self, prioritized=True):
        """
        Get a mini batch of experience
        Args:
            prioritized:

        Returns:

        """
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
        episode_active = self.mem_episode_active[choice_indices]

        return choice_indices, states, rewards, sucstates, episode_active

    def mcts(self, node):
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

        # First make a prediction for each child state
        for move in self.env.board.generate_legal_moves():
            if move not in node.children.keys():
                node.children[move] = Node(self.env.board, parent=node)

            episode_end, reward = self.env.step(move)

            if episode_end:
                successor_state_value = 0
            else:
                successor_state_value = np.squeeze(
                    self.agent.model.predict(np.expand_dims(self.env.layer_board, axis=0))
                )

            child_value = reward + self.gamma * successor_state_value

            node.update_child(move, child_value)
            self.env.board.pop()
            self.env.init_layer_board()
        if not node.values:
            node.values = [0]

        while starttime + self.search_time > time.time() or sim_count < self.min_sim_count:
            depth = 0
            color = 1
            node_rewards = []

            # Select the best node from where to start MCTS
            while node.children:
                node, move = node.select(color=color)
                if not move:
                    # No move means that the node selects itself, not a child node.
                    break
                else:
                    depth += 1
                    color = color * -1  # switch color
                    episode_end, reward = self.env.step(move)  # Update the environment to reflect the node
                    node_rewards.append(reward)
                    # Check best node is terminal

                    if self.env.board.result() == "1-0" and depth == 1:  # -> Direct win for white, no need for mcts.
                        self.env.board.pop()
                        self.env.init_layer_board()
                        node.update(1)
                        node = node.parent
                        return node
                    elif episode_end:  # -> if the explored tree leads to a terminal state, simulate from root.
                        while node.parent:
                            self.env.board.pop()
                            self.env.init_layer_board()
                            node = node.parent
                        break
                    else:
                        continue

            # Expand the game tree with a simulation
            Returns, move = node.simulate(self.agent.fixed_model,
                                          self.env,
                                          temperature=self.temperature,
                                          depth=0)
            self.env.init_layer_board()

            if move not in node.children.keys():
                node.children[move] = Node(self.env.board, parent=node)

            node.update_child(move, Returns)

            # Return to root node and backpropagate Returns
            while node.parent:
                latest_reward = node_rewards.pop(-1)
                Returns = latest_reward + self.gamma * Returns
                node.update(Returns)
                node = node.parent

                self.env.board.pop()
                self.env.init_layer_board()
            sim_count += 1

        board_out = self.env.board.fen()
        assert board_in == board_out

        return node
