import numpy as np
import time
import math
import gc
from RLC.chess.montecarlo import PlayOut
from RLC.chess.agent import NeuralNetworkAgent, GreedyAgent
from RLC.chess.environment import Board


def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class TD_search(object):

    def __init__(self, env, white, black, gamma=0.9, search_time=1, memsize=5000, batch_size=64, max_sim_depth=1):
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
        self.white = white
        self.black = black
        self.learning_agent_color = 1
        self.gamma = gamma
        self.memsize = memsize
        self.batch_size = batch_size
        self.reward_trace = []  # Keeps track of the rewards
        self.piece_balance_trace = []  # Keep track of the material value on the board
        self.ready = False  # Whether to start training
        self.search_time = search_time
        self.min_sim_count = 10
        self.max_sim_depth = max_sim_depth

        self.mem_state = np.zeros(shape=(1, 8, 8, 8))
        self.mem_sucstate = np.zeros(shape=(1, 8, 8, 8))
        self.mem_reward = np.zeros(shape=(1))
        self.mem_error = np.zeros(shape=(1))
        self.mem_episode_active = np.ones(shape=(1))

        self.mem_state_mc = np.zeros(shape=(1, 8, 8, 8))
        self.mem_returns_mc = np.zeros(shape=(1))
        self.mem_error_mc = np.zeros(shape=(1))

    def init_random(self, train_iters=100):
        i=0
        while self.mem_state.shape[0] < self.memsize:
            print("init random",self.mem_state.shape[0])
            self.play_random()
            self.env.reset()
            self.env.node.children = {}
            i+=1
        learning_player = self.white if self.learning_agent_color == 1 else self.black
        for j in range(train_iters):
            self.update_agent(learning_player)



    def learn(self, iters=40, c=5, timelimit_seconds=36000, maxiter=80):
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
        learning_player = self.white if self.learning_agent_color == 1 else self.black
        for k in range(iters):
            print("iter", k)
            self.env.reset()
            if k % c == 0:
                learning_player.fix_model()
            if k > c:
                self.ready = True
            self.play_game(k, maxiter=maxiter)
            if starttime + timelimit_seconds < time.time():
                break
        return self.env.board

    def play_random(self):
        episode_end = False
        turncount = 0
        while not episode_end:
            moves = [x for x in self.env.board.generate_legal_moves()]
            move = np.random.choice(moves)
            state = np.expand_dims(self.env.layer_board.copy(), axis=0)
            episode_end, reward = self.env.step(move)
            if turncount > 200:
                episode_end = True
            sucstate = state = np.expand_dims(self.env.layer_board.copy(), axis=0)
            error = 1.
            episode_active = 1 - int(episode_end)
            turncount+=1

        if turncount <= 200:
            print(turncount, reward)
            self.mem_state = np.append(self.mem_state, state, axis=0)
            self.mem_reward = np.append(self.mem_reward, reward)
            self.mem_sucstate = np.append(self.mem_sucstate, sucstate, axis=0)
            self.mem_error = np.append(self.mem_error, error)
            self.reward_trace = np.append(self.reward_trace, reward)
            self.mem_episode_active = np.append(self.mem_episode_active, episode_active)

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

        current_player = self.white if self.env.board.turn else self.black
        learning_player = self.white if self.learning_agent_color == 1 else self.black
        # Play a game of chess
        while not episode_end:
            self.env.node.clean(color=current_player.color)  # Memory management, delete least relevant move

            starttime = time.time()
            n_sims = 0

            # CALCULATE STATE VALUE BEFORE THE MOVE
            state = np.expand_dims(self.env.layer_board, axis=0)
            state_value = learning_player.predict(state)

            # PERFORM MCTS
            while time.time() < starttime + self.search_time and n_sims < self.min_sim_count:
                self.mcts(self.white, self.black)
                n_sims += 1
            move = current_player.select_move_from_node(self.env.node, force_select=True)

            # EXECUTE THE MOVE
            self.env.node.checkpoint = False  # Remove the checkpoint
            episode_end, reward = self.env.step(move)
            turncount += 1

            # CALCULATE NEW STATE VALUE
            sucstate = np.expand_dims(self.env.layer_board, axis=0)
            new_state_value = learning_player.predict(sucstate)

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

            if turncount % 1 == 0:
                self.update_agent(learning_player)
                self.update_agent_mc(learning_player)

        piece_balance = self.env.get_material_value()
        self.piece_balance_trace.append(piece_balance)
        print("game ended with result", reward, "and material balance", piece_balance, "in", turncount, "halfmoves")

        return self.env.board

    def update_agent(self, learning_player):
        """
        Update the Agent with TD learning
        Returns:
            None
        """
        if self.ready:
            choice_indices, states, rewards, sucstates, episode_active = self.get_minibatch()
            td_errors = learning_player.TD_update(states, rewards, sucstates, episode_active, gamma=self.gamma)
            self.mem_error[choice_indices.tolist()] = td_errors

    def update_agent_mc(self, learning_player):
        """
        Learn from Monte Carlo
        :param learning_player:
        :return:
        """
        if self.ready:
            sampling_indices = np.random.choice([x for x in range(self.mem_state_mc.shape[0])],
                                                min(self.mem_state_mc.shape[0],
                                                    self.batch_size),
                                                replace=False)
            states = self.mem_state_mc[sampling_indices]
            returns = self.mem_returns_mc[sampling_indices]
            mc_errors = learning_player.MC_update(states, returns)

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

    def mcts(self, white, black):
        episode_end, returns = self.select(white, black)
        if not episode_end:
            episode_end, reward = self.expand(white, black)
            returns = returns + self.gamma * reward
        if not episode_end:
            returns = returns + self.simulate(white, black)

        self.backprop(returns, gamma=self.gamma)

    def select(self, white, black):
        self.env.node.checkpoint = True
        selected = False
        returns = 0
        episode_end = False
        while not selected:
            current_player = white if self.env.board.turn else black
            move = current_player.select_move_from_node(self.env.node, force_select=False)
            if move:
                episode_end, reward = self.env.step(move)
                returns = returns + self.gamma * reward
            else:
                selected = True
        return episode_end, returns

    def expand(self, white, black):
        successor_values = []
        moves = []
        for move in self.env.board.generate_legal_moves():
            current_player = white if self.env.board.turn else black
            successor_values.append(current_player.evaluate(move, self.env))
            moves.append(move)
        selected_move = current_player.select_move_from_values(moves, successor_values)
        episode_end, reward = self.env.step(selected_move)
        return episode_end, reward

    def simulate(self, white, black):
        playout = PlayOut()
        returns = playout.sim(self.env, white, black, self.learning_agent_color)
        self.env.init_layer_board()
        return returns

    def backprop(self, value, gamma):
        self.env.backprop(value, gamma=gamma)

if __name__ == '__main__':
    white = NeuralNetworkAgent()
    black = NeuralNetworkAgent()
    env = Board()

    learning_process = TD_search(white=white, black=black, env=env)
    learning_process.init_random()
    learning_process.learn(iters=1000, timelimit_seconds=int(1e12))
