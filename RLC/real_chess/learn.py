import numpy as np
import time
import gc
import torch
import pandas as pd


class ReinforcementLearning(object):

    def __init__(self, env, agent, gamma=0.8, search_time=1, memsize=512, batch_size=512):
        """
        Chess algorithm that combines bootstrapped monte carlo tree search with Q Learning
        Args:
            env: RLC chess environment
            agent: RLC chess agent
            gamma: discount factor
            search_time: maximum time spent doing tree search
            memsize: Amount of training samples to keep in-memory
            batch_size: Size of the training batches
        """
        self.env = env
        self.agent = agent  # The agent that's learning
        self.fixed_agent = type(agent)()  # The agent that's bootstrapped from
        self.fixed_agent.load_state_dict(self.agent.state_dict())
        self.gamma = gamma  # Discount rate for future returns
        self.memsize = memsize  # Amount of steps to be kept in memory for experience replay
        self.batch_size = batch_size  # Batch size for TD Learning
        self.reward_trace = []  # Keeps track of the rewards
        self.piece_balance_trace = []  # Keep track of the material value on the board
        self.search_time = search_time  # How long to do MCTS
        self.min_sim_count = 1  # Minimal amount of playouts in MCTS
        self.episode_memory = []  # Experience replay memory
        self.best_so_far = 0.001  # Best test result so far
        self.mc_replay_buffer = []
        self.maxiter = 10

    def learn(self, iters=400, c=10, timelimit_seconds=80000, test_at_zero=False):
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
            if k % c == 0 and (test_at_zero or k > 0):

                train_on_buffer_frac = 1/4
                n_samples = int(len(self.mc_replay_buffer)
                                * train_on_buffer_frac)
                for _ in range(n_samples):
                    training_example = self.mc_replay_buffer.pop(
                        np.random.randint(0, len(self.mc_replay_buffer)))
                    self.agent.mc_update_agent(
                        *training_example, self.fixed_agent)
                self.test(k)
                print("iter", k)
            self.play_game(k, maxiter=self.maxiter)
            # self.td_update_agent()
            if starttime + timelimit_seconds < time.time():
                break
        return self.env.board

    def test(self, k):
        """
        Test the performance against the fixed agent or against a random agent.
        :param int k: test iteration
        :return:
        """
        results = []
        testsize = 100
        for i in range(testsize):
            results.append(self.test_game(
                self.agent, self.fixed_agent, random=-1))
        for i in range(testsize):
            results.append(self.test_game(
                self.fixed_agent, self.agent, random=1))
        end_result = pd.DataFrame(results)
        end_result.columns = ['Return']
        end_result['color'] = ['white'] * testsize + ['black'] * testsize

        if end_result['Return'].mean() > self.best_so_far:
            print("replacing fixed agent by updated agent")
            self.fixed_agent.load_state_dict(self.agent.state_dict())
            self.best_so_far = end_result['Return'].mean()
            self.save_agent(k='best')
            self.maxiter += 2
        else:
            self.save_agent(k=str(k))
        end_result.to_csv('end_result_' + str(k))

    def save_agent(self, k='last'):
        checkpoint = {'agent': type(self.agent)(),
                      'critic_state_dict': self.agent.critic.state_dict(),
                      'critic_optimizer': self.agent.critic_optimizer.state_dict(),
                      'actor_state_dict': self.agent.actor.state_dict(),
                      'actor_optimizer': self.agent.actor_optimizer.state_dict(),
                      }
        torch.save(checkpoint, f'agent_{k}.pth')

    def test_game(self, white, black, random=None):
        """
        Play a test game
        :param white: Player who plays as white
        :param black: Player who plays as white
        :param random: color who plays random (1 for white, -1 for black, else both agents play their policy)
        :return:
        """

        episode_end = False
        turncount = 0
        color = 1
        self.env.node = self.env.node.get_root()  # Initialize the game tree
        self.env.reset()
        maxiter = self.maxiter
        Returns = 0

        # Play a game of chess

        while not episode_end:
            current_player = white if color == 1 else black

            if color == random:
                move = np.random.choice(
                    [x for x in self.env.board.generate_legal_moves()])
            else:
                move, _ = current_player.select_action(self.env, greedy=False)
            episode_end, reward = self.env.step(move)
            color = color * -1
            Returns += reward

            gc.collect()

            turncount += 1
            if turncount >= maxiter and not episode_end:
                episode_end = True

        if self.agent == black:
            Returns *= -1

        piece_balance = self.env.get_material_value()
        self.piece_balance_trace.append(piece_balance)
        print("game ended with Returns", Returns)

        return Returns

    def play_game(self, k, maxiter=100):
        """
        Play a chess game and learn from it
        Args:
            k: the play iteration number
            maxiter: maximum duration of the game (halfmoves)

        Returns:
            board: Chess environment on terminal state
        """

        # INITIALIZE GAME STATE
        memory_sara = []  # state, action, reward, actionspace
        episode_end = False
        turncount = 0
        color = 1
        self.env.node = self.env.node.get_root()  # Initialize the game tree
        self.env.reset()
        # Play a game of chess

        while not episode_end:
            state = torch.from_numpy(np.expand_dims(
                self.env.layer_board, axis=0)).float()
            action_space = torch.from_numpy(np.expand_dims(
                self.env.project_legal_moves(), axis=0)).float()

            # Do a Monte Carlo Tree Search after game iteration k
            self.mcts(color)
            max_move = None
            max_value = np.NINF
            for move, child in self.env.node.children.items():
                sampled_value = np.max(child.values) * color
                if sampled_value > max_value:
                    max_value = sampled_value
                    max_move = move

            # Execute the best move
            episode_end, reward = self.env.step(max_move)
            self.env.node = self.env.node.children[max_move]

            memory_sara.append([state, max_move, reward, action_space])

            color = color * -1

            gc.collect()

            turncount += 1
            if turncount >= maxiter and not episode_end:
                episode_end = True

        self.episode_memory.append(memory_sara)

        piece_balance = self.env.get_material_value()
        print("game ended with result", reward, "and material balance",
              piece_balance, "in", turncount, "halfmoves")

        return self.env.board

    def td_update_agent(self):

        episode_actives, states, moves, rewards, successor_states, successor_actions, action_spaces = self.get_minibatch()
        for i in range(1):
            self.agent.td_update(self.fixed_agent, episode_actives, states, moves, rewards, successor_states,
                                 successor_actions, action_spaces)

        if len(self.episode_memory) > self.memsize:
            self.episode_memory = self.episode_memory[1:]

    def get_minibatch(self):
        """
        Update the Agent with TD learning
        Returns:
            episode_active: Tensor of shape (n_samples, 1)
            current_turn: Tensor of shape (n_samples, 1)
            states: Tensor of shape (n_samples, 8, 8, 8)
            moves: list[chess.move]
            rewards: Tensor of shape (n_samples, 1)
            successor_states: Tensor of shape (n_samples, 8, 8, 8)
            successor_action: list[chess.move]
        """
        episode_actives = []
        states = []
        moves = []
        rewards = []
        successor_states = []
        successor_actions = []
        action_spaces = []

        for i in range(len(self.episode_memory)):
            episode = self.episode_memory[i]
            episode_len = len(episode)
            subtract = np.random.choice([2, 3, 4, 5])
            if episode_len < subtract:
                continue
            learn_event_index = episode_len - subtract

            episode_active = torch.tensor([0.]).float() if learn_event_index == episode_len - 1 else torch.tensor(
                [1.]).float()
            state, action, reward = episode[learn_event_index][0], episode[learn_event_index][1], \
                episode[learn_event_index][2]
            successor_state, successor_action, _ = episode[learn_event_index + 1][0], episode[learn_event_index + 1][1], \
                episode[learn_event_index + 1][2]
            action_space = episode[learn_event_index][3]
            episode_actives.append(episode_active)
            states.append(state)
            moves.append(action)
            rewards.append(reward)
            successor_states.append(successor_state)
            successor_actions.append(successor_action)
            action_spaces.append(action_space)
            self.episode_memory[i] = self.episode_memory[i][:-subtract]

        episode_actives = torch.cat(episode_actives, dim=0).unsqueeze(dim=1)
        states = torch.cat(states, dim=0)
        rewards = torch.tensor(rewards).unsqueeze(dim=1)
        successor_states = torch.cat(successor_states, dim=0)
        action_spaces = torch.cat(action_spaces)

        return episode_actives, states, moves, rewards, successor_states, successor_actions, action_spaces

    def mcts(self, starting_color):
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

        if not self.env.node.values:
            self.env.node.values = [0]

        while starttime + self.search_time > time.time() or sim_count < self.min_sim_count:
            depth = 0
            Returns = 0
            episode_end = False
            node_rewards = []
            color = starting_color

            # 1. Select the best node from where to start MCTS
            while self.env.node.children:
                legal_moves = self.env.board.generate_legal_moves()
                state = torch.from_numpy(np.expand_dims(
                    self.env.layer_board, axis=0)).float()
                _, q_values = self.fixed_agent(state)
                self.env.node, move = self.env.node.select(
                    color=color, legal_moves=legal_moves, q_values=q_values)
                depth += 1
                color = color * -1  # switch color
                # Update the environment to reflect the node
                episode_end, reward = self.env.step(move)
                node_rewards.append(reward)

            # 2. Expand the game tree with a new move
            if not episode_end:
                move = None
                loop_max = 20
                loop = 0
                while move in self.env.node.children.keys() or not move:
                    move, _ = self.fixed_agent.select_action(self.env)
                    loop += 1
                    if loop > loop_max:
                        break
                episode_end, reward = self.env.step(move)
                if move not in self.env.node.children.keys():
                    self.env.node.add_child(move)
                self.env.node = self.env.node.get_down(move)
                depth += 1
                color = color * -1
                node_rewards.append(reward)

            # 3. Monte Carlo Simulation to make a proxy node value
            if not episode_end:
                Returns, move = self.env.node.simulate(self.fixed_agent,
                                                       self.env,
                                                       depth=0)
                action_space = self.env.project_legal_moves()
                for epochs in range(1):
                    self.mc_replay_buffer.append((self.env.layer_board.copy(),
                                                  move,
                                                  Returns,
                                                  action_space.copy(),
                                                  ))
            else:
                Returns = 0  # episode is over, no future returns

            # 4. Backpropagate Returns
            self.env.node.update(Returns)
            while depth > 0:
                self.env.reverse()
                self.env.node = self.env.node.parent
                latest_reward = node_rewards.pop(-1)
                Returns = latest_reward + self.gamma * Returns
                self.env.node.update(Returns)

                depth -= 1

            sim_count += 1

            board_out = self.env.board.fen()
            assert board_in == board_out
