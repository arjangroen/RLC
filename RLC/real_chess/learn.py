import numpy as np
import time
import gc
import torch
import pandas as pd
import logging
import os
from hyperparams import GAMMA
import chess
from experiences import TemporalDifferenceStep, TemporalDifferenceTrainingSample, MonteCarloTrainingSample,\
    MonteCarloTrainingData, TemporalDifferenceMemory, TemporalDifferenceEpisode, TemporalDifferenceTrainingBatch, MonteCarloTrainingBatch, TemporalDifferenceTrainingData
from RLC.real_chess.environment import ChessEnv, moves_mirror


class ReinforcementLearning(object):

    def __init__(self, env, agent, search_time=1, memsize=512, batch_size=512):
        """
        Chess algorithm that combines bootstrapped monte carlo tree search with Q Learning
        Args:
            env: RLC chess environment
            agent: RLC chess agent
            search_time: maximum time spent doing tree search
            memsize: Amount of training samples to keep in-memory
            batch_size: Size of the training batches
        """
        self.env: ChessEnv = env
        self.agent = agent  # The agent that's learning
        self.fixed_agent = type(agent)()  # The agent that's bootstrapped from
        self.fixed_agent.load_state_dict(self.agent.state_dict())
        self.batch_size = batch_size  # Batch size for TD Learning
        self.search_time = search_time  # How long to do MCTS
        self.min_sim_count = 8  # Minimal amount of playouts in MCTS
        self.best_so_far = 0.001  # Best test result so far
        self.maxiter = 20
        self.td_memory = TemporalDifferenceMemory(episodes=[])
        self.td_training_data = TemporalDifferenceTrainingData(samples=[])
        self.mc_training_data = MonteCarloTrainingData(samples=[])

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
            if k % c == 0 and (test_at_zero or k > 100):
                self.build_td_training_samples()
                td_training_batches = self.get_td_training_batches()
                mc_training_batches = self.get_mc_training_batches()

                for mc_training_batch in mc_training_batches:
                    self.agent.mc_update(self.fixed_agent, mc_training_batch)
                self.test(k)
                for td_training_batch in td_training_batches:
                    self.agent.td_update(self.fixed_agent, td_training_batch)
                self.test(k+1)

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
        logging.info("---testing new agent------")
        results = []
        testsize = 25
        for _ in range(testsize):
            results.append(self.test_game(
                self.agent, self.fixed_agent, random=-1))
        for _ in range(testsize):
            results.append(self.test_game(
                self.fixed_agent, self.agent, random=1))
        end_result = pd.DataFrame(results)
        end_result.columns = ['Return']
        end_result['Return'] = end_result['Return'].apply(float)
        end_result['color'] = ['white'] * testsize + ['black'] * testsize
        mean_return = end_result['Return'].mean() / self.min_sim_count
        logging.info("Average Return newest agent: %s",
                     mean_return)
        logging.info("Best So Far: %s", self.best_so_far)

        end_result['iter'] = k
        self.record_end_result(end_result)

        if mean_return > self.best_so_far:
            logging.info("new highest score")
            self.fixed_agent.load_state_dict(self.agent.state_dict())
            self.best_so_far = mean_return
            self.save_agent(k='best')
            self.maxiter += 1
            self.min_sim_count += 1

        logging.info("---test completed------\n")

    def record_end_result(self, end_result):
        fn_existing = "end_result.csv"
        if fn_existing in os.listdir("."):
            existing = pd.read_csv(fn_existing)
            updated = pd.concat([end_result, existing], axis=0)
        else:
            updated = end_result
        updated.to_csv(fn_existing, index=False)

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

        episode_active = 1
        turncount = 0
        color = 1
        self.env.node = self.env.node.get_root()  # Initialize the game tree
        self.env.reset()
        maxiter = self.maxiter
        Returns = 0

        # Play a game of chess

        while episode_active > 0.:
            current_player = white if color == 1 else black

            if color == random:
                move = np.random.choice(
                    [x for x in self.env.board.generate_legal_moves()])
            else:
                move, _ = current_player.select_action(self.env, greedy=False)
            episode_active, reward = self.env.step(move)
            color = color * -1
            Returns += reward

            gc.collect()

            turncount += 1
            if turncount >= maxiter and episode_active:
                episode_active = 0

        if self.agent == black:
            Returns *= -1

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
        # state, action, reward, actionspace
        logging.info("Practicing Chess through self-play iteration %s", k)
        episode_memory = TemporalDifferenceEpisode(steps=[])
        episode_active = torch.tensor(1.)
        turncount = 0
        color = 1
        self.env.node = self.env.node.get_root()  # Initialize the game tree
        self.env.reset()
        # Play a game of chess

        while episode_active > 0.:

            # Do a Monte Carlo Tree Search after game iteration k
            self.mcts(color)
            max_move = None
            max_value = np.NINF
            for move, child in self.env.node.children.items():
                sampled_value = torch.tensor(child.values).mean() * color
                if sampled_value > max_value:
                    max_value = sampled_value
                    max_move = move

            # Execute the best move
            layer_board = self.env.layer_board.clone()
            action_space = self.env.project_legal_moves().clone()

            episode_active, reward = self.env.step(max_move)
            self.env.node = self.env.node.children[max_move]

            td_step = TemporalDifferenceStep(
                layer_board=layer_board, reward=reward, action_space=action_space, color=torch.tensor(color), move_to=move.to_square, episode_active=episode_active)

            episode_memory.steps.append(td_step)

            color = color * -1

            gc.collect()

            turncount += 1
            if turncount >= maxiter and episode_active:
                episode_active = torch.tensor(0.)

        self.td_memory.episodes.append(episode_memory)

        return self.env.board

    def build_td_training_samples(self, shuffle=True):
        """
        Take the last step of each episode
        """

        td_episodes = self.td_memory.episodes

        for episode in td_episodes:
            if len(episode.steps) < 2:
                continue
            successor_step = None
            for step in episode.steps[::-1]:
                if successor_step == None:
                    if step.episode_active < 1.:
                        td_training_sample = TemporalDifferenceTrainingSample(layer_board=step.layer_board,
                                                                              move_to=step.move_to,
                                                                              reward=step.reward,
                                                                              action_space=step.action_space,
                                                                              episode_active=step.episode_active,
                                                                              color=step.color,
                                                                              successor_layer_board=step.layer_board,  # Dummy value
                                                                              successor_action_space=step.action_space,  # Dummy value
                                                                              successor_move_to=step.move_to  # Dummy value
                                                                              )
                    else:
                        successor_step = step
                        continue
                else:
                    td_training_sample = TemporalDifferenceTrainingSample(layer_board=step.layer_board,
                                                                          move_to=step.move_to,
                                                                          reward=step.reward,
                                                                          color=step.color,
                                                                          episode_active=step.episode_active,
                                                                          action_space=step.action_space,
                                                                          successor_layer_board=successor_step.layer_board,
                                                                          successor_move_to=successor_step.move_to,
                                                                          successor_action_space=successor_step.action_space
                                                                          )

                td_training_sample = self.make_color_invariant_td_sample(
                    td_training_sample)

                self.td_training_data.samples.append(td_training_sample)

                successor_step = step

        del self.td_memory
        self.td_memory = TemporalDifferenceMemory(episodes=[])

    def get_td_training_batches(self, frac=0.1, batch_size=32):
        total_n_samples = len(self.td_training_data.samples)
        train_n_samples = int(total_n_samples*frac)
        train_on_indices = np.random.choice(
            range(total_n_samples-train_n_samples), size=train_n_samples)

        # Cut train_on_indices in batches
        batches = [train_on_indices[i:i+batch_size]
                   for i in range(0, len(train_on_indices), batch_size)][:-1]

        batches_out = []

        for batch in batches:
            layer_boards = []
            moves_to = []
            rewards = []
            action_spaces = []
            colors = []
            successor_layer_boards = []
            successor_action_spaces = []
            successor_moves_to = []
            episode_actives = []
            for index in batch:
                td_training_sample = self.td_training_data.samples.pop(index)

                layer_boards.append(td_training_sample.layer_board)
                moves_to.append(td_training_sample.move_to)
                rewards.append(td_training_sample.reward)
                action_spaces.append(td_training_sample.action_space)
                colors.append(td_training_sample.color)
                successor_layer_boards.append(
                    td_training_sample.successor_layer_board)
                successor_action_spaces.append(
                    td_training_sample.successor_action_space)
                successor_moves_to.append(td_training_sample.successor_move_to)
                episode_actives.append(td_training_sample.episode_active)

            batch = TemporalDifferenceTrainingBatch(layer_boards=torch.cat(layer_boards, dim=0),
                                                    moves_to=torch.LongTensor(
                                                        moves_to).unsqueeze(dim=1),
                                                    rewards=torch.tensor(
                                                        rewards).unsqueeze(dim=1),
                                                    action_spaces=torch.cat(
                                                        action_spaces, dim=0),
                                                    colors=torch.tensor(
                                                        colors).unsqueeze(dim=1),
                                                    episode_actives=torch.tensor(
                                                        episode_actives).unsqueeze(dim=1),
                                                    successor_layer_boards=torch.cat(
                                                        successor_layer_boards, dim=0),
                                                    successor_moves_to=torch.LongTensor(
                                                        successor_moves_to).unsqueeze(dim=1),
                                                    successor_action_spaces=torch.cat(
                                                        successor_action_spaces, dim=0)
                                                    )
            batches_out.append(batch)
        return batches_out

    def get_mc_training_batches(self, batch_size=64, frac=0.1):
        total_n_samples = len(self.mc_training_data.samples)
        train_n_samples = int(total_n_samples*frac)
        train_on_indices = np.random.choice(
            range(total_n_samples-train_n_samples), size=train_n_samples)

        # Cut train_on_indices in batches
        batches = [train_on_indices[i:i+batch_size]
                   for i in range(0, len(train_on_indices), batch_size)][:-1]
        batches_out = []

        for batch in batches:
            layer_boards = []
            moves_to = []
            Returns = []
            action_spaces = []
            colors = []
            for i in batch:
                sample = self.mc_training_data.samples.pop(i)
                sample = self.make_color_invariant_mc_sample(sample)
                layer_boards.append(sample.layer_board)
                moves_to.append(sample.move_to)
                Returns.append(sample.returns)
                action_spaces.append(sample.action_space)
                colors.append(sample.color)

            layer_boards = torch.cat(layer_boards, dim=0)
            Returns = torch.tensor(Returns).unsqueeze(dim=1)
            action_spaces = torch.cat(action_spaces, dim=0)
            colors = torch.tensor(colors).unsqueeze(dim=1)
            moves_to = torch.LongTensor(moves_to).unsqueeze(dim=1)
            batches_out.append(MonteCarloTrainingBatch(
                layer_boards=layer_boards, returns=Returns, action_spaces=action_spaces, moves_to=moves_to, colors=colors))

        return batches_out

    @staticmethod
    def flip_layer_board(layer_board: torch.Tensor):
        layer_board = layer_board.flip(dims=(2,)).clone()
        layer_board = layer_board * torch.tensor(-1.)
        return layer_board

    @staticmethod
    def flip_actionspace(action_space: torch.Tensor):
        action_space = action_space.view(-1, 8,
                                         8).flip(dims=(1,)).view(-1, 64).clone()
        return action_space

    def make_color_invariant_mc_sample(self, sample: MonteCarloTrainingSample):
        if sample.color < 0:
            sample.layer_board = self.flip_layer_board(sample.layer_board)
            sample.returns = sample.returns * sample.color
            sample.action_space = self.flip_actionspace(sample.action_space)
            sample.move_to = moves_mirror[sample.move_to]
        return sample

    def make_color_invariant_td_sample(self, sample: TemporalDifferenceTrainingSample):
        if sample.color < 0:
            sample.layer_board = self.flip_layer_board(sample.layer_board)
            sample.reward = sample.reward * sample.color
            sample.action_space = self.flip_actionspace(sample.action_space)
            sample.move_to = moves_mirror[sample.move_to]
        else:
            sample.successor_layer_board = self.flip_layer_board(
                sample.layer_board)
            sample.successor_action_space = self.flip_actionspace(
                sample.successor_action_space)
            sample.successor_move_to = moves_mirror[sample.move_to]
        return sample

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
            episode_active = torch.tensor(1.)
            node_rewards = []
            color = starting_color

            # 1. Select the best node from where to start MCTS
            while self.env.node.children:
                legal_moves = self.env.board.generate_legal_moves()
                q_values = self.fixed_agent.get_q_values(self.env)
                self.env.node, move = self.env.node.select(
                    color=color, legal_moves=legal_moves, q_values=q_values)
                depth += 1
                color = color * -1  # switch color
                # Update the environment to reflect the node
                episode_active, reward = self.env.step(move)
                node_rewards.append(reward)

            # 2. Expand the game tree with a new move
            if episode_active > 0.:
                move = None
                loop_max = 20
                loop = 0
                while move in self.env.node.children.keys() or not move:
                    move, _ = self.fixed_agent.select_action(self.env)
                    loop += 1
                    if loop > loop_max:
                        break
                episode_active, reward = self.env.step(move)
                if move not in self.env.node.children.keys():
                    self.env.node.add_child(move)
                self.env.node = self.env.node.get_down(move)
                depth += 1
                color = color * -1
                node_rewards.append(reward)

            # 3. Monte Carlo Simulation to estimate leaf node value
            if episode_active > 0.:  # if the game is active
                Returns, move = self.env.node.simulate(self.fixed_agent,
                                                       self.env,
                                                       depth=0)

                color = torch.tensor(
                    1.) if self.env.board.turn else torch.tensor(-1.)

                mc_sample = MonteCarloTrainingSample(
                    layer_board=self.env.layer_board, returns=Returns, action_space=self.env.project_legal_moves(),
                    color=color, move_to=move.to_square)

                self.mc_training_data.samples.append(mc_sample)
            else:
                # episode is over, no future returns
                Returns = torch.tensor(0.)

            # 4. Backpropagate Returns
            self.env.node.update(Returns)
            while depth > 0:
                self.env.reverse()
                self.env.node = self.env.node.parent
                latest_reward = node_rewards.pop(-1)
                Returns = latest_reward + GAMMA * Returns
                self.env.node.update(Returns)

                depth -= 1

            sim_count += 1

            board_out = self.env.board.fen()
            assert board_in == board_out
