from RLC.capture_chess.agent import Agent
from RLC.capture_chess.environment import Board
import numpy as np
from chess.pgn import Game
import pandas as pd


class Q_learning(object):

    def __init__(self, agent, env, memsize=1000):
        """
        Reinforce object to learn capture chess
        Args:
            agent: The agent playing the chess game as white
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.agent = agent
        self.env = env
        self.memory = []
        self.memsize = memsize
        self.reward_trace = []
        self.memory = []
        self.sampling_probs = []

    def learn(self, iters=100, c=10):
        """
        Run the Q-learning algorithm. Play greedy on the final iter
        Args:
            iters: int
                amount of games to train
            c: int
                update the network every c games

        Returns: pgn (str)
            pgn string describing final game

        """
        for k in range(iters):
            if k % c == 0:
                print("iter", k)
                self.agent.fix_model()
            greedy = True if k == iters - 1 else False
            self.env.reset()
            self.play_game(k, greedy=greedy)

        pgn = Game.from_board(self.env.board)
        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return pgn

    def play_game(self, k, greedy=False, maxiter=25):
        """
        Play a game of capture chess
        Args:
            k: int
                game count, determines epsilon (exploration rate)
            greedy: Boolean
                if greedy, no exploration is done
            maxiter: int
                Maximum amount of steps per game

        Returns:

        """
        episode_end = False
        turncount = 0

        # Here we determine the exploration rate. k is divided by 250 to slow down the exploration rate decay.
        eps = max(0.05, 1 / (1 + (k / 250))) if not greedy else 0.

        # Play a game of chess
        while not episode_end:
            state = self.env.layer_board
            explore = np.random.uniform(0, 1) < eps  # determine whether to explore
            if explore:
                move = self.env.get_random_action()
                move_from = move.from_square
                move_to = move.to_square
            else:
                action_values = self.agent.get_action_values(np.expand_dims(state, axis=0))
                action_values = np.reshape(np.squeeze(action_values), (64, 64))
                action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
                action_values = np.multiply(action_values, action_space)
                move_from = np.argmax(action_values, axis=None) // 64
                move_to = np.argmax(action_values, axis=None) % 64
                moves = [x for x in self.env.board.generate_legal_moves() if \
                         x.from_square == move_from and x.to_square == move_to]
                if len(moves) == 0:  # If all legal moves have negative action value, explore.
                    move = self.env.get_random_action()
                    move_from = move.from_square
                    move_to = move.to_square
                else:
                    move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board
            if len(self.memory) > self.memsize:
                self.memory.pop(0)
                self.sampling_probs.pop(0)
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0
            if episode_end:
                new_state = new_state * 0
            self.memory.append([state, (move_from, move_to), reward, new_state])
            self.sampling_probs.append(1)

            self.reward_trace.append(reward)

            self.update_agent(turncount)

        return self.env.board

    def sample_memory(self, turncount):
        """
        Get a sample from memory for experience replay
        Args:
            turncount: int
                turncount limits the size of the minibatch

        Returns: tuple
            a mini-batch of experiences (list)
            indices of chosen experiences

        """
        minibatch = []
        memory = self.memory[:-turncount]
        probs = self.sampling_probs[:-turncount]
        sample_probs = [probs[n] / np.sum(probs) for n in range(len(probs))]
        indices = np.random.choice(range(len(memory)), min(1028, len(memory)), replace=True, p=sample_probs)
        for i in indices:
            minibatch.append(memory[i])

        return minibatch, indices

    def update_agent(self, turncount):
        """
        Update the agent using experience replay. Set the sampling probs with the td error
        Args:
            turncount: int
                Amount of turns played. Only sample the memory of there are sufficient samples
        Returns:

        """
        if turncount < len(self.memory):
            minibatch, indices = self.sample_memory(turncount)
            td_errors = self.agent.network_update(minibatch)
            for n, i in enumerate(indices):
                self.sampling_probs[i] = np.abs(td_errors[n])


class Reinforce(object):

    def __init__(self, agent, env):
        """
        Reinforce object to learn capture chess
        Args:
            agent: The agent playing the chess game as white
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.agent = agent
        self.env = env
        self.reward_trace = []
        self.action_value_mem = []

    def learn(self, iters=100, c=10):
        """
        Run the Q-learning algorithm. Play greedy on the final iter
        Args:
            iters: int
                amount of games to train
            c: int
                update the network every c games

        Returns: pgn (str)
            pgn string describing final game

        """
        for k in range(iters):
            self.env.reset()
            states, actions, rewards, action_spaces = self.play_game(k)
            self.reinforce_agent(states, actions, rewards, action_spaces)

        pgn = Game.from_board(self.env.board)
        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return pgn

    def play_game(self, k, maxiter=25):
        """
        Play a game of capture chess
        Args:
            k: int
                game count, determines epsilon (exploration rate)
            greedy: Boolean
                if greedy, no exploration is done
            maxiter: int
                Maximum amount of steps per game

        Returns:

        """
        episode_end = False
        turncount = 0

        states = []
        actions = []
        rewards = []
        action_spaces = []

        # Play a game of chess
        while not episode_end:
            state = self.env.layer_board
            action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
            action_probs = self.agent.model.predict([np.expand_dims(state, axis=0),
                                                     np.zeros((1, 1)),
                                                     action_space.reshape(1, 4096)])
            self.action_value_mem.append(action_probs)
            action_probs = action_probs / action_probs.sum()
            move = np.random.choice(range(4096), p=np.squeeze(action_probs))
            move_from = move // 64
            move_to = move % 64
            moves = [x for x in self.env.board.generate_legal_moves() if \
                     x.from_square == move_from and x.to_square == move_to]
            assert len(moves) > 0  # should not be possible
            if len(moves) > 1:
                move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.
            elif len(moves) == 1:
                move = moves[0]

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0
            if episode_end:
                new_state = new_state * 0

            states.append(state)
            actions.append((move_from, move_to))
            rewards.append(reward)
            action_spaces.append(action_space.reshape(1, 4096))

        self.reward_trace.append(np.sum(rewards))

        return states, actions, rewards, action_spaces

    def reinforce_agent(self, states, actions, rewards, action_spaces):
        """
        Update the agent using experience replay. Set the sampling probs with the td error
        Args:
            turncount: int
                Amount of turns played. Only sample the memory of there are sufficient samples
        Returns:

        """
        self.agent.policy_gradient_update(states, actions, rewards, action_spaces)


class ActorCritic(object):

    def __init__(self, actor, critic, env):
        """
        ActorCritic object to learn capture chess
        Args:
            actor: Policy Gradient Agent
            critic: Q-learning Agent
            env: The environment including the python-chess board
            memsize: maximum amount of games to retain in-memory
        """
        self.actor = actor
        self.critic = critic
        self.env = env
        self.reward_trace = []
        self.action_value_mem = []
        self.memory = []
        self.sampling_probs = []

    def learn(self, iters=100, c=10):
        """
        Run the Q-learning algorithm. Play greedy on the final iter
        Args:
            iters: int
                amount of games to train
            c: int
                update the network every c games

        Returns: pgn (str)
            pgn string describing final game

        """
        for k in range(iters):
            if k % c == 0:
                self.critic.fix_model()
            self.env.reset()
            end_state = self.play_game(k)

        pgn = Game.from_board(self.env.board)
        reward_smooth = pd.DataFrame(self.reward_trace)
        reward_smooth.rolling(window=10, min_periods=0).mean().plot()

        return pgn

    def play_game(self, k, greedy=False, maxiter=25):
        """
        Play a game of capture chess
        Args:
            k: int
                game count, determines epsilon (exploration rate)
            greedy: Boolean
                if greedy, no exploration is done
            maxiter: int
                Maximum amount of steps per game

        Returns:

        """
        episode_end = False
        turncount = 0

        # Play a game of chess
        state = self.env.layer_board
        while not episode_end:
            state = self.env.layer_board
            action_space = self.env.project_legal_moves()  # The environment determines which moves are legal
            action_probs = self.actor.model.predict([np.expand_dims(state, axis=0),
                                                     np.zeros((1, 1)),
                                                     action_space.reshape(1, 4096)])
            self.action_value_mem.append(action_probs)
            # print(action_probs)
            # print(np.max(action_probs))
            action_probs = action_probs / action_probs.sum()
            move = np.random.choice(range(4096), p=np.squeeze(action_probs))
            move_from = move // 64
            move_to = move % 64
            moves = [x for x in self.env.board.generate_legal_moves() if \
                     x.from_square == move_from and x.to_square == move_to]
            assert len(moves) > 0  # should not be possible
            if len(moves) > 1:
                move = np.random.choice(moves)  # If there are multiple max-moves, pick a random one.
            elif len(moves) == 1:
                move = moves[0]

            episode_end, reward = self.env.step(move)
            new_state = self.env.layer_board
            turncount += 1
            if turncount > maxiter:
                episode_end = True
                reward = 0
            if episode_end:
                new_state = new_state * 0

            self.memory.append([state, (move_from, move_to), reward, new_state, action_space.reshape(1, 4096)])
            self.sampling_probs.append(1)
            self.reward_trace.append(reward)

        self.update_actorcritic(turncount)

        return self.env.board

    def sample_memory(self, turncount):
        """
        Get a sample from memory for experience replay
        Args:
            turncount: int
                turncount limits the size of the minibatch

        Returns: tuple
            a mini-batch of experiences (list)
            indices of chosen experiences

        """
        minibatch = []
        memory = self.memory[:-turncount]
        probs = self.sampling_probs[:-turncount]
        sample_probs = [probs[n] / np.sum(probs) for n in range(len(probs))]
        indices = np.random.choice(range(len(memory)), min(1028, len(memory)), replace=False, p=sample_probs)
        for i in indices:
            minibatch.append(memory[i])

        return minibatch, indices

    def update_actorcritic(self, turncount):
        """Actor critic"""

        if turncount < len(self.memory):

            # Get a sampple
            minibatch, indices = self.sample_memory(turncount)

            # Update critic and find td errors for prioritized experience replay
            td_errors = self.critic.network_update(minibatch)

            # Get a Q value from the critic
            states = [x[0] for x in minibatch]
            actions = [x[1] for x in minibatch]
            Q_est = self.critic.get_action_values(np.stack(states, axis=0))
            action_spaces = [x[4] for x in minibatch]

            self.actor.policy_gradient_update(states, actions, Q_est, action_spaces, actor_critic=True)

            # Update sampling probs
            for n, i in enumerate(indices):
                self.sampling_probs[i] = np.abs(td_errors[n])

    def update_critic(self, turncount):
        """
        Update the agent using experience replay. Set the sampling probs with the td error
        Args:
            turncount: int
                Amount of turns played. Only sample the memory of there are sufficient samples
        Returns:

        """
        if turncount < len(self.memory):
            minibatch, indices = self.sample_memory(turncount)
            td_errors = self.critic.network_update(minibatch)

            for n, i in enumerate(indices):
                self.sampling_probs[i] = np.abs(td_errors[n])
