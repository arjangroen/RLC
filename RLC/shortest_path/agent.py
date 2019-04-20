import numpy as np
import pprint


class Piece(object):

    def __init__(self, env, piece='king', k_max=32, synchronous=True, lamb=0.9):
        self.env = env
        self.piece = piece
        self.k_max = k_max
        self.synchronous = synchronous
        self.lamb = lamb
        self.init_actionspace()
        self.value_function = np.zeros(shape=env.reward_space.shape)
        self.N = np.zeros(self.value_function.shape)
        self.action_function = np.zeros(shape=(env.reward_space.shape[0],
                                               env.reward_space.shape[1],
                                               len(self.action_space)))

    def init_actionspace(self):
        assert self.piece in ["king", "rook", "bishop",
                              "knight"], f"{self.piece} is not a supported piece try another one"
        if self.piece == 'king':
            self.action_space = [(-1, 0),  # north
                                 (-1, 1),  # north-west
                                 (0, 1),  # west
                                 (1, 1),  # south-west
                                 (1, 0),  # south
                                 (1, -1),  # south-east
                                 (0, -1),  # east
                                 (-1, -1),  # north-east
                                 ]
        elif self.piece == 'rook':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, 0))  # north
                self.action_space.append((0, amplitude))  # west
                self.action_space.append((amplitude, 0))  # south
                self.action_space.append((0, -amplitude))  # east
        elif self.piece == 'knight':
            self.action_space = [(-2, 1),  # north-north-west
                                 (-1, 2),  # n-w-w
                                 (1, 2),  # s-w-w
                                 (2, 1),  # s-s-w
                                 (2, -1),  # s-s-e
                                 (1, -2),  # s-e-e
                                 (-1, -2),  # n-e-e
                                 (-2, -1)]  # n-n-e
        elif self.piece == 'bishop':
            self.action_space = []
            for amplitude in range(1, 8):
                self.action_space.append((-amplitude, amplitude))  # north-west
                self.action_space.append((amplitude, amplitude))  # south-west
                self.action_space.append((amplitude, -amplitude))  # south-east
                self.action_space.append((-amplitude, -amplitude))  # north-east

    def evaluate_policy(self):
        self.value_function_old = self.value_function.copy()  # For synchronous updates
        for row in range(self.value_function.shape[0]):
            for col in range(self.value_function.shape[1]):
                self.value_function[row, col] = self.evaluate_state((row, col))

    def monte_carlo_iteration(self):
        pass

    def monte_carlo_evaluation(self, first_visit=True):
        state = (np.random.randint(0, 8), np.random.randint(0, 8))
        self.env.state = state
        self.ensure_reachable_state()
        states = []
        actions = []
        rewards = []
        episode_end = False

        # Play out an episode
        while not episode_end:
            states.append(state)
            max_action_value = np.max(self.action_function[state[0], state[1], :])
            max_indices = [i for i, a in enumerate(self.action_function[state[0], state[1], :]) if
                           a == max_action_value]
            action_index = np.random.choice(max_indices)
            action = self.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            state = self.env.state
            rewards.append(reward)

        visited_states = set()
        for idx, state in enumerate(states):
            if state not in visited_states and first_visit:
                self.N[state[0], state[1]] += 1
                n = self.N[state[0], state[1]]
                forward_rewards = np.sum(rewards[idx:])
                expected_rewards = self.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif not first_visit:
                self.N[state[0], state[1]] += 1
                n = self.N[state[0], state[1]]
                forward_rewards = np.sum(rewards[idx:])
                expected_rewards = self.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif state in visited_states and first_visit:
                continue

    def temporal_difference_evaluation(self, n=1):
        state = (np.random.randint(0, 8), np.random.randint(0, 8))
        self.env.state = state
        self.ensure_reachable_state()
        states = []
        actions = []
        rewards = []
        episode_end = False
        while not episode_end:
            states.append(state)
            max_action_value = np.max(self.action_function[state[0], state[1], :])
            max_indices = [i for i, a in enumerate(self.action_function[state[0], state[1], :]) if
                           a == max_action_value]
            action_index = np.random.choice(max_indices)
            action = self.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            state = self.env.state
            rewards.append(reward)

            for idx in range(n):
                pass

    def TD_zero(self, alpha=0.05):
        state = (np.random.randint(0, 8), np.random.randint(0, 8))
        self.env.state = state
        self.ensure_reachable_state()
        states = []
        actions = []
        rewards = []
        episode_end = False
        while not episode_end:
            states.append(state)
            max_action_value = np.max(self.action_function[state[0], state[1], :])
            max_indices = [i for i, a in enumerate(self.action_function[state[0], state[1], :]) if
                           a == max_action_value]
            action_index = np.random.choice(max_indices)
            action = self.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            self.value_function[state[0], state[1]] = self.value_function[state[0], state[1]] + alpha * (
                    reward + self.lamb * self.value_function[suc_state[0], suc_state[1]] - self.value_function[state[0], state[1]])
            state = self.env.state

    def TD_lambda(self,alpha=0.05,gamma=0.9):
        self.E = np.zeros(self.value_function.shape)
        state = (np.random.randint(0, 8), np.random.randint(0, 8))
        self.env.state = state
        self.ensure_reachable_state()
        states = []
        actions = []
        rewards = []
        episode_end = False
        while not episode_end:
            states.append(state)
            max_action_value = np.max(self.action_function[state[0], state[1], :])
            max_indices = [i for i, a in enumerate(self.action_function[state[0], state[1], :]) if
                           a == max_action_value]
            action_index = np.random.choice(max_indices)
            action = self.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            delta = reward + self.lamb * self.value_function[suc_state[0], suc_state[1]] - self.value_function[state[0],
                                                                                                               state[1]]
            self.E[state[0],state[1]] += 1
            for row in range(self.value_function.shape[0]):
                for col in range(self.value_function.shape[1]):
                    self.value_function[row,col] = self.value_function[row,col] + alpha * delta * self.E[row,col]
                    self.E[row,col] = gamma * self.lamb * self.E[row, col]
            state = self.env.state

    def evaluate_state(self, state):
        action_values = self.action_function[state[0], state[1], :]
        max_action_value = np.max(action_values)
        max_indices = [i for i, a in enumerate(action_values) if a == max_action_value]
        prob = 1 / len(max_indices)
        state_value = 0
        for i in max_indices:
            self.env.state = state  # reset state to the one being evaluated
            reward, episode_end = self.env.step(self.action_space[i])
            if self.synchronous:
                successor_state_value = self.value_function_old[self.env.state]
            else:
                successor_state_value = self.value_function[self.env.state]
            state_value += (prob * (reward + self.lamb * successor_state_value))
        return state_value

    def improve_policy(self):
        for row in range(self.action_function.shape[0]):
            for col in range(self.action_function.shape[1]):
                for action in range(self.action_function.shape[2]):
                    self.env.state = (row, col)  # reset state to the one being evaluated
                    reward, episode_end = self.env.step(self.action_space[action])
                    successor_state_value = 0 if episode_end else self.value_function[self.env.state]
                    self.action_function[row, col, action] = reward + successor_state_value

    def policy_iteration(self, eps=0.1, iteration=1):
        policy_stable = True
        print("\n\n______iteration:", iteration, "______")
        print("\n policy:")
        self.visualize_policy()

        print("")
        value_delta_max = 0
        for k in range(self.k_max):
            self.evaluate_policy()
            value_delta = np.max(np.abs(self.value_function_old - self.value_function))
            value_delta_max = max(value_delta_max, value_delta)
            if value_delta_max < eps:
                break
        print("Value function for this policy:")
        print(self.value_function.astype(int))
        action_function_old = self.action_function.copy()
        print("\n Improving policy:")
        self.improve_policy()
        policy_delta = np.sum(np.abs(np.argmax(action_function_old, axis=2) - np.argmax(self.action_function, axis=2)))
        print("policy difference in improvement", policy_delta)
        print("________________________________")

        if policy_delta > 0 and iteration < 20:
            iteration += 1
            self.policy_iteration(iteration=iteration)
        elif policy_delta == 0:
            print("Optimal policy found in", iteration, "steps of policy evaluation")
        else:
            print("failed to converge.")

    def visualize_policy(self):
        greedy_policy = self.action_function.argmax(axis=2)
        policy_visualization = {}
        if self.piece == 'king':
            arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.piece == 'knight':
            arrows = "↑↗ ↗→ →↘ ↓↘ ↙↓ ←↙ ←↖ ↖↑"
            visual_row = ["[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]"]
        elif self.piece == 'bishop':
            arrows = "↗ ↗ ↗ ↗ ↗ ↗ ↗ ↘ ↘ ↘ ↘ ↘ ↘ ↘ ↙ ↙ ↙ ↙ ↙ ↙ ↙ ↖ ↖ ↖ ↖ ↖ ↖ ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.piece == 'rook':
            arrows = "↑ ↑ ↑ ↑ ↑ ↑ ↑ → → → → → → → ↓ ↓ ↓ ↓ ↓ ↓ ↓ ← ← ← ← ← ← ←"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        arrowlist = arrows.split(" ")
        for idx, arrow in enumerate(arrowlist):
            policy_visualization[idx] = arrow
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                visual_board[row][col] = policy_visualization[greedy_policy[row, col]]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "Q"
        pprint.pprint(visual_board)

    def ensure_reachable_state(self):
        if np.sum(self.env.state) % 2 == 1 and self.piece == 'bishop':
            print('moving terminal state to avoid endless MDP for bishop')
            self.env.terminal_state = (7, 6)
            print('new terminal state', self.env.terminal_state)
