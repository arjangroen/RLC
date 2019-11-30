import numpy as np
import pprint


class Reinforce(object):

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env

    def play_episode(self, state, max_steps=1e3, epsilon=0.1):
        """
        Play an episode of move chess
        :param state: tuple describing the starting state on 8x8 matrix
        :param max_steps: integer, maximum amount of steps before terminating the episode
        :param epsilon: exploration parameter
        :return: tuple of lists describing states, actions and rewards in a episode
        """
        self.env.state = state
        states = []
        actions = []
        rewards = []
        episode_end = False

        # Play out an episode
        count_steps = 0
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon)  # get the index of the next action
            action = self.agent.action_space[action_index]
            actions.append(action_index)
            reward, episode_end = self.env.step(action)
            state = self.env.state
            rewards.append(reward)

            #  avoid infinite loops
            if count_steps > max_steps:
                episode_end = True

        return states, actions, rewards

    def sarsa_td(self, n_episodes=1000, alpha=0.01, gamma=0.9):
        """
        Run the sarsa control algorithm (TD0), finding the optimal policy and action function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :return: finds the optimal policy for move chess
        """
        for k in range(n_episodes):
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (1 + k), 0.05)
            while not episode_end:
                state = self.env.state
                action_index = self.agent.apply_policy(state, epsilon)
                action = self.agent.action_space[action_index]
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                successor_action_value = self.agent.action_function[successor_state[0],
                                                                    successor_state[1], successor_action_index]

                q_update = alpha * (reward + gamma * successor_action_value - action_value)

                self.agent.action_function[state[0], state[1], action_index] += q_update
                self.agent.policy = self.agent.action_function.copy()

    def sarsa_lambda(self, n_episodes=1000, alpha=0.05, gamma=0.9, lamb=0.8):
        """
        Run the sarsa control algorithm (TD lambda), finding the optimal policy and action function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :param lamb: lambda parameter describing the decay over n-step returns
        :return: finds the optimal move chess policy
        """
        for k in range(n_episodes):
            self.agent.E = np.zeros(shape=self.agent.action_function.shape)
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (1 + k), 0.2)
            action_index = self.agent.apply_policy(state, epsilon)
            action = self.agent.action_space[action_index]
            while not episode_end:
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, epsilon)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                if not episode_end:
                    successor_action_value = self.agent.action_function[successor_state[0],
                                                                        successor_state[1], successor_action_index]
                else:
                    successor_action_value = 0
                delta = reward + gamma * successor_action_value - action_value
                self.agent.E[state[0], state[1], action_index] += 1
                self.agent.action_function = self.agent.action_function + alpha * delta * self.agent.E
                self.agent.E = gamma * lamb * self.agent.E
                state = successor_state
                action = self.agent.action_space[successor_action_index]
                action_index = successor_action_index
                self.agent.policy = self.agent.action_function.copy()

    def q_learning(self, n_episodes=1000, alpha=0.05, gamma=0.9):
        """
        Run Q-learning (also known as sarsa-max, finding the optimal policy and value function
        :param n_episodes: int, amount of episodes to train
        :param alpha: learning rate
        :param gamma: discount factor of future rewards
        :return: finds the optimal move chess policy
        """
        for k in range(n_episodes):
            state = (0, 0)
            self.env.state = state
            episode_end = False
            epsilon = max(1 / (k + 1), 0.1)
            while not episode_end:
                action_index = self.agent.apply_policy(state, epsilon)
                action = self.agent.action_space[action_index]
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.agent.apply_policy(successor_state, -1)

                action_value = self.agent.action_function[state[0], state[1], action_index]
                if not episode_end:
                    successor_action_value = self.agent.action_function[successor_state[0],
                                                                        successor_state[1], successor_action_index]
                else:
                    successor_action_value = 0

                av_new = self.agent.action_function[state[0], state[1], action_index] + alpha * (reward +
                                                                                                 gamma *
                                                                                                 successor_action_value
                                                                                                 - action_value)
                self.agent.action_function[state[0], state[1], action_index] = av_new
                self.agent.policy = self.agent.action_function.copy()
                state = successor_state

    def monte_carlo_learning(self, epsilon=0.1):
        """
        Learn move chess through monte carlo control
        :param epsilon: exploration rate
        :return:
        """
        state = (0, 0)
        self.env.state = state

        # Play out an episode
        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        first_visits = []
        for idx, state in enumerate(states):
            action_index = actions[idx]
            if (state, action_index) in first_visits:
                continue
            r = np.sum(rewards[idx:])
            if (state, action_index) in self.agent.Returns.keys():
                self.agent.Returns[(state, action_index)].append(r)
            else:
                self.agent.Returns[(state, action_index)] = [r]
            self.agent.action_function[state[0], state[1], action_index] = \
                np.mean(self.agent.Returns[(state, action_index)])
            first_visits.append((state, action_index))
        # Update the policy. In Monte Carlo Control, this is greedy behavior with respect to the action function
        self.agent.policy = self.agent.action_function.copy()

    def monte_carlo_evaluation(self, epsilon=0.1, first_visit=True):
        """
        Find the value function of states using MC evaluation
        :param epsilon: exploration rate
        :param first_visit: Boolean, count only from first occurence of state
        :return:
        """
        state = (0, 0)
        self.env.state = state
        states, actions, rewards = self.play_episode(state, epsilon=epsilon)

        visited_states = set()
        for idx, state in enumerate(states):
            if state not in visited_states and first_visit:
                self.agent.N[state[0], state[1]] += 1
                n = self.agent.N[state[0], state[1]]
                forward_rewards = np.sum(rewards[idx:])
                expected_rewards = self.agent.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.agent.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif not first_visit:
                self.agent.N[state[0], state[1]] += 1
                n = self.agent.N[state[0], state[1]]
                forward_rewards = np.sum(rewards[idx:])
                expected_rewards = self.agent.value_function[state[0], state[1]]
                delta = forward_rewards - expected_rewards
                self.agent.value_function[state[0], state[1]] += ((1 / n) * delta)
                visited_states.add(state)
            elif state in visited_states and first_visit:
                continue

    def TD_zero(self, epsilon=0.1, alpha=0.05, max_steps=1000, lamb=0.9):
        """
        Find the value function of move chess states
        :param epsilon: exploration rate
        :param alpha: learning rate
        :param max_steps: max amount of steps in an episode
        """
        state = (0, 0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        episode_end = False
        count_steps = 0
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon=epsilon)
            action = self.agent.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            self.agent.value_function[state[0], state[1]] = self.agent.value_function[state[0], state[1]] + alpha * (
                    reward + lamb * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[
                state[0], state[1]])
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True

    def TD_lambda(self, epsilon=0.1, alpha=0.05, gamma=0.9, max_steps=1000, lamb=0.9):
        self.agent.E = np.zeros(self.agent.value_function.shape)
        state = (0, 0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        count_steps = 0
        episode_end = False
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.agent.apply_policy(state, epsilon=epsilon)
            action = self.agent.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            delta = reward + lamb * self.agent.value_function[suc_state[0], suc_state[1]] - self.agent.value_function[
                state[0],
                state[1]]
            self.agent.E[state[0], state[1]] += 1

            # Note to self: vectorize code below.
            self.agent.value_function = self.agent.value_function + alpha * delta * self.E
            self.agent.E = gamma * lamb * self.agent.E
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True

    def evaluate_state(self, state, gamma=0.9, synchronous=True):
        """
        Calculates the value of a state based on the successor states and the immediate rewards.
        Args:
            state: tuple of 2 integers 0-7 representing the state
            gamma: float, discount factor
            synchronous: Boolean

        Returns: The expected value of the state under the current policy.

        """
        greedy_action_value = np.max(self.agent.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.agent.policy[state[0], state[1], :]) if
                          a == greedy_action_value]  # List of all greedy actions
        prob = 1 / len(greedy_indices)  # probability of an action occuring
        state_value = 0
        for i in greedy_indices:
            self.env.state = state  # reset state to the one being evaluated
            reward, episode_end = self.env.step(self.agent.action_space[i])
            if synchronous:
                successor_state_value = self.agent.value_function_prev[self.env.state]
            else:
                successor_state_value = self.agent.value_function[self.env.state]
            state_value += (prob * (
                    reward + gamma * successor_state_value))  # sum up rewards and discounted successor state value
        return state_value

    def evaluate_policy(self, gamma=0.9, synchronous=True):
        self.agent.value_function_prev = self.agent.value_function.copy()  # For synchronous updates
        for row in range(self.agent.value_function.shape[0]):
            for col in range(self.agent.value_function.shape[1]):
                self.agent.value_function[row, col] = self.evaluate_state((row, col), gamma=gamma,
                                                                          synchronous=synchronous)

    def improve_policy(self):
        """
        Finds the greedy policy w.r.t. the current value function
        """

        self.agent.policy_prev = self.agent.policy.copy()
        for row in range(self.agent.action_function.shape[0]):
            for col in range(self.agent.action_function.shape[1]):
                for action in range(self.agent.action_function.shape[2]):
                    self.env.state = (row, col)  # reset state to the one being evaluated
                    reward, episode_end = self.env.step(self.agent.action_space[action])
                    successor_state_value = 0 if episode_end else self.agent.value_function[self.env.state]
                    self.agent.policy[row, col, action] = reward + successor_state_value

                max_policy_value = np.max(self.agent.policy[row, col, :])
                max_indices = [i for i, a in enumerate(self.agent.policy[row, col, :]) if a == max_policy_value]
                for idx in max_indices:
                    self.agent.policy[row, col, idx] = 1

    def policy_iteration(self, eps=0.1, gamma=0.9, iteration=1, k=32, synchronous=True):
        """
        Finds the optimal policy
        Args:
            eps: float, exploration rate
            gamma: float, discount factor
            iteration: the iteration number
            k: (int) maximum amount of policy evaluation iterations
            synchronous: (Boolean) whether to use synchronous are asynchronous back-ups 

        Returns:

        """
        policy_stable = True
        print("\n\n______iteration:", iteration, "______")
        print("\n policy:")
        self.visualize_policy()

        print("")
        value_delta_max = 0
        for _ in range(k):
            self.evaluate_policy(gamma=gamma, synchronous=synchronous)
            value_delta = np.max(np.abs(self.agent.value_function_prev - self.agent.value_function))
            value_delta_max = value_delta
            if value_delta_max < eps:
                break
        print("Value function for this policy:")
        print(self.agent.value_function.round().astype(int))
        action_function_prev = self.agent.action_function.copy()
        print("\n Improving policy:")
        self.improve_policy()
        policy_stable = self.agent.compare_policies() < 1
        print("policy diff:", policy_stable)

        if not policy_stable and iteration < 1000:
            iteration += 1
            self.policy_iteration(iteration=iteration)
        elif policy_stable:
            print("Optimal policy found in", iteration, "steps of policy evaluation")
        else:
            print("failed to converge.")

    def visualize_policy(self):
        """
        Gives you are very ugly visualization of the policy
        Returns: None

        """
        greedy_policy = self.agent.policy.argmax(axis=2)
        policy_visualization = {}
        if self.agent.piece == 'king':
            arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'knight':
            arrows = "↑↗ ↗→ →↘ ↓↘ ↙↓ ←↙ ←↖ ↖↑"
            visual_row = ["[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]", "[  ]"]
        elif self.agent.piece == 'bishop':
            arrows = "↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖ ↗ ↘ ↙ ↖"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        elif self.agent.piece == 'rook':
            arrows = "↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ← ↑ → ↓ ←"
            visual_row = ["[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]", "[ ]"]
        arrowlist = arrows.split(" ")
        for idx, arrow in enumerate(arrowlist):
            policy_visualization[idx] = arrow
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                idx = greedy_policy[row, col]

                visual_board[row][col] = policy_visualization[idx]

        visual_board[self.env.terminal_state[0]][self.env.terminal_state[1]] = "F"
        pprint.pprint(visual_board)
