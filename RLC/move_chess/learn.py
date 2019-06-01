

class Reinforce(object):

    def __init__(self,agent,env):
        self.agent = agent
        self.env = env


    def play_episode(self,state,max_steps=1e3,epsilon=0.1):
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
            action_index = self.apply_policy(state,epsilon)
            action = self.action_space[action_index]
            actions.append(action_index)
            reward, episode_end = self.env.step(action)
            state = self.env.state
            rewards.append(reward)

            #  avoid infinite loops
            if count_steps > max_steps:
                episode_end = True

        return states, actions, rewards

    def evaluate_policy(self):
        self.value_function_old = self.value_function.copy()  # For synchronous updates
        for row in range(self.value_function.shape[0]):
            for col in range(self.value_function.shape[1]):
                self.value_function[row, col] = self.evaluate_state((row, col))

    def sarsa_control(self,n_episodes=1e3,alpha=0.01,gamma=0.9):
        for k in range(n_episodes):
            state = (0,0)
            self.env.state = state
            episode_end = False
            epsilon = max(1/(1+k),0.05)
            while not episode_end:
                state = self.env.state
                action_index = self.apply_policy(state,epsilon)
                action = self.action_space[action_index]
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.apply_policy(successor_state,epsilon)

                action_value = self.action_function[state[0],state[1],action_index]
                successor_action_value = self.action_function[successor_state[0],
                                                              successor_state[1],successor_action_index]

                q_update = alpha * (reward + gamma * successor_action_value - action_value)

                self.action_function[state[0],state[1],action_index] += q_update
                self.policy = self.action_function.copy()

    def sarsa_lambda(self,n_episodes=1000,alpha=0.05,gamma=0.9,lamb=0.8):
        for k in range(n_episodes):
            self.E = np.zeros(shape=self.action_function.shape)
            state=(0,0)
            self.env.state = state
            episode_end = False
            epsilon = max(1/(1+k),0.2)
            action_index = self.apply_policy(state, epsilon)
            action = self.action_space[action_index]
            while not episode_end:
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.apply_policy(successor_state, epsilon)

                action_value = self.action_function[state[0], state[1], action_index]
                if not episode_end:
                    successor_action_value = self.action_function[successor_state[0],
                                                                  successor_state[1], successor_action_index]
                else:
                    successor_action_value = 0
                delta = reward + gamma * successor_action_value - action_value
                self.E[state[0],state[1],action_index] += 1
                self.action_function = self.action_function + alpha*delta*self.E
                self.E = gamma * lamb * self.E
                state = successor_state
                action = self.action_space[successor_action_index]
                action_index = successor_action_index
                self.policy = self.action_function.copy()

    def q_learning(self,n_episodes=1000, alpha=0.05,gamma=0.9):
        for k in range(n_episodes):
            state = (0,0)
            self.env.state = state
            episode_end = False
            epsilon = max(1/(k+1),0.1)
            while not episode_end:
                action_index = self.apply_policy(state,epsilon)
                action = self.action_space[action_index]
                reward, episode_end = self.env.step(action)
                successor_state = self.env.state
                successor_action_index = self.apply_policy(successor_state, -1)

                action_value = self.action_function[state[0], state[1], action_index]
                if not episode_end:
                    successor_action_value = self.action_function[successor_state[0],
                                                                  successor_state[1], successor_action_index]
                else:
                    successor_action_value = 0

                av_new = self.action_function[state[0],state[1],action_index] + alpha * (reward +\
                                                                                      gamma*successor_action_value -\
                                                                                      action_value)
                self.action_function[state[0],state[1],action_index] =  av_new
                self.policy = self.action_function.copy()
                state = successor_state










    def monte_carlo_control(self,epsilon=0.1):
        state = (0,0)
        self.env.state = state

        # Play out an episode
        states, actions, rewards = self.play_episode(state,epsilon=epsilon)

        first_visits = []
        for idx, state in enumerate(states):
            action_index = actions[idx]
            if (state,action_index) in first_visits:
                continue
            R = np.sum(rewards[idx:])
            if (state,action_index) in self.Returns.keys():
                self.Returns[(state, action_index)].append(R)
            else:
                self.Returns[(state,action_index)] = [R]
            self.action_function[state[0],state[1],action_index] = np.mean(self.Returns[(state,action_index)])
            first_visits.append((state,action_index))
        # Update the policy. In Monte Carlo Control, this is greedy behavior with respect to the action function
        self.policy = self.action_function.copy()


    def monte_carlo_evaluation(self, epsilon=0.1, first_visit=True):
        state = (0,0)
        self.env.state = state
        states, actions, rewards = self.play_episode(state,epsilon=epsilon)

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

    def TD_zero(self,epsilon=0.1, alpha=0.05,max_steps=1e3):
        state = (0,0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        episode_end = False
        count_steps = 0
        while not episode_end:
            count_steps+=1
            states.append(state)
            action_index = self.apply_policy(state,epsilon=epsilon)
            action = self.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            self.value_function[state[0], state[1]] = self.value_function[state[0], state[1]] + alpha * (
                    reward + self.lamb * self.value_function[suc_state[0], suc_state[1]] - self.value_function[state[0], state[1]])
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True

    def TD_lambda(self,epsilon = 0.1, alpha=0.05,gamma=0.9,max_steps=1e3):
        self.E = np.zeros(self.value_function.shape)
        state = (0,0)
        self.env.state = state
        states = []
        actions = []
        rewards = []
        count_steps=0
        episode_end = False
        while not episode_end:
            count_steps += 1
            states.append(state)
            action_index = self.apply_policy(state,epsilon=epsilon)
            action = self.action_space[action_index]
            actions.append(action)
            reward, episode_end = self.env.step(action)
            suc_state = self.env.state
            delta = reward + self.lamb * self.value_function[suc_state[0], suc_state[1]] - self.value_function[state[0],
                                                                                                               state[1]]
            self.E[state[0],state[1]] += 1

            # Note to self: vectorize code below.
            for row in range(self.value_function.shape[0]):
                for col in range(self.value_function.shape[1]):
                    self.value_function[row,col] = self.value_function[row,col] + alpha * delta * self.E[row,col]
                    self.E[row,col] = gamma * self.lamb * self.E[row, col]
            state = self.env.state

            if count_steps > max_steps:
                episode_end = True

    def evaluate_state(self, state):
        greedy_action_value = np.max(self.policy[state[0], state[1], :])
        greedy_indices = [i for i, a in enumerate(self.policy[state[0], state[1], :]) if
                          a == greedy_action_value]
        prob = 1 / len(greedy_indices)
        state_value = 0
        for i in greedy_indices:
            self.env.state = state  # reset state to the one being evaluated
            reward, episode_end = self.env.step(self.action_space[i])
            if self.synchronous:
                successor_state_value = self.value_function_old[self.env.state]
            else:
                successor_state_value = self.value_function[self.env.state]
            state_value += (prob * (reward + self.lamb * successor_state_value))
        return state_value

    def improve_policy(self):
        self.policy_old = self.policy.copy()
        for row in range(self.action_function.shape[0]):
            for col in range(self.action_function.shape[1]):
                for action in range(self.action_function.shape[2]):
                    self.env.state = (row, col)  # reset state to the one being evaluated
                    reward, episode_end = self.env.step(self.action_space[action])
                    successor_state_value = 0 if episode_end else self.value_function[self.env.state]
                    self.policy[row,col,action] = reward + successor_state_value

                max_policy_value = np.max(self.policy[row,col,:])
                max_indices = [i for i, a in enumerate(self.policy[row,col,:]) if a == max_policy_value]
                for idx in max_indices:
                    self.policy[row,col,idx] = 1

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
        print(self.value_function.round().astype(int))
        action_function_old = self.action_function.copy()
        print("\n Improving policy:")
        self.improve_policy()
        policy_stable = self.compare_policies() < 1
        print("policy diff:",policy_stable)

        if not policy_stable and iteration < 20:
            iteration += 1
            self.policy_iteration(iteration=iteration)
        elif policy_stable:
            print("Optimal policy found in", iteration, "steps of policy evaluation")
        else:
            print("failed to converge.")