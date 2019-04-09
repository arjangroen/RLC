import gym
import pprint
import numpy as np

class Board(object):
    
    def __init__(self):
        self.state = (0,0)
        self.reward_space = np.zeros(shape=(8,8)) - 1
        self.action_space = [(-1,0),  # north
                             (-1,1),  # north-west
                             (0,1),  # west
                             (1,1),  # south-west
                             (1,0),  # south
                             (1,-1),  # south-east
                             (0,-1),  # east
                             (-1,-1),  # north-east
                            ]
        self.terminal_state = (7,5)
        
    def step(self,action):
        reward = self.reward_space[self.state[0],self.state[1]]
        if self.state==self.terminal_state:
            episode_end = True
            return 0, episode_end
        else:
            episode_end = False
            old_state = self.state
            self.state = (self.state[0] + action[0], self.state[1] + action[1])  # step
            self.state = old_state if np.min(self.state) < 0 or np.max(self.state) > 7 else self.state
            return reward, episode_end
        
        
    def render(self):
        visual_row = ["[ ]","[ ]","[ ]","[ ]","[ ]","[ ]","[ ]","[ ]"]
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())
        visual_board[self.state[0]][self.state[1]] = "[K]"
        visual_board[self.terminal_state[0]][self.terminal_state[1]] = "[Q]"
        self.visual_board = visual_board
        
class Agent(object):
    
    def __init__(self,env):
        self.env = env
        self.value_function = np.zeros(shape=env.reward_space.shape)
        self.action_function = np.zeros(shape=(env.reward_space.shape[0],
                                              env.reward_space.shape[1],
                                              len(env.action_space)))
    
    def evaluate_policy(self):
        self.value_function_old = self.value_function.copy()
        for row in range(self.value_function.shape[0]):
            for col in range(self.value_function.shape[1]):
                self.value_function[row,col] = self.evaluate_state((row,col))
                
    def evaluate_state(self,state):
        action_values = self.action_function[state[0],state[1],:]
        max_action_value = np.max(action_values)
        max_indices = [i for i,a in enumerate(action_values) if a==max_action_value]
        prob = 1/len(max_indices)
        state_value = 0
        for i in max_indices:
            self.env.state = state  # reset state to the one being evaluated
            reward, episode_end = self.env.step(self.env.action_space[i])
            successor_state_value = self.value_function_old[self.env.state]
            state_value += (prob*(reward + successor_state_value))
        return state_value
    
    def improve_policy(self):
        for row in range(self.action_function.shape[0]):
            for col in range(self.action_function.shape[1]):
                for action in range(self.action_function.shape[2]):
                    self.env.state=(row,col)  # reset state to the one being evaluated
                    reward, episode_end = self.env.step(self.env.action_space[action])
                    successor_state_value = 0 if episode_end else self.value_function[self.env.state]
                    self.action_function[row,col,action] = reward + successor_state_value
        
    
    def policy_iteration(self,k_max=50,eps=0.1,iteration=1):
        policy_stable = True
        print("\n\n______iteration:",iteration,"______")
        print("\n policy:")
        self.visualize_policy()
        
        print("")
        value_delta_max = 0
        for k in range(k_max):
            self.evaluate_policy()
            value_delta = np.max(np.abs(self.value_function_old - self.value_function))
            value_delta_max = max(value_delta_max,value_delta)
            if value_delta_max < eps:
                break
        print("Value function for this policy:")
        print(self.value_function.astype(int))
        action_function_old = self.action_function.copy()
        print("\n Improving policy:")
        self.improve_policy()
        policy_delta = np.sum(np.abs(np.argmax(action_function_old,axis=2) - np.argmax(self.action_function,axis=2)))
        print("policy difference in improvement",policy_delta)
        print("________________________________")
        
        if policy_delta > 0 and iteration < 20:
            iteration+=1
            self.policy_iteration(iteration=iteration)
        else:   
            print("Optimal policy found in",iteration,"steps of policy evaluation")
                
    def visualize_policy(self):
        greedy_policy = King.action_function.argmax(axis=2)
        policy_visualization = {}
        arrows = "↑ ↗ → ↘ ↓ ↙ ← ↖"
        arrowlist = arrows.split(" ")
        for idx, arrow in enumerate(arrowlist):
            policy_visualization[idx] = arrow
            
        visual_row = ["[ ]","[ ]","[ ]","[ ]","[ ]","[ ]","[ ]","[ ]"]
        visual_board = []
        for c in range(8):
            visual_board.append(visual_row.copy())

        for row in range(greedy_policy.shape[0]):
            for col in range(greedy_policy.shape[1]):
                visual_board[row][col] = policy_visualization[greedy_policy[row,col]]

        visual_board[King.env.terminal_state[0]][King.env.terminal_state[1]] = "Q"
        pprint.pprint(visual_board)
        
if __name__ == '__main__':
    board = Board()
    board.render()
    print("\n In this environment, the King is looking for the shortest path to the Queen")
    print("Every step the King takes costs -1.")
    print("The environment looks as follows")
    pprint.pprint(board.visual_board)
    print("\n Policy Evaluation find the best policy for the King:")
    King = Agent(board)
    King.policy_iteration(eps=0.1)