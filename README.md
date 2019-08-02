
# Reinforcement Learning Chess
#### Arjan Groen



RLC works in three chess environments:

#### 1. Move Chess (Simple)
- Goal: Learn to find the shortest path between 2 squares on a chess board  
- Motivation: Move Chess has a small statespace, which allows us to tackle this with simple RL algorithms.
- Concepts: Dynamic Programming, Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration, Synchronous & Asynchronous back-ups, Monte Carlo (MC) Prediction, MC Control, Temporal Difference (TD) Learning, TD control, TD-lambda, SARSA(-max)

#### 2. Capture Chess (Intermediate)
- Goal: Capture as many pieces from the opponent within n fullmoves
- Motivation: Piece captures happen more frequently than win-lose-draw events. This give the algorithm more information to learn from.
- Concepts: Q-learning, value function approximation, experience replay, fixed-q-targets, policy gradients, REINFORCE, Actor-Critic.


#### 3. Real Chess (Hard)
- Goal: Play chess competitively against a human beginner
- Motivation: An actual RL chess AI, how cool is that?
- Concepts: Deep Q learning, Monte Carlo Tree Search 


# Installation
```bash
pip install git+https://github.com/arjangroen/RLC.git
```
    
# Usage
    
#### 1. Move Chess | Policy Iteration

```python
from RLC.move_chess.environment import Board
from RLC.move_chess.agent import Piece
from RLC.move_chess.learn import Reinforce

env = Board()
p = Piece(piece='rook')
r = Reinforce(p,env)

r.policy_iteration(k=1,gamma=1,synchronous=True)
```

##### 2. Move Chess | Q-learning

```python
from RLC.move_chess.environment import Board
from RLC.move_chess.agent import Piece
from RLC.move_chess.learn import Reinforce

p = Piece(piece='king')
env = Board()
r = Reinforce(p,env)
r.q_learning(n_episodes=1000,alpha=0.2,gamma=0.9)
r.visualize_policy()
r.agent.action_function.max(axis=2).round().astype(int)
```

#### 3. Capture Chess | Q-learning with value function approximation
```python
from RLC.capture_chess.environment import Board
from RLC.capture_chess.learn import Q_learning
from RLC.capture_chess.agent import Agent

board = Board()
agent = Agent(network='conv',gamma=0.1,lr=0.07)
R = Q_learning(agent,board)
pgn = R.learn(iters=750)
```

#### 4. Capture Chess | Policy Gradients - REINFORCE
```python
import chess
board = chess.Board()
from RLC.capture_chess.environment import Board
from RLC.capture_chess.learn import Reinforce
from RLC.capture_chess.agent import Agent, policy_gradient_loss

board = Board()
agent = Agent(network='conv_pg',lr=0.3)
R = Reinforce(agent,board)
pgn = R.learn(iters=3000)

```

#### 5. Capture Chess | Policy Gradients - Actor Critic
```python
import chess
from chess.pgn import Game
import RLC
from RLC.capture_chess.environment import Board
from RLC.capture_chess.learn import ActorCritic
from RLC.capture_chess.agent import Agent

board = Board()
critic = Agent(network='conv',lr=0.1)
critic.fix_model()
actor = Agent(network='conv_pg',lr=0.3)
R = ActorCritic(actor, critic,board)
pgn = R.learn(iters=1000)

```

# Kaggle kernels
https://www.kaggle.com/arjanso/reinforcement-learning-chess-1-policy-iteration  
https://www.kaggle.com/arjanso/reinforcement-learning-chess-2-model-free-methods  
https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks  
https://www.kaggle.com/arjanso/reinforcement-learning-chess-4-policy-gradients  
  
# References

1. Reinforcement Learning: An Introduction  
   Richard S. Sutton and Andrew G. Barto  
   1st Edition  
   MIT Press, march 1998
2. RL Course by David Silver: Lecture playlist  
   https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
3. Notes on Policy Gradients in autodiff frameworks  
   Aleksis Pirinen  
   https://aleksispi.github.io/assets/pg_autodiff.pdf, May 2018 
