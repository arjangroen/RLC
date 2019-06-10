
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
- Concepts: Q-learning, value function approximation, experience replay, fixed-q-targets


#### 3. Real Chess (Hard)
- Goal: Play chess competitively against a human beginner
- Motivation: An actual RL chess AI, how cool is that?
- Concepts: Deep Q learning, policy gradients, actor-critic model and maybe more. 


# Installation


```python
!pip install git+https://github.com/arjangroen/RLC.git
```

    Collecting git+https://github.com/arjangroen/RLC.git
      ...
    Successfully built RLC
    
# Running Policy Iteration on Move Chess

```python
from RLC.move_chess.environment import Board
from RLC.move_chess.agent import Piece
from RLC.move_chess.learn import Reinforce

env = Board()
p = Piece(piece='rook')
r = Reinforce(p,env)

r.policy_iteration(k=1,gamma=1,synchronous=True)
```


# References

1. Reinforcement Learning: An Introduction  
   Richard S. Sutton and Andrew G. Barto  
   1st Edition  
   MIT Press, march 1998
2. RL Course by David Silver: Lecture playlist  
   https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ

