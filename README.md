
# Reinforcement Learning Chess
#### Arjan Groen



RLC works three chess variants:

#### 1. Move Chess 
- Goal: Learn to find the shortest path between 2 squares on a chess board  
- Motivation: Move Chess has a small statespace, which allows us to tackle this with simple RL algorithms.
- Concepts: Dynamic Programming, Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration, Synchronous & Asynchronous back-ups, Monte Carlo (MC) Prediction, MC Control, Temporal Difference (TD) Learning, TD control, TD-lambda, SARSA(-max)

#### 2. Capture Chess
- Goal: Capture as many pieces from the opponent within n fullmoves
- Motivation: Piece captures happen more frequently than win-lose-draw events. This give the algorithm more information to learn from.
- Concepts: Q-learning, value function approximation, experience replay, fixed-q-targets


#### 3. Real Chess (a.k.a. chess)
- Goal: Play chess competitively against a human beginner
- Motivation: An actual RL chess AI, how cool is that?
- Concepts: Deep Q learning, policy gradients, actor-critic model and maybe more. 


# Installation


```python
!pip install git+https://github.com/arjangroen/RLC.git
```

    Collecting git+https://github.com/arjangroen/RLC.git
      Cloning https://github.com/arjangroen/RLC.git to /private/var/folders/kw/5gm0lng13pbf8ccm0xg74qj00000gn/T/pip-req-build-2m_9voyp
    Requirement already satisfied (use --upgrade to upgrade): RLC==0.1 from git+https://github.com/arjangroen/RLC.git in /Users/a.groen/.conda/envs/RLC/lib/python3.7/site-packages
    Building wheels for collected packages: RLC
      Building wheel for RLC (setup.py) ... [?25ldone
    [?25h  Stored in directory: /private/var/folders/kw/5gm0lng13pbf8ccm0xg74qj00000gn/T/pip-ephem-wheel-cache-c6tl4x_y/wheels/04/68/a5/cb835cd3d76a49de696a942739c71a56bfe66d0d8ea7b4b446
    Successfully built RLC

# References

1. Reinforcement Learning: An Introduction  
   Richard S. Sutton and Andrew G. Barto  
   1st Edition  
   MIT Press, march 1998
2. RL Course by David Silver: Lecture playlist  
   https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ

