# RLC
Arjan Groen  
June 3rd, 2019

# Reinforcement Learning Chess
Hi there! If you're interested in learning about reinforcement learning, you are in the right place. As we all know the best way to learn about a topic is to build something. In this series of notebooks I work may way towards a full-fledged chess AI named RLC (Reinforcement Learning Chess). 

Tackling chess is a big challenge, mainly because of its huge state-space. Therefore I start with simpler forms of chess and solve these problems with elementary RL-techniques. Gradually I will expand this untill we end up in a chess AI that can play actual games of chess somewhat intelligibly. The forms of chess I cover in my notebooks are:  

#### 1. Move Chess
- Goal: Learn to find the shortest path between 2 squares on a chess board  
- Motivation: Move Chess has a small statespace, which allows us to tackle this with simple RL algorithms.
- Techniques & Concepts: Dynamic Programming, Policy Iteration, Value Iteration, Synchronous & Asynchronous back-ups

#### 2. Capture Chess
- Goal: Capture as many pieces from the opponent within n fullmoves
- Motivation: Piece captures happen more frequently than win-lose-draw events. This give the algorithm more rewards to learn from
- Techniques & Concepts: Q-learning, value function approximation, experience replay, fixed-q-targets


#### 3. Real Chess (a.k.a. chess)
- Goal: Deliver checkmate
- Motivation: It's the real deal
- Techniques & Concepts: Deep Q learning, policy gradients, actor-critic model, more? 



The content is based on David Silver's (Deepmind) lectures that are available on Youtube and the book Introduction to Reinforcement Learning by Sutton and Barto.
