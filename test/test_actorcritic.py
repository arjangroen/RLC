from RLC.real_chess.agent import ActorCritic
import numpy as np
import torch

test_state = torch.from_numpy(np.random.randn(1, 8,8,8)).float()


ac = ActorCritic()


actor_out, critic_out = ac(test_state)

print('done')
