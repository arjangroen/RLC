from RLC.real_chess.agent import ActorCritic
import numpy as np
import torch

test_state = torch.from_numpy(np.random.randn(3, 8,8,8)).float()
test_actionspace = np.zeros(shape=(3,64, 64))
test_actionspace[:,0,0] = 1.
test_actionspace = torch.from_numpy(test_actionspace).float()

ac = ActorCritic()


actor_out, critic_out = ac(test_state, test_actionspace)

print('done')
