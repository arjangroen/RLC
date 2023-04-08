import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import logging
logging.basicConfig(level=logging.INFO)

torch.autograd.set_detect_anomaly(True)

NUM_FEATURES = 1

priority = [
    "0", "O",  # Castling
    "a", "h", "b", "g", "c", "f", "d", "e",  # Pawn moves outside-in
    "B", "N", "R", "Q", "K"   # Bishop - Knight - Rook - Queen - King
]
priorities = {c: i for i, c in enumerate(priority)}


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.Conv2d(out_c, out_c, stride=2, kernel_size=2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class ChessUNet(nn.Module):

    def __init__(self, is_actor=False):
        super().__init__()
        self.is_actor = is_actor
        self.input_embedding_layer = nn.Conv2d(
            8, 3, kernel_size=3, padding=1)  # output: 3 x 8 x 8
        self.e1 = EncoderBlock(3, 8)  # output: 8 x 4 x 4
        self.e2 = EncoderBlock(8, 16)  # output: 16 x 2 x 2
        self.b = ConvBlock(16, 32)  # output 32 x 2 x 2
        self.d1 = DecoderBlock(32, 16)  # output 16 x 4 x 4
        self.d2 = DecoderBlock(16, 8)  # output 8 x 8 x 8
        self.output_layer2d = nn.Conv2d(
            in_channels=8, out_channels=1, kernel_size=1)
        self.output_flat = nn.Flatten()

    def forward(self, inputs):

        # For computing action probabilities, states are color invariant.
        if self.is_actor:
            color = inputs[:, 6, :, :].mean()
            inputs = inputs * color
            inputs = torch.flip(inputs, dims=[2])

        p0 = self.input_embedding_layer(inputs)
        s1, p1 = self.e1(p0)
        s2, p2 = self.e2(p1)
        b = self.b(p2)
        d1 = self.d1(b, s2)
        d2 = self.d2(d1, s1)
        outputs2d = self.output_layer2d(d2)

        # Flip back
        if self.is_actor:
            outputs2d = torch.flip(outputs2d, dims=[2])

        outputs = self.output_flat(outputs2d)
        return outputs


class NanoActorCritic(nn.Module):

    def __init__(self, gamma=0.8, lr=.001, verbose=0):
        """
        Agent that plays the white pieces in capture chess.
        The action space is defined as a combination of a "source square" and a "target square".
        The actor gives an action probability for each square combination.
        The critic gives an action value for each square combination.
        Args:
            gamma: float
                Temporal discount factor
            network: str
                'linear' or 'conv'
            lr: float
                Learning rate, ideally around 0.001
        """
        super(NanoActorCritic, self).__init__()
        self.gamma = gamma
        self.lr = lr
        self.verbose = verbose
        self.actor = ChessUNet(is_actor=True)
        self.critic = ChessUNet()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=1e-4)
        self.historical_pg_losses = torch.tensor([0.]).float()
        self.replay_importance = []

    def forward(self, state):
        """
        A forward pass through the neural network
        :param state: torch.Tensor of (1, 8, 8, 8)
        :return: (action_probabilites, q_values)
        """

        action_logits = self.actor(state)
        action_values = self.critic(state)

        return action_logits, action_values

    def select_action(self, env, greedy=False):
        """
        Select an action using the action probabilites supplied by the actor
        :param env: Python chess environment
        :param bool greedy: Whether to always pick the most probable action
        :return: a move and its probability
        """
        action_space = torch.from_numpy(np.expand_dims(env.project_legal_moves(),
                                                       axis=0)).float()  # The environment determines which moves are legal
        state = torch.from_numpy(np.expand_dims(
            env.layer_board, axis=0)).float()
        action_logits, action_values = self(state)
        action_logits = self.legalize_action_logits(
            action_logits, action_space)
        action_logits = action_logits.reshape(1, 64)
        action_probs = F.softmax(
            action_logits, dim=1).detach()
        if greedy:
            move_to_idx = np.argmax(action_probs[0])
        else:
            move_to_idx = np.random.choice(
                range(64), p=action_probs[0].numpy())
        move_proba = action_probs[0][move_to_idx]
        moves = [x for x in env.board.generate_legal_moves()
                 if x.to_square == move_to_idx]

        # multiple moves can have the same target square. For now, move with the weakest piece
        if len(moves) > 1:
            min_prio = 100
            min_move = None
            for move in moves:
                ucistr = move.uci()
                prio = priorities[ucistr[0]]
                if prio < min_prio:
                    min_prio = prio
                    min_move = move
        else:
            min_move = moves[0]
        return min_move, move_proba

    def td_update(self, fixed_model, episode_actives, states, moves, rewards, successor_states, successor_moves,
                  action_spaces):
        """
        Performs a Temporal Difference Update on the network
        :param fixed_model, stationary ActorCritic model
        :param states: Tensor of shape (n_samples, 8, 8, 8)
        :param actions: list[chess.move]
        :param rewards: Tensor of shape (n_samples, 1)
        :param successor_states: Tensor of shape (n_samples, 8, 8, 8)
        :param successor_actions: list[chess.move]
        :param action_spaces: Tensor for shape (n_samples, 64, 64)
        :return:
        """
        successor_qs = fixed_model.critic(successor_states).detach()
        predicted_q = self.critic(states)
        current_action_logits = self.actor(states)
        old_action_logits = fixed_model.actor(states).detach()

        # Calculate Expected Q
        bootstrapped_q = rewards + episode_actives * self.gamma * successor_qs
        bootstrapped_q_indexer = torch.LongTensor(
            [m.to_square for m in successor_moves]).unsqueeze(dim=1)
        bootstrapped_q_sliced = torch.gather(
            bootstrapped_q, 1, bootstrapped_q_indexer)

        # Calculate Predicted Q
        move_indexer = torch.LongTensor(
            [m.to_square for m in moves]).unsqueeze(dim=1)
        predicted_q_sliced = torch.gather(predicted_q, 1, move_indexer)

        # Q Loss
        Q_loss = F.mse_loss(predicted_q_sliced.float(),
                            bootstrapped_q_sliced.float())

        # Action Probas
        colors = states[:, 6, 0, 0].unsqueeze(1)
        advantages_unbased = bootstrapped_q_sliced * colors

        # Substract Baseline for stable performance
        advantages = (advantages_unbased - advantages_unbased.mean()
                      ) / advantages_unbased.std()

        actions_probs_all = F.softmax(self.legalize_action_logits(
            current_action_logits, action_spaces), dim=1)
        action_probs = torch.gather(actions_probs_all, 1, move_indexer)
        actions_probs_fixed_all = F.softmax(self.legalize_action_logits(
            old_action_logits, action_spaces), dim=1)
        action_probs_fixed = torch.gather(
            actions_probs_fixed_all, 1, move_indexer)

        proba_rt_unclipped = action_probs/action_probs_fixed
        proba_rt_clipped = torch.clamp(proba_rt_unclipped, min=0.9, max=1.1)

        ppo_loss = (-torch.min(proba_rt_clipped*advantages,
                               proba_rt_unclipped*advantages)).mean()

        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        ppo_loss.backward()
        self.actor_optimizer.step()

    def get_proba_for_move(self, move, action_logits, action_space_tensor):
        probas = F.softmax(self.legalize_action_logits(
            action_logits, action_space_tensor))
        proba = probas.view(probas.shape[0], 64)[
            [0], move.to_square]
        return proba

    def flatten_for_softmax(self, action_logits):
        batch_dim = action_logits.shape[0]
        if len(action_logits.shape) == 3:
            action_logits = action_logits.reshape(batch_dim, 64)
        return action_logits

    def legalize_action_logits(self, action_logits, action_space_tensor):
        """
        Sets the logit of illegal moves to -inf and re-normalizes the probabilities over the action space
        :param action_probs: torch.tensor = Action probabilites gives by the policy
        :param action_space_tensor: torch.tensor = Action Space
        :param eps: float = Small number for numerical stability
        :return: torch.tensor = action_probs_legal_normalized
        """
        action_logits_legal = action_logits.masked_fill(
            action_space_tensor == 0, float('-inf'))
        return action_logits_legal

    def mc_update_agent(self, state, move, Returns, action_space, fixed_model):
        """
        Update the actor and the critic based on observed Returns in Monte Carlo simulations
        :param torch.tensor state: The root state of the board
        :param move: The move that was taken
        :param Returns: The observed return in the Monte Carlo simulations
        :param action_space: The action space
        :return:
        """
        n_legal_actions = action_space.sum()
        if n_legal_actions == 1:
            return

        Returns_tensor = torch.tensor(Returns).float()
        state_tensor = torch.from_numpy(state).unsqueeze(dim=0).float()
        action_space_tensor = torch.from_numpy(
            action_space).unsqueeze(dim=0).float()

        if state_tensor[0, 6, 0, :].detach().numpy().sum() == -8.0:
            color = torch.Tensor([-1.])
        else:
            color = torch.Tensor([1.])

        action_logits, q_values = self(state_tensor)
        action_logits_old, _ = fixed_model(state_tensor)
        action_logits_old = action_logits_old.detach()
        action_prob = self.get_proba_for_move(
            move, action_logits, action_space_tensor)
        action_prob_fixed = self.get_proba_for_move(
            move, action_logits_old, action_space_tensor)

        # entropy = torch.ones_like(action_space_tensor) / torch.tensor([4096.]).float()
        # alpha = torch.tensor([1e3])
        # entropy_bonus = alpha * - \
        #    (torch.log(action_probs) * action_probs).mean().mean()

        # Adjust advantage for mean advantage
        q_copy = q_values.clone().detach()
        # advantage = (Returns_tensor - (q_copy *
        #             action_space_tensor).sum()/n_legal_actions) * color
        advantage = (Returns_tensor * color) - torch.tensor([0.001])
        # advantage = Returns_tensor  # Use returns a advantage for now

        predicted_returns = q_values[[0], move.to_square]

        proba_rt_clipped = torch.clamp(
            action_prob/action_prob_fixed, min=0.9, max=1.1)
        proba_rt_unclipped = action_prob/action_prob_fixed

        ppo_loss = -torch.min(proba_rt_clipped*advantage,
                              proba_rt_unclipped*advantage)

        q_value_loss = F.mse_loss(Returns_tensor, predicted_returns)

        # The chance updating to proportional to the relative size of the loss.
        self.historical_pg_losses = torch.cat(
            (self.historical_pg_losses, ppo_loss.clone().detach()))
        if len(self.historical_pg_losses) > 1000:
            self.historical_pg_losses = self.historical_pg_losses[1:]

        self.critic_optimizer.zero_grad()
        q_value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        ppo_loss.backward()
        self.actor_optimizer.step()

        # VERIFY successfull optimization step
        action_logits_post, q_values_post = self(state_tensor)
        predicted_returns_post = q_values_post[[
            0], move.to_square]
        proba_post = self.get_proba_for_move(
            move, action_logits_post, action_space_tensor)

        logging.info("\nLEARNED POLICY")
        logging.info("Q LOSS %s", q_value_loss)
        logging.info("POLICY GRADIENT LOSS %s", ppo_loss)
        logging.info("COLOR: %s", color)
        logging.info("Returns %s", Returns)
        logging.info("Advantage %s", advantage)
        logging.info("proba %s", action_prob)
        logging.info("new proba %s", proba_post)
        logging.info("Change in proba %s", proba_post - action_prob)
        logging.info("right policy direction %s", advantage *
                     (proba_post - action_prob) >= 0)
        logging.info("proba_rt %s", proba_rt_unclipped)
        logging.info("new proba_rt %s", proba_post/action_prob_fixed)

        logging.info("expected %s", predicted_returns)
        logging.info("new expected %s", predicted_returns_post)
        logging.info("Change in Q %s",
                     predicted_returns_post - predicted_returns)
        logging.info("Loss reduction Q %s", F.mse_loss(
            Returns_tensor, predicted_returns_post) - q_value_loss)


class RandomAgent(object):

    def predict(self, board_layer):
        return np.random.randint(-5, 5) / 5

    def select_move(self, board):
        moves = [x for x in board.generate_legal_moves()]
        return np.random.choice(moves)
