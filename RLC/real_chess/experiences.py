from pydantic import BaseModel
import numpy as np
import chess
from typing import Optional, List
import torch


class MonteCarloTrainingSample(BaseModel):
    layer_board: torch.Tensor
    returns: torch.Tensor
    action_space: torch.Tensor
    color: torch.Tensor
    move_to: int

    class Config:
        arbitrary_types_allowed = True


class MonteCarloTrainingBatch(BaseModel):
    layer_boards: torch.Tensor
    returns: torch.Tensor
    action_spaces: torch.Tensor
    colors: torch.Tensor
    moves_to: torch.LongTensor

    class Config:
        arbitrary_types_allowed = True


class MonteCarloTrainingData(BaseModel):
    samples: Optional[List[MonteCarloTrainingSample]]


class TemporalDifferenceStep(BaseModel):
    layer_board: torch.Tensor
    reward: torch.Tensor
    action_space: torch.Tensor
    color: torch.Tensor
    move_to: int
    episode_active: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class TemporalDifferenceEpisode(BaseModel):
    steps: List[TemporalDifferenceStep]


class TemporalDifferenceMemory(BaseModel):
    episodes: Optional[List[TemporalDifferenceEpisode]]


class TemporalDifferenceTrainingSample(BaseModel):
    layer_board: torch.Tensor
    reward: torch.Tensor
    action_space: torch.Tensor
    color: torch.Tensor
    move_to: int
    successor_layer_board: torch.Tensor
    successor_action_space: torch.Tensor
    successor_move_to: int
    episode_active: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class TemporalDifferenceTrainingBatch(BaseModel):
    layer_boards: torch.Tensor
    rewards: torch.Tensor
    action_spaces: torch.Tensor
    colors: torch.Tensor
    moves_to: torch.LongTensor
    successor_layer_boards: torch.Tensor
    successor_action_spaces: torch.Tensor
    successor_moves_to: torch.LongTensor
    episode_actives: torch.Tensor

    class Config:
        arbitrary_types_allowed = True


class TemporalDifferenceTrainingData(BaseModel):
    samples: List[TemporalDifferenceTrainingSample]
