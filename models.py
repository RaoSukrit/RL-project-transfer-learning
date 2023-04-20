# Resnet CNN
import os
import torch
import torch.nn as nn

from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torchvision.models import resnet18, resnet34

# os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_GL'] = "osmesa"
os.environ.get('MUJOCO_GL', 'MUJOCO_GL not set')


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box,
                 fc_features_dim: int = 256,
                 cnn_model_type: str = 'resnet18'):
        super().__init__(observation_space, fc_features_dim, cnn_model_type)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        if cnn_model_type == 'resnet18':
            model = resnet18()
        elif cnn_model_type == 'resnet34':
            model = resnet34()

        self.cnn = model

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, fc_features_dim),
                                    nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))
