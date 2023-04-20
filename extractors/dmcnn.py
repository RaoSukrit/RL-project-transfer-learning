# Deepmind CNN
import torch as th
import torch.nn as nn

from gym import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box,
                 fc_features_dim=128):
        super().__init__(observation_space, fc_features_dim)

        # get input channels
        n_input_channels = observation_space.shape[0]

        # 3 conv layers with 32 filters each 
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # 2 linear fc layers
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 200), 
            nn.ReLU(),
            nn.Linear(200, fc_features_dim), 
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))