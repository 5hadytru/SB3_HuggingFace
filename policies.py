# custom modules
from .vision_models import ViTWrapper 
import torch
from torch import nn
from torch.cuda.amp import autocast


class CustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space, pretrained_model_name):
        super(CustomPolicy, self).__init__()
        self.vit = ViTWrapper(pretrained_model_name)
        self.fc1 = nn.Linear(self.vit.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, action_space.shape[0])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs):
        obs = obs.to(self.device)
        with autocast():
            x = self.vit(obs)
            x = nn.functional.relu(self.fc1(x))
            x = self.fc2(x)
        return x