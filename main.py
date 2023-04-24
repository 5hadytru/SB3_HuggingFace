import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pybullet_envs
import pybullet as p
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import ViTFeatureExtractor, ViTModel
import torch
from torch import nn
from torch.cuda.amp import autocast
import cv2
import time
from stable_baselines3.common.callbacks import BaseCallback
from gym3 import ViewerWrapper, ToGymEnv


class CustomResolutionWrapper(gym.ObservationWrapper):
    def __init__(self, env, resolution=(224, 224)):
        super().__init__(env)
        self.resolution = resolution
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(resolution[1], resolution[0], 3), dtype=np.uint8)

    def observation(self, obs):
        def process_image(img):
            img_resized = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
            img_normed = img_resized.astype(np.float32)
            return img_normed
        
        print(obs.shape, np.min(obs), np.max(obs), obs.dtype)

        processed_images = np.apply_along_axis(process_image, 1, obs)

        print(processed_images.shape)

        return processed_images


class NoRenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


class ViT_Policy(nn.Module):
    def __init__(self, action_space, pretrained_model_name, device):
        super(ViT_Policy, self).__init__()
        self.vit = HF_ViT(pretrained_model_name, device).to(device)

        self.input_dim = self.vit.output_dim
        self.features_dim = action_space.shape[0]

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, action_space.shape[0])

    def forward(self, obs):
        x = self.vit(obs)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


vit_features_dims = {
    "google/vit-base-patch16-224": 768
}


class HF_ViT(nn.Module):
    def __init__(self, pretrained_model_name, device):
        super(HF_ViT, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(pretrained_model_name)
        self.vit_model = ViTModel.from_pretrained(pretrained_model_name)
        self.output_dim = vit_features_dims[pretrained_model_name]
        self.device = device
    
        for param in self.vit_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.squeeze().cpu()
        pixel_values = self.feature_extractor(images, return_tensors="pt")["pixel_values"].to(self.device)
        outputs = self.vit_model(pixel_values)

        return outputs.last_hidden_state[:, 0].squeeze()


def run_training():
    env_name = "CarRacing-v0"
    env = gym.make(env_name)
    env = CustomResolutionWrapper(env, resolution=(224, 224))
    env = DummyVecEnv([lambda: env])

    device = torch.device("cuda:0")

    pretrained_model_name = "google/vit-base-patch16-224"

    policy_kwargs = dict(
        net_arch=[
            dict(
                pi=[224, 256, 3],
                vf=[224, 256, 1]
            )
        ],
        features_extractor_class=ViT_Policy,
        features_extractor_kwargs=dict(pretrained_model_name=pretrained_model_name, device=device),
    )

    callback = NoRenderCallback()
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cuda:0")
    model.learn(total_timesteps=2048, progress_bar=True, callback=callback)

    rewards = model.episode_reward_history
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(f"plots/{pretrained_model_name}_rewards_{time.time()}.png")

    model.save("ppo_CarRacing-v0_vit")
    env.close()


def visualize():
    env_name = "CarRacing-v0"
    env = gym.make(env_name)
    model = PPO.load("ppo_humanoid_vit", device="cuda:0")

    obs = env.reset()
    done = False
    img_array = []

    while not done:
        img = env.render(mode='rgb_array')
        img_array.append(img)
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

    image_folder = "./humanoid_images"
    os.makedirs(image_folder, exist_ok=True)

    for idx, frame in enumerate(img_array):
        img = Image.fromarray(frame.astype(np.uint8))
        img.save(f"{image_folder}/frame_{idx:04d}.png")

    env.close()


if __name__ == "__main__":
    run_training()
    #visualize()
