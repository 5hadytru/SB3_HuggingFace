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

# custom modules
from .policies import CustomPolicy


def run_training():
    env_name = "HumanoidBulletEnv-v0"
    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    pretrained_model_name = "google/vit-base-patch16-224"

    policy_kwargs = dict(
        net_arch=[64, dict(pi=[64, 64], vf=[64, 64])],
        features_extractor_class=CustomPolicy,
        features_extractor_kwargs=dict(pretrained_model_name=pretrained_model_name),
    )

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, device="cuda")
    model.learn(total_timesteps=100000)

    model.save("ppo_humanoid_vit")
    env.close()


def visualize_humanoid():
    env_name = "HumanoidBulletEnv-v0"
    env = gym.make(env_name)
    model = PPO.load("ppo_humanoid_vit", device="cuda")

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
        img = Image.fromarray(frame)
        img.save(f"{image_folder}/frame_{idx:04d}.png")

    env.close()


if __name__ == "__main__":
    run_training()
    visualize_humanoid()
