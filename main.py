import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from transformers import ViTFeatureExtractor, ViTModel, AutoFeatureExtractor, ResNetForImageClassification
import torch
from torch import nn
from torch.cuda.amp import autocast
import cv2
import time, sys
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import random
import tqdm


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
        
        #print(obs.shape, np.min(obs), np.max(obs), obs.dtype)

        # Remove np.apply_along_axis
        processed_image = process_image(obs)

        #print(processed_image.shape)

        return processed_image


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, env_rank:int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.env_rank = env_rank
        self.save_path = os.path.join(log_dir, f"PPO_{sys.argv[1]}_{sys.argv[3]}.pth")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True


class Policy(nn.Module):
    def __init__(self, action_space, model_constr, device):
        super(Policy, self).__init__()
        self.model = model_constr(device)

        self.input_dim = self.model.output_dim
        self.features_dim = action_space.shape[0]

        self.fc1 = nn.Linear(self.input_dim, 256)
        self.fc2 = nn.Linear(256, action_space.shape[0])

    def forward(self, obs):
        x = self.model(obs)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HF_ViT_B(nn.Module):
    def __init__(self, device):
        super(HF_ViT_B, self).__init__()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.output_dim = 768
        self.device = device
    
        for param in self.vit_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.squeeze().cpu()
        pixel_values = self.feature_extractor(images, return_tensors="pt")["pixel_values"].to(self.device)
        outputs = self.vit_model(pixel_values)

        return outputs.last_hidden_state[:, 0].squeeze()


class HF_RN18(nn.Module):
    def __init__(self, device):
        super(HF_RN18, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
        self.resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-18").resnet
        self.output_dim = 512
        self.device = device
    
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.squeeze().cpu()
        pixel_values = self.feature_extractor(images, return_tensors="pt")["pixel_values"].to(self.device)
        outputs = self.resnet(pixel_values)["pooler_output"]

        return outputs.squeeze()


class HF_RN50(nn.Module):
    def __init__(self, device):
        super(HF_RN50, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.resnet = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").resnet
        self.output_dim = 2048
        self.device = device
    
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, images):
        images = images.squeeze().cpu()
        pixel_values = self.feature_extractor(images, return_tensors="pt")["pixel_values"].to(self.device)
        outputs = self.resnet(pixel_values)["pooler_output"]

        return outputs.squeeze()


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()
    plt.savefig(f"PPO_{sys.argv[1]}_{sys.argv[2]}_{sys.argv[3]}_{time.time()}.png".replace("/", "_"))


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None

    def _init_callback(self):
        if self.verbose > 0:
            self.progress_bar = tqdm.tqdm(total=self.total_timesteps, desc="Training progress")

    def _on_step(self):
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        return True

    def _on_training_end(self):
        if self.progress_bar is not None:
            self.progress_bar.close()



def make_env(env_id, log_dir, rank, seed=0):
    def _init():
        env = gym.make(env_id)
        #env = CustomResolutionWrapper(env, resolution=(224, 224))
        env.seed(seed + rank)
        monitor_file_prefix = os.path.join(log_dir, str(rank))
        env = Monitor(env, monitor_file_prefix)
        return env
    return _init

model_index = {
    "resnet-18": HF_RN18,
    "resnet-50": HF_RN50,
    "vit-b": HF_ViT_B
}

def run_training():
    num_envs = 4
    run_id = int(time.time())
    log_dir = f"logs/run_{run_id}/"

    os.mkdir(log_dir)
    for i in range(num_envs):
        os.mkdir(log_dir + f"{i}")

    env_name = sys.argv[1]
    envs = [make_env(env_name, log_dir, i) for i in range(num_envs)]
    env = SubprocVecEnv(envs)
    env = VecTransposeImage(env)

    device = torch.device("cuda:0")

    pretrained_model_name = sys.argv[2]

    policy_kwargs = dict(
        net_arch=[
            dict(
                pi=[256],
                vf=[256]
            )
        ],
        # features_extractor_class=Policy,
        # features_extractor_kwargs=dict(model_constr=model_index[pretrained_model_name], device=device),
    )

    model = PPO(
        "CnnPolicy", 
        env, 
        learning_rate=float(sys.argv[4]), 
        #policy_kwargs=policy_kwargs, 
        verbose=1, 
        device="cuda:0")

    callbacks = [SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir + f"/{i}", env_rank=i) for i in range(num_envs)]
    callbacks.extend([ProgressBarCallback(int(sys.argv[3]))])

    model.learn(total_timesteps=int(sys.argv[3]), callback=callbacks)

    # Helper from the library
    for i in range(num_envs):
        env_log_dir = f"{log_dir}/{i}"
        results_plotter.plot_results(
            [env_log_dir], int(sys.argv[3]), results_plotter.X_TIMESTEPS, "PPO CarRacing-v0"
        )
        plot_results(env_log_dir)

    model.save(f"PPO_{sys.argv[1]}_{sys.argv[2].replace('/', '_')}_{sys.argv[3]}_{time.time()}.pth")
    env.close()

    print("Finished training")


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
