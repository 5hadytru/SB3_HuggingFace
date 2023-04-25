import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from transformers import ViTFeatureExtractor, ViTModel
import torch
from torch import nn
from torch.cuda.amp import autocast
import cv2
import time, sys
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter


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

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, f"best_model_PPO_{sys.argv[1]}_{sys.argv[3]}_{time.time()}.pth")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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


def run_training():
    env_name = sys.argv[1]
    env = gym.make(env_name)
    env = CustomResolutionWrapper(env, resolution=(224, 224))
    #env = DummyVecEnv([lambda: env])

    log_dir = "logs"
    env = Monitor(env, log_dir, filename=f"monitor_{time.time()}")

    device = torch.device("cuda:0")

    pretrained_model_name = sys.argv[2]

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

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model = PPO("MlpPolicy", env, learning_rate=float(sys.argv[4]), policy_kwargs=policy_kwargs, verbose=1, device="cuda:0")
    model.learn(total_timesteps=int(sys.argv[3]), callback=callback, progress_bar=True)

    # Helper from the library
    results_plotter.plot_results(
        [log_dir], 1e5, results_plotter.X_TIMESTEPS, "PPO CarRacing-v0"
    )

    plot_results(log_dir)

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
