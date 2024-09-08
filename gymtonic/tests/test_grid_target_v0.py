import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import gymtonic.envs

n_rows = 8
n_columns = 10

env = gym.make('gymtonic/GridTarget-v0', n_rows=n_rows, n_columns=n_columns, render_mode=None)
check_env(env, warn=True, skip_render_check=False) 

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25_000)

env = gym.make('gymtonic/GridTarget-v0', n_rows=n_rows, n_columns=n_columns, render_mode='human')
env = Monitor(env)
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10, deterministic=True, render=True)
print(f"Mean_reward:{mean_reward:.2f}")