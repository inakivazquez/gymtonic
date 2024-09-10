import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import gymtonic

seed = 42

train = False
load_model = True

if train:
    env = gym.make('gymtonic/SoccerSingle-v0', render_mode=None)
    #check_env(env, warn=True) 
    if load_model:
        model = PPO.load("policies/ppo_soccer_single", env, seed=seed, verbose=1)
    else:
        model = PPO("MlpPolicy", env, seed=seed, verbose=1)
    
    model.learn(total_timesteps=400_000, reset_num_timesteps=not load_model, progress_bar=True)
    model.save("policies/ppo_soccer_single")
    env.close()

env = gym.make('gymtonic/SoccerSingle-v0', render_mode='human')
env = Monitor(env)
model = PPO.load("policies/ppo_soccer_single", env, seed=seed, verbose=1)

avd_reward = 0

for _ in range(100):
    obs, _ = env.reset()
    terminated = truncated = False
    total_reward = 0

    while not terminated and not truncated:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

    print(f"Episode reward: {total_reward}")
    avd_reward += total_reward
print(f"Average reward: {avd_reward/100}")


# Pybullet rendering not working with evaluate_policy
#mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=100, deterministic=True, render=True)
#print(f"Mean_reward:{mean_reward:.2f}")

env.close()