import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import gymtonic
from montecarlo import Montecarlo_FirstVisit
from common import EpsilonGreedyPolicy, evaluate_policy, max_policy

n_rows = 6
n_columns = 6

env = gym.make('gymtonic/GridTargetSimple-v0', n_rows=n_rows, n_columns=n_columns, render_mode=None)
check_env(env, warn=True, skip_render_check=False) 

model = Montecarlo_FirstVisit(env)
epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=0.1)
model.learn(epsilon_greedy_policy, n_steps=50_000, verbose=True)

env = gym.make('gymtonic/GridTargetSimple-v0', n_rows=n_rows, n_columns=n_columns, render_mode='human')
evaluate_policy(env, model.q_table, max_policy, n_episodes=10, verbose=True)
