import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat
from gymnasium import spaces
from gymtonic.envs.grid_base import GridBaseEnv
import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Any

class GridTargetEnv(GridBaseEnv):

    def __init__(self, n_rows=5, n_columns=5, smooth_movement=False, render_mode=None):
        super(GridTargetEnv, self).__init__(n_rows=n_rows, n_columns=n_columns,
                                        smooth_movement=smooth_movement, render_mode=render_mode)
        
        # Redefine the observation space        
        # Discrete observation space of the size of the board with 3 possible values (empty, agent or target)
        #self.observation_space = spaces.MultiDiscrete(np.array([self.n_rows, self.n_columns, 3], dtype=np.int32))
        self.observation_space = spaces.MultiDiscrete(np.array([3] * self.n_rows * self.n_columns, dtype=np.int32))
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())        
        self.pybullet_target_id = p.loadURDF("duck_vhacd.urdf", [0, 0, 0], useFixedBase=False, globalScaling=7, baseOrientation=[0.5, 0.5, 0.5, 0.5])
        # Mass 0 for the target to avoid collision physics
        p.changeDynamics(self.pybullet_target_id, -1, mass=0)

        # Fixed target position
        self.target_pos = np.array([self.n_rows-1, self.n_columns-1], dtype=np.int32)

    def update_visual_objects(self, smooth=True):
        super().update_visual_objects(smooth=smooth)
        self.rotate_target_a_little()


    def rotate_target_a_little(self):
        # Rotate the target a little bit each time for a nice effect
        _, orientation = p.getBasePositionAndOrientation(self.pybullet_target_id)
        euler_angles = p.getEulerFromQuaternion(orientation)
        euler_angles = [euler_angles[0], euler_angles[1], euler_angles[2] - 0.15]
        quaternion = p.getQuaternionFromEuler(euler_angles)
        p.resetBasePositionAndOrientation(self.pybullet_target_id,[self.target_pos[0], self.target_pos[1], self.floor_height + 0.1], quaternion)
        self.render()


    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)

        if self.is_target_reached():
            terminated = True
            print("Target reached!")

        return obs, reward, terminated, truncated, info
    
    def calculate_reward(self):
        if self.is_target_reached():
            reward = 1
        else:
            reward = 0
        return reward

    def get_observation(self):
        observation = np.zeros((self.n_rows, self.n_columns), dtype=np.int32)
        observation[self.agent_pos[0], self.agent_pos[1]] = 1
        observation[self.target_pos[0], self.target_pos[1]] = 2
        return observation.flatten()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0], dtype=np.int32)

        # Fixed target position
        self.target_pos = np.array([self.n_rows-1, self.n_columns-1], dtype=np.int32)
        
        # Random target position
        x_random = np.random.randint(0, self.n_columns)
        y_random = np.random.randint(0, self.n_rows)
        self.target_pos = [x_random, y_random]

        if self.render_mode == 'human':
            self.update_visual_objects()
        
        obs = self.get_observation()
        info = {}
        return obs, info
    
    def is_target_reached(self):
        return np.array_equal(self.agent_pos, self.target_pos)

# Example usage
if __name__ == "__main__":
    render_model = 'human'
    #env = GridTargetEnv(n_rows=6, n_columns=10, smooth_movement=True,  render_mode=render_model)
    env = gym.make('gymtonic/GridTarget-v0', render_mode='human')
    obs, info = env.reset()
    n_episodes = 10
    for _ in range(n_episodes):
        for _ in range(100): # Maximum 100 steps
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            if render_model == 'human':
                time.sleep(0.1)
            if terminated or truncated:
                break

        obs, info = env.reset()    
    
    env.close()
