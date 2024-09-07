from gymnasium import spaces
from gymtonic.envs.grid_target_v0 import GridTargetEnv
import numpy as np

class GridTargetDirectionalEnv(GridTargetEnv):

    def __init__(self, n_rows=5, n_columns=5, smooth_movement=False, render_mode=None):
        super(GridTargetDirectionalEnv, self).__init__(n_rows=n_rows, n_columns=n_columns,
                                        smooth_movement=smooth_movement, render_mode=render_mode)
        
        # Redefine the observation space        
        # Observation space just the target directional vector
        self.observation_space = spaces.Box(low=np.array([-n_columns, -n_rows]), high=np.array([n_columns, n_rows]), shape=(2,), dtype=np.int32)

    def get_observation(self):
        rel_pos_x = self.target_pos[0] - self.agent_pos[0]
        rel_pos_y = self.target_pos[1] - self.agent_pos[1]
        observation = np.array([rel_pos_x, rel_pos_y], dtype=np.int32)
        return observation