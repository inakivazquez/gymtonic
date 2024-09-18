import logging
import math

import numpy as np
import pybullet as p
from gymnasium.spaces import Box, Discrete

from gymtonic.envs.soccer_single_v0 import SoccerSingleEnv

logger = logging.getLogger(__name__)

class SoccerSingleDiscreteEnv(SoccerSingleEnv):

    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    SIMULATION_STEP_DELAY = 1.0 / 240.0

    def __init__(self, max_speed = 1, perimeter_side = 10, goal_target='right', render_mode=None, record_video_file=None):
        super(SoccerSingleDiscreteEnv, self).__init__(max_speed=max_speed, perimeter_side=perimeter_side, goal_target=goal_target, render_mode=render_mode, record_video_file=record_video_file)

        vision_length = self.perimeter_side

        self.observation_space = Box(low=np.array([-2*math.pi] + [-self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length, -vision_length]*2), high=np.array([+2*math.pi] + [self.perimeter_side/2,self.perimeter_side/2] + [+vision_length, +vision_length]*2), shape=(7,), dtype=np.float32)
        # Añadir velocidad y entonces no esperar a que se estabilice el movmiento
        self.action_space = Discrete(4)
        # Probar con acciones discretas: giro de n grados, avance de n fuerza
        # 0: Rotate right, 1: Rotate left, 2: Move forward, 3: Move backward
        #self.action_space = Discrete(5)

    def execute_action(self, action):
        action_map = {
            0: (math.pi / 12, 0),  # Action 0: small positive rotation, no force
            1: (-math.pi / 12, 0), # Action 1: small negative rotation, no force
            2: (0, 1),             # Action 2: no rotation, forward force
            3: (0, -1)             # Action 3: no rotation, backward force
        }
        action = action.item()
        rotation_offset, force = action_map[action]

        if rotation_offset != 0:
            # Rotate the player
            position,_ = p.getBasePositionAndOrientation(self.pybullet_player_id)
            angle = self.get_orientation(self.pybullet_player_id)
            angle += rotation_offset
            p.resetBasePositionAndOrientation(self.pybullet_player_id, position, p.getQuaternionFromEuler([0, 0, angle]))
        else:
            # Move the player
            force_factor = 500
            force *= force_factor * self.max_speed
            forward_force = [force, 0, 0]  # Apply force in the cube's local forward direction (X-axis)
            p.applyExternalForce(self.pybullet_player_id, -1, forward_force, [0, 0, 0], p.LINK_FRAME)        
            self.wait_for_simulation()

    def get_observation(self):
        obs = np.array([], dtype=np.float32)

        my_orientation = self.get_orientation(self.pybullet_player_id)
        obs = np.concatenate((obs, [my_orientation]), dtype=np.float32)

        my_pos, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        obs = np.concatenate((obs, my_pos[:2]), dtype=np.float32)

        # Now the ball
        ball_pos, _ = p.getBasePositionAndOrientation(self.pybullet_ball_id)
        ball_vector = self.get_oriented_distance_vector(self.pybullet_player_id, self.pybullet_ball_id)[:2]
        obs = np.concatenate((obs, ball_vector), dtype=np.float32)
        #obs = np.concatenate((obs, ball_pos[:2]), dtype=np.float32)

        goal_line_id = self.pybullet_goal_right_id if self.goal_direction == 'right' else self.pybullet_goal_left_id

        goal_line_vector = self.get_oriented_distance_vector(self.pybullet_player_id, goal_line_id)[:2]
        obs = np.concatenate((obs, goal_line_vector), dtype=np.float32)

        return obs
