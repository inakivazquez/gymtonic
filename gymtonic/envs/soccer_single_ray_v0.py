import logging
import math
import time

import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import Env
from gymnasium.spaces import Box
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from typing import Any

from gymtonic.envs.soccer_stadium import create_stadium, create_player, create_ball
from gymtonic.envs.soccer_single_v0 import SoccerSingleEnv
from gymtonic.envs.raycast import raycast_horizontal_detect

logger = logging.getLogger(__name__)

class SoccerSingleRaycastEnv(SoccerSingleEnv):

    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    SIMULATION_STEP_DELAY = 1.0 / 240.0
    DISCRETE_ROTATION = True

    def __init__(self, max_speed = 1, perimeter_side = 10, goal_target='right', render_mode=None, record_video_file=None):
        super(SoccerSingleRaycastEnv, self).__init__(max_speed=max_speed, perimeter_side=perimeter_side, goal_target=goal_target, render_mode=render_mode, record_video_file=record_video_file)

        vision_length = self.perimeter_side

        self.raycast_vision_length = self.perimeter_side*2
        raycast_len = 5 # Five components per raycast: one-hot for ball, goal-right, goal-left or wall, and distance
        self.n_raycasts = 5
        self.raycast_cover_angle = 2*math.pi/12

        # Create the agents
        # Observation space is
        # rotation and position of the agent, 
        # vector to the ball
        self.observation_space=Box(low=np.array([0,0,0,0, -self.raycast_vision_length]*self.n_raycasts), high=np.array([1,1,1,1, self.raycast_vision_length]*self.n_raycasts), shape=(raycast_len*self.n_raycasts,), dtype=np.float32)

        #self.observation_space = Box(low=np.array([-2*math.pi] + [-self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length, -vision_length]), high=np.array([+2*math.pi] + [self.perimeter_side/2,self.perimeter_side/2] + [+vision_length, +vision_length]), shape=(5,), dtype=np.float32)
        #self.observation_space = Box(low=np.array([-2*math.pi] + [-self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length, -vision_length]*2), high=np.array([+2*math.pi] + [self.perimeter_side/2,self.perimeter_side/2] + [+vision_length, +vision_length]*2), shape=(7,), dtype=np.float32)


    def get_observation(self):
        obs = np.array([])
        my_pos,_ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        #obs = np.append(obs, my_pos)

        my_angle = self.get_orientation(self.pybullet_player_id)

        raycast_data = self.raycast_detect_objects(my_pos, my_angle, covering_angle=self.raycast_cover_angle)
        obs = np.concatenate((obs, raycast_data), dtype=np.float32)
        return obs

    def wait_until_stable(self, sim_steps=500):
        super().wait_until_stable(sim_steps)
        # This is the visual optimal point for raycast removal
        if self.render_mode == 'human':
            p.removeAllUserDebugItems()


    def raycast_detect_objects(self, source_pos_x_y, source_angle_z, covering_angle=2*math.pi):
        source_pos = np.array([source_pos_x_y[0], source_pos_x_y[1], 0.5])
        detections = raycast_horizontal_detect(source_pos, source_angle_z, n_raycasts=self.n_raycasts, covering_angle=covering_angle, vision_length=self.raycast_vision_length, draw_lines=self.render_mode)
        results = np.array([])

        for object_id, distance in detections:
            if object_id == self.pybullet_ball_id:
                one_hot_type = [1, 0, 0, 0]  # Type 0 (ball)
            elif object_id == self.pybullet_goal_right_id:
                one_hot_type = [0, 1, 0, 0]  # Type 1 (goal-right)
            elif object_id == self.pybullet_goal_left_id:
                one_hot_type = [0, 0, 1, 0]  # Type 2 (goal-left)
            elif object_id in self.pybullet_wall_ids:
                one_hot_type = [0, 0, 0, 1] # Type 3 (wall)
            else:
                one_hot_type = [0, 0, 0, 0]

            results = np.append(results, one_hot_type + [distance])
        return results
