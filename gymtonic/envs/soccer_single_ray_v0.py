import logging
import math

import numpy as np
import pybullet as p
from gymnasium.spaces import Box

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
        self.n_raycasts = 11
        self.raycast_cover_angle = 2*math.pi/3

        # Create the agents
        # Observation space is
        # rotation and velocity of the agent, 
        # raycast info
        self.observation_space=Box(low=np.array([-2*math.pi, -10] + [0,0,0,0, -self.raycast_vision_length]*self.n_raycasts), high=np.array([+2*math.pi, +10] + [1,1,1,1, self.raycast_vision_length]*self.n_raycasts), shape=(2 + raycast_len*self.n_raycasts,), dtype=np.float32)


    def get_observation(self):
        obs = np.array([])
        
        my_pos,_ = p.getBasePositionAndOrientation(self.pybullet_player_id)
        my_angle = self.get_orientation(self.pybullet_player_id)

        # Add angle
        obs = np.concatenate((obs, [my_angle]), dtype=np.float32)

        # Add speed
        velocity, _ = p.getBaseVelocity(self.pybullet_player_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        obs = np.concatenate((obs, [velocity]), dtype=np.float32)

        # Add raycasts
        raycast_data = self.raycast_detect_objects(my_pos, my_angle, covering_angle=self.raycast_cover_angle)
        obs = np.concatenate((obs, raycast_data), dtype=np.float32)
        return obs

    def wait_for_simulation(self, sim_steps=500):
        if self.render_mode == 'human':
            p.removeAllUserDebugItems()
        super().wait_for_simulation(sim_steps)
        # This is the visual optimal point for raycast removal


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

