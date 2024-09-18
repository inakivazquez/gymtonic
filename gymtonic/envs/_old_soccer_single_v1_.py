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

logger = logging.getLogger(__name__)

class SoccerSingleContinuousEnv(SoccerSingleEnv):

    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    SIMULATION_STEP_DELAY = 1.0 / 240.0
    DISCRETE_ROTATION = True

    def __init__(self, max_speed = 1, perimeter_side = 10, goal_target='right', render_mode=None, record_video_file=None):
        super(SoccerSingleContinuousEnv, self).__init__(max_speed=max_speed, perimeter_side=perimeter_side, goal_target=goal_target, render_mode=render_mode, record_video_file='videos/soccer_single_continuous.mp4')

        vision_length = self.perimeter_side

        # Create the agent
        # Observation space is rotation, and linea volicity of the agent, 
        # its position in the field, the vector to the ball, and the vector to the goal line

        self.observation_space = Box(low=np.array([-2*math.pi, -10] + [-self.perimeter_side/2,-self.perimeter_side/2] + [-vision_length, -vision_length]*2), high=np.array([+2*math.pi, 10] + [self.perimeter_side/2,self.perimeter_side/2] + [+vision_length, +vision_length]*2), shape=(8,), dtype=np.float32)

        self.action_space = Box(low=np.array([-1, -1]), high=np.array([+1, +1]), shape=(2,), dtype=np.float32)

    def execute_action(self, action):

        rotation_offset = action[0]
        force = action[1]

        # Rotate the player towards the force direction
        position, _ = p.getBasePositionAndOrientation(self.pybullet_player_id)

        #linear_velocity, angular_velocity = p.getBaseVelocity(pybullet_object_id) # Conserving inertia for the next step
        linear_velocity = angular_velocity = [0,0,0] # Starting still in next step

        angle = self.get_orientation(self.pybullet_player_id)
        if self.DISCRETE_ROTATION:
            rotation_offset = rotation_offset * math.pi / 6 # 30 degrees max rotation
            #angle += rotation_offset
            n_steps_rotation = abs(int(rotation_offset / (math.pi/360*2)))
            for i in range(n_steps_rotation):
                angle += rotation_offset/n_steps_rotation
                p.resetBasePositionAndOrientation(self.pybullet_player_id, position, p.getQuaternionFromEuler([0, 0, angle]))
                self.step_simulation()
            p.resetBaseVelocity(self.pybullet_player_id, linearVelocity=linear_velocity, angularVelocity=angular_velocity)

        self.move_player(force)        
        self.wait_for_simulation()

    def move_player(self, force):
        factor = 1000
        force *= factor * self.max_speed
        force = [force, 0, 0]
        position, orientation = p.getBasePositionAndOrientation(self.pybullet_player_id)
        # Convert quaternion to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)
        # Transform the force vector from local to world coordinates
        force_world = np.dot(rotation_matrix, force)

        p.applyExternalForce(self.pybullet_player_id, -1, force_world, position, p.WORLD_FRAME)

    def get_observation(self):
        obs = np.array([], dtype=np.float32)

        my_orientation = self.get_orientation(self.pybullet_player_id)
        obs = np.concatenate((obs, [my_orientation]), dtype=np.float32)

        # Add speed
        velocity, _ = p.getBaseVelocity(self.pybullet_player_id)
        velocity = math.sqrt(velocity[0]**2 + velocity[1]**2)
        obs = np.concatenate((obs, [velocity]), dtype=np.float32)

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
