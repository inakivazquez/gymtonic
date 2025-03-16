from __future__ import annotations
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from gymnasium.envs.mujoco import MujocoEnv
import mujoco
import numpy as np
import random
import os

# Generate XML for MuJoCo
def generate_mujoco_xml():
    xml = f"""<mujoco model="swarm_cubes">
    <option timestep="0.01" gravity="0 0 -9.81"/>
    <visual>
        <headlight diffuse="0.8 0.8 0.8" ambient="0.8 0.8 0.8" specular="0 0 0"/>
        <rgba haze="0.15 0.25 0.35 1"/>
        <global azimuth="120" elevation="-20"/>
    </visual>
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"
        markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.05"/>
    </asset>
    <worldbody>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" friction="0.01 0.01 0.01"/>
        <body name="cube" pos="0 0 0.01">
            <geom name="geom_cube" type="box" size="0.01 0.01 0.01" rgba="0.2 0.2 1.0 1" density="1000" friction="0.01 0.01 0.01"/>
            <joint name="cube_slide_x" type="slide" axis="1 0 0"/> <!-- Move along X -->
            <joint name="cube_slide_y" type="slide" axis="0 1 0"/> <!-- Move along Y -->
            <joint name="cube_yaw" type="hinge" axis="0 0 1"/>  <!-- Rotate around Z -->
            <body name="direction_indicator" pos="0.01 0 0">
                <geom name="indicator" type="cylinder" size="0.003 0.00001" rgba="1 1 1 1" euler="0 90 0" density="0"/>
            </body>            
        </body>
        <body name="block" pos="0 0 0.1">
            <joint type="free"/>
            <geom name="geom_block" type="box" size="0.05 0.05 0.05" rgba="0.9 0.4 0 1" density="1000" friction="0.01 0.01 0.01"/>
        </body>
        <body name="target" pos="0 0 0.1">
            <geom name="geom_target" type="cylinder" size="0.15 0.1" rgba="0.0 0.8 0.0 0.4" density="0" contype="0" conaffinity="0" />
        </body>
    </worldbody>
    <actuator>
        <general name="actuator_cube_x" joint="cube_slide_x"/>
        <general name="actuator_cube_y" joint="cube_slide_y"/>
    </actuator>
    </mujoco>"""
    
    return xml


class BlockPush(MujocoEnv):

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array", "rgbd_tuple",],
    }
    
    # Overriden initialize_simulation function will use this XML string to create the model instead of model_path
    def _initialize_simulation(self,):
        """
        Initialize MuJoCo simulation data structures `mjModel` and `mjData`.
        """
        model = mujoco.MjModel.from_xml_string(self.xml_model) # This line is the difference
        model.vis.global_.offwidth = self.width
        model.vis.global_.offheight = self.height
        data = mujoco.MjData(model)
        return model, data

    def __init__(self, **kwargs):

        default_camera_config = {
            "distance": 2.5,
            "elevation": -10.0,
            "azimuth": 90.0,
            "lookat": [0.0, 0.0, 0.0],
        }

        screen_width = screen_height = 800

        # Overriden initialize_simulation function will use this XML string to create the model instead of model_path 
        self.xml_model = generate_mujoco_xml()

        MujocoEnv.__init__(
            self,
            model_path=os.path.abspath(__file__), # Dummy value, not used, but it must be a valid path
            frame_skip=5,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=screen_width,
            height=screen_height,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        # Create a data structure to hold the simulation state
        #self.data = mujoco.MjData(self.model)

        # Target pos, current pos of ee, joint positions
        self.observation_space = gym.spaces.Box(low=np.array([-2]*4 + [-np.pi], dtype=np.float32), high=np.array([+2]*4 + [+np.pi] , dtype=np.float32), shape=(5,))

        # Action space is 6 joint angles and the gripper open/close level (0 to 1)
        self.action_space = gym.spaces.Box(low=np.array([-1]*2, dtype=np.float32), high=np.array([+1]*2 , dtype=np.float32), shape=(2,))

        self.cubes_components_ids = []
        idx_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_slide_x")
        idx_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_slide_y")
        idx_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_yaw")
        self.cubes_components_ids.append((idx_x, idx_y, idx_yaw))

        self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")

       
    def reset_model(self):
        block_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.block_id]]
        self.data.qpos[block_qpos_addr:block_qpos_addr+3] = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.1]

        # As the joint is slide, the qpos is relative to the original location
        cube_0_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.cube_id]]
        self.data.qpos[cube_0_qpos_addr:cube_0_qpos_addr+3] = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.1]

        # Necessary to call this function to update the positions before computation
        self.do_simulation(self.data.ctrl, self.frame_skip)

        #target_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.target_id]]
        #self.data.qpos[target_qpos_addr:target_qpos_addr+3] = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.000001]

        self.initial_distance = self.distance_xy(self.block_id, self.target_id)

        return self.get_observation()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        self.rotate(0, action[0])
        #self.do_simulation(self.data.ctrl, self.frame_skip)
        self.move(0, action[1])
        self.do_simulation(self.data.ctrl, self.frame_skip)

        distance_agent_block = self.distance_xy(self.cube_id, self.block_id )
        distance_block_target = self.distance_xy(self.block_id, self.target_id )

        reward = -0.1 # Time penalty
        reward += -distance_agent_block
        reward += (self.initial_distance - distance_block_target)
        
        terminated = False
        truncated = False
        if distance_block_target < 0.1 and distance_block_target < self.initial_distance:
            print("Target!")
            terminated = True
            reward = 100
        obs = self.get_observation()
        info = {}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        rdv_cube_to_block, yaw_diff = self.relative_distance_vector(self.cube_id, self.block_id)
        rdv_block_to_target, _ = self.relative_distance_vector(self.cube_id, self.target_id)
        obs = np.concatenate([rdv_cube_to_block[0:2], rdv_block_to_target[0:2], [yaw_diff]], dtype=np.float32)
        #print(obs)
        return obs
    
    def rotate(self, cube_id, value=1):
        idx_x = self.cubes_components_ids[cube_id][0]
        idx_y = self.cubes_components_ids[cube_id][1]
        idx_yaw = self.cubes_components_ids[cube_id][2]
        self.data.ctrl[cube_id * 2] = 0  # Stop X movement
        self.data.ctrl[cube_id * 2 + 1] = 0  # Stop Y movement     
        self.data.qvel[idx_x] = 0  # Stop X velocity
        self.data.qvel[idx_y] = 0  # Stop Y velocity   
        self.data.qvel[idx_yaw] = 5*value

    def move(self, cube_id, value=1):
        idx_yaw = self.cubes_components_ids[cube_id][2]
        idx_x = self.cubes_components_ids[cube_id][0]
        idx_y = self.cubes_components_ids[cube_id][1]
        # Get the cube’s quaternion (assuming cube_id is the index of the cube)
        quat = self.data.xquat[self.model.body("cube").id]
        # Convert quaternion to yaw angle
        yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),  
                        1 - 2 * (quat[2]**2 + quat[3]**2))
        # Define speed magnitude
        speed_magnitude = 2 * value
        # Compute speed components
        vx = speed_magnitude * np.cos(yaw)  # Forward X direction  
        vy = speed_magnitude * np.sin(yaw)  # Forward Y direction  

        #self.data.qvel[idx_yaw] = 0 # Stop rotation
        self.data.qvel[idx_x] = vx  #  X velocity
        self.data.qvel[idx_y] = vy  #  Y velocity   

    def distance_xy(self, body_id_1, body_id_2):
        """
        Compute the Euclidean distance between two objects in the X-Y plane.
        
        Args:
            body_id_1: The ID of the first body.
            body_id_2: The ID of the second body.
        
        Returns:
            The Euclidean distance between the two objects in the X-Y plane.
        """

        # Get positions
        pos1 = self.data.xpos[body_id_1]
        pos2 = self.data.xpos[body_id_2]

        distance = np.linalg.norm(pos1[0:2] - pos2[0:2])
        return distance

    def relative_distance_vector(self, body_id_1, body_id_2):
        """
        Compute the distance vector from object 1 to object 2 in object 1's local frame,
        and also return the difference in yaw (orientation) between the two bodies.
        
        Args:
            body_id_1: The ID of the first body (reference).
            body_id_2: The ID of the second body.
        
        Returns:
            A tuple: 
                - A NumPy array representing the distance vector in object 1's local frame.
                - A float representing the difference in yaw (orientation) between the two bodies.
        """
        if body_id_1 == -1 or body_id_2 == -1:
            raise ValueError("Invalid body IDs. Ensure both objects exist in the model.")

        # Get positions
        pos1 = self.data.xpos[body_id_1]  # (x, y, z) position of Object 1
        pos2 = self.data.xpos[body_id_2]  # (x, y, z) position of Object 2

        # Compute global distance vector
        distance_vector = pos2 - pos1  # (dx, dy, dz)

        # Get quaternion of Object 1
        quat1 = self.data.xquat[body_id_1]
        # Compute yaw angle for body 1
        yaw1 = np.arctan2(2 * (quat1[0] * quat1[3] + quat1[1] * quat1[2]),  
                        1 - 2 * (quat1[2]**2 + quat1[3]**2))

        # Get quaternion of Object 2
        quat2 = self.data.xquat[body_id_2]
        # Compute yaw angle for body 2
        yaw2 = np.arctan2(2 * (quat2[0] * quat2[3] + quat2[1] * quat2[2]),  
                        1 - 2 * (quat2[2]**2 + quat2[3]**2))

        # Compute the rotation matrix for body 1's yaw
        rot_matrix1 = np.array([
            [np.cos(-yaw1), -np.sin(-yaw1), 0],  # Rotate in the opposite direction
            [np.sin(-yaw1),  np.cos(-yaw1), 0],
            [0,             0,            1]  # Z-axis remains unchanged
        ])

        # Transform distance vector into Object 1's local frame
        local_distance_vector = rot_matrix1 @ distance_vector

        # Compute the yaw difference between body 1 and body 2
        yaw_difference = yaw2 - yaw1
        # Normalize the yaw difference to be between -pi and pi
        yaw_difference = (yaw_difference + np.pi) % (2 * np.pi) - np.pi

        return local_distance_vector, yaw_difference

import time
if __name__ == "__main__":
    env = BlockPush(render_mode="human")
    #env = gym.make("mujobot/ur5-paddle-v1", render_mode="human")
    for _ in range(10000):
        print("Resetting")
        obs = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
    env.close()