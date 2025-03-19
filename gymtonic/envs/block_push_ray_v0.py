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
            <geom name="geom_cube" type="box" size="0.01 0.01 0.01" rgba="0.2 0.2 1.0 1" density="5000" friction="0.01 0.01 0.01"/>
            <joint name="cube_slide_x" type="slide" axis="1 0 0"/> <!-- Move along X -->
            <joint name="cube_slide_y" type="slide" axis="0 1 0"/> <!-- Move along Y -->
            <joint name="cube_yaw" type="hinge" axis="0 0 1"/>  <!-- Rotate around Z -->
            <body name="direction_indicator" pos="0.01 0 0">
                <geom name="indicator" type="cylinder" size="0.003 0.00001" rgba="1 1 1 1" euler="0 90 0" density="0"/>
            </body>            
        </body>
        <body name="block" pos="0 0 0.1">
            <joint type="free"/>
            <geom name="geom_block" group="1" type="box" size="0.05 0.05 0.05" rgba="0.9 0.4 0 1" density="1000" friction="0.01 0.01 0.01"/>
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


class BlockPushRay(MujocoEnv):

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

    def __init__(self, agent_speed=0.5, nrays=5, span_angle_degrees=180, **kwargs):

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

        self.nrays = nrays
        self.span_angle_degrees = span_angle_degrees

        # Observation space
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]*self.nrays + [-2]*2, dtype=np.float32), high=np.array([1, 1]*self.nrays + [+2]*2, dtype=np.float32), shape=(2*self.nrays+2,))

        # Action space
        self.action_space = gym.spaces.Box(low=np.array([-1]*2, dtype=np.float32), high=np.array([+1]*2 , dtype=np.float32), shape=(2,))

        self.agent_speed = agent_speed

        idx_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_slide_x")
        idx_y = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_slide_y")
        idx_yaw = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_yaw")
        self.cubes_components_ids = [idx_x, idx_y, idx_yaw]

        self.block_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.indicator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "direction_indicator")

    def do_simulation(self, ctrl, n_frames):
        value = super().do_simulation(ctrl, n_frames)
        if self.render_mode == "human":
            self.render()
        return value

    def reset_model(self):
        block_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.block_id]]
        self.data.qpos[block_qpos_addr:block_qpos_addr+3] = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.1]

        # As the joint is slide, the qpos is relative to the original location
        cube_0_qpos_addr = self.model.jnt_qposadr[self.model.body_jntadr[self.cube_id]]
        self.data.qpos[cube_0_qpos_addr:cube_0_qpos_addr+3] = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 0.1]

        # Necessary to call this function to update the positions before computation
        self.do_simulation(self.data.ctrl, self.frame_skip)

        self.setup_raycast(self.nrays, self.span_angle_degrees)

        self.best_distance = self.distance_xy(self.block_id, self.target_id)

        return self.get_observation()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        distance_done, _ = self.move(action[0], action[1])

        distance_agent_block = self.distance_xy(self.cube_id, self.block_id )
        distance_block_target = self.distance_xy(self.block_id, self.target_id )

        #reward = -0.1 # Time penalty
        reward = -distance_done # Distance penalty
        reward += -distance_agent_block
        reward += (self.best_distance - distance_block_target)
        if distance_block_target < self.best_distance:
            self.best_distance = distance_block_target
        
        terminated = False
        truncated = False
        if distance_block_target < 0.1 and distance_block_target <= self.best_distance:
            print("Target!")
            terminated = True
            reward = 100
        obs = self.get_observation()
        info = {}
        return obs, reward, terminated, truncated, info

    def get_observation(self):
        detected_body_ids, normalized_distances = self.perform_raycast()
        block_obs = [0, 0] * self.nrays
        for i, detected_body_id in enumerate(detected_body_ids):
            if detected_body_id == self.block_id:
                block_obs[i*2] = 1
                block_obs[i*2+1] = normalized_distances[i]
            # For debugging
            if False and detected_body_id != -1:
                body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, detected_body_id)
                print(f"Ray {i} hit {body_name} at {normalized_distances[i]}")

        
        rdv_cube_to_target, _ = self.relative_distance_vector(self.cube_id, self.target_id)

        obs = np.concatenate([block_obs, rdv_cube_to_target[0:2]], dtype=np.float32)
        return obs
    
    def single_raycast(self):
        indicator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "direction_indicator")

        # Get cube position and orientation
        cube_indicator_pos = self.data.xpos[indicator_id]  # World position of the cube
        cube_indicator_mat = self.data.xmat[indicator_id].reshape(3, 3)  # Rotation matrix (3x3)

        # Extract the forward direction from rotation matrix
        world_forward = cube_indicator_mat[:, 0]  # Column 0 is the X direction (forward)

        # Define ray start and end points
        ray_start = np.array(cube_indicator_pos, dtype=np.float64)
        ray_dir = np.array(world_forward * 10, dtype=np.float64)  # 10 meters in forward direction

        # Ensure they are (3,) shape
        ray_start = ray_start.flatten()
        ray_dir = ray_dir.flatten()

        # Optional: Define geomgroup (set to None to include all groups)
        geomgroup = np.zeros(6, dtype=np.uint8)
        geomgroup[1] = 1
        # Flag to include static geoms
        flg_static = 1  # 1 to include, 0 to exclude
        # Body ID to exclude from the test (-1 to include all)z
        bodyexclude = 0
        # Prepare geomid as a writable array
        geomid = np.full(1, -1, dtype=np.int32)

        # Perform raycast
        fraction = mujoco.mj_ray(self.model, self.data, ray_start, ray_dir, geomgroup, flg_static, bodyexclude, geomid)
        
        return geomid[0], fraction

    def setup_raycast(self, nrays, angle_covered_degrees):
        """Precomputes raycast structures (only called once)."""

        # Generate evenly spaced ray directions
        angles = np.linspace(-angle_covered_degrees / 2, angle_covered_degrees / 2, nrays)  # Spread symmetrically
        if nrays == 1:
            angles = [0]  # Single ray at angle 0

        rot_mats = np.zeros((nrays, 2, 2), dtype=np.float64)
        for i, angle in enumerate(angles):
            # Compute rotated direction using 2D rotation matrix (Z-axis rotation)
            rot_mats[i] = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])

        # Store raycast structure
        self.raycast_rot_mats = rot_mats

    def perform_raycast(self, ray_length=10):
        """Performs raycast using stored structures (called at every step)."""

        nrays = self.nrays
        rot_mats = self.raycast_rot_mats
        geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)  # Detect group 1 (block)
        flg_static = 1 # Include static geoms
        bodyexclude = 0 # Exclude no bodies

        # Get cube position dynamically (it changes over time)
        ray_start = np.array(self.data.xpos[self.indicator_id], dtype=np.float64)  # Updated position

        # Get cube position and orientation
        cube_indicator_mat = self.data.xmat[self.indicator_id].reshape(3, 3)  # Rotation matrix (3x3)

        # Forward direction (X-axis in local frame)
        world_forward = cube_indicator_mat[:, 0]  # Column 0 is the X direction

        ray_dirs = np.zeros((nrays, 3), dtype=np.float64)

        for i, rot_mat in enumerate(rot_mats):
            rotated_dir = rot_mat @ world_forward[:2]  # Apply rotation to X, Y components
            ray_dirs[i] = np.array([rotated_dir[0], rotated_dir[1], world_forward[2]]) * ray_length  # Scale rays

        # Output arrays
        geomids = np.full(nrays, -1, dtype=np.int32)  # No hit by default
        fractions = np.zeros(nrays, dtype=np.float64)

        mujoco.mj_multiRay(
            self.model,
            self.data,
            ray_start.flatten(),
            ray_dirs.flatten(),  
            geomgroup.flatten(),  
            flg_static,
            bodyexclude,
            geomids, 
            fractions,
            nrays,
            ray_length 
        )

        return geomids, fractions  # Arrays of hit geom IDs and fractions
        
    def move(self, speed, rotation, rotation_step_size=0.1):
        previous_pos = self.data.xpos[self.cube_id]
        distance_done = 0

        idx_x = self.cubes_components_ids[0]
        idx_y = self.cubes_components_ids[1]
        idx_yaw = self.cubes_components_ids[2]
        
        # Get the index of the yaw joint in qpos
        qpos_yaw = self.model.jnt_qposadr[idx_yaw]

        # Get current yaw position
        yaw_current = self.data.qpos[qpos_yaw]
        yaw_target = yaw_current + rotation

        # Determine step direction (+ or -)
        step_direction = np.sign(rotation)


        # SPEED_MODE = 0 (force constant speed during the movement
        # SPEED_MODE = 1 (do not force constant speed during the movement)
        SPEED_MODE = 1
        # Initial speed at the beginning of the movement
        current_speed = self.agent_speed * speed

        # Perform gradual rotation
        while abs(yaw_target - yaw_current) > rotation_step_size:
            yaw_current += rotation_step_size * step_direction
            self.data.qpos[qpos_yaw] = yaw_current

            # Rotate velocity vector
            vx_new = current_speed * np.cos(yaw_current)  # Forward X direction  
            vy_new = current_speed * np.sin(yaw_current)  # Forward Y direction  

            # Apply new velocities
            self.data.qvel[idx_x] = vx_new
            self.data.qvel[idx_y] = vy_new

            self.do_simulation(self.data.ctrl, self.frame_skip)

            distance_done += np.linalg.norm(self.data.xpos[self.cube_id] - previous_pos)
            previous_pos = self.data.xpos[self.cube_id]

            if SPEED_MODE == 0:
                # Reset speed to the initial value
                current_speed = self.agent_speed * speed
            else:
                # Update velocities after simulation step (they might have reduced due to friction) 
                vx = self.data.qvel[idx_x]
                vy = self.data.qvel[idx_y]
                current_speed = np.linalg.norm([vx, vy])

        # Apply final step (if any remaining rotation is < step_size)
        self.data.qpos[qpos_yaw] = yaw_target

        # Apply final speed components
        vx_new = current_speed * np.cos(yaw_target)  # Forward X direction  
        vy_new = current_speed * np.sin(yaw_target)  # Forward Y direction  
        self.data.qvel[idx_x] = vx_new
        self.data.qvel[idx_y] = vy_new
        
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # Update the final distance done
        distance_done += np.linalg.norm(self.data.xpos[self.cube_id] - previous_pos)
        return distance_done, abs(rotation)

    def _move(self, speed, rotation, rotation_step_size=0.1):
        previous_pos = self.data.xpos[self.cube_id]
        distance_done = 0

        idx_x = self.cubes_components_ids[0]
        idx_y = self.cubes_components_ids[1]
        idx_yaw = self.cubes_components_ids[2]
        
        # Get the index of the yaw joint in qpos
        qpos_yaw = self.model.jnt_qposadr[idx_yaw]

        # Get current yaw position
        yaw_current = self.data.qpos[qpos_yaw]
        yaw_target = yaw_current + rotation

        # Determine step direction (+ or -)
        step_direction = np.sign(rotation)

        # Set velocity
        current_speed = self.agent_speed * speed / 10

        # Perform gradual rotation
        while abs(yaw_target - yaw_current) > rotation_step_size:
            yaw_current += rotation_step_size * step_direction
            self.data.qpos[qpos_yaw] = yaw_current

            # Rotate velocity vector
            vx_new = current_speed * np.cos(yaw_current)  # Forward X direction  
            vy_new = current_speed * np.sin(yaw_current)  # Forward Y direction  

            # Apply new velocities
            #self.data.qvel[idx_x] = vx_new
            #self.data.qvel[idx_y] = vy_new

            self.data.ctrl[0] = vx_new
            self.data.ctrl[1] = vy_new

            self.do_simulation(self.data.ctrl, self.frame_skip)

            distance_done += np.linalg.norm(self.data.xpos[self.cube_id] - previous_pos)
            previous_pos = self.data.xpos[self.cube_id]

            # Update velocities after simulation step (they might have changed due to friction) 
            vx = self.data.qvel[idx_x]
            vy = self.data.qvel[idx_y]
            #current_speed = np.linalg.norm([vx, vy])

        # Apply final step (if any remaining rotation is < step_size)
        self.data.qpos[qpos_yaw] = yaw_target

        # Apply final speed components
        vx_new = current_speed * np.cos(yaw_target)  # Forward X direction  
        vy_new = current_speed * np.sin(yaw_target)  # Forward Y direction  
        #self.data.qvel[idx_x] = vx_new
        #self.data.qvel[idx_y] = vy_new
        self.data.ctrl[0] = vx_new
        self.data.ctrl[1] = vy_new
        
        self.do_simulation(self.data.ctrl, self.frame_skip)

        # Update the final distance done
        distance_done += np.linalg.norm(self.data.xpos[self.cube_id] - previous_pos)
        return distance_done, abs(rotation)

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