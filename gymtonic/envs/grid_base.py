from gymnasium import Env
from gymnasium.core import ObsType, ActType, SupportsFloat, RenderFrame
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from typing import Any

class GridBaseEnv(Env):
    """
    Custom Gymnasium environment using PyBullet representing a n_rows (length of axis y) x n_columns(length of axis x) grid.
    The agent can move in four directions: north, south, east, west.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}
    
    def __init__(self, n_rows=5, n_columns=5, smooth_movement = False, render_mode=None):
        """
            Initializes a GridBaseEnv.
            Parameters:
            - n_rows (int): Number of rows in the grid. Default is 5.
            - n_columns (int): Number of columns in the grid. Default is 5.
            - smooth_movement (bool): Flag indicating whether smooth movement is enabled (otherwise discrete jumpings). Default is False.
            - render_mode (str): Rendering mode for visualization. Options are 'human' or None. Default is None.
        """

        super(GridBaseEnv, self).__init__()
        
        self.n_rows = n_rows  # Number of rows
        self.n_columns = n_columns  # Number of columns

        self.smooth_movement = smooth_movement
        self.render_mode = render_mode
        
        # Action space: 0 = North, 1 = South, 2 = East, 3 = West
        self.action_space = spaces.Discrete(4)
                
        # Discrete observation space of the size of the grid with 2 possible values (empty or agent)
        self.observation_space = spaces.MultiDiscrete(np.array([2] * self.n_rows * self.n_columns, dtype=np.int32))
        
        # Initialize agent's position at 0,0
        self.agent_pos = np.array([0, 0], dtype=np.int32)
        
        if self.render_mode == 'human':
            p.connect(p.GUI)
            # For recording a video, use the following line instead
            #p.connect(p.GUI, options=f"--mp4='grid_target.mp4'")

            # Hide debug panels in PyBullet
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)

            # For the agent to be at the center of the grid cell (each cell is 1x1)
            self.visual_offset = 0.5

            # Reorient the camera
            p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=0, cameraPitch=-60, cameraTargetPosition=[self.n_columns/2 - self.visual_offset ,0,0])
            p.setGravity(0, 0, -9.81)

            # Load the plane and objects from the pybullet_data directory
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.pybullet_plane_id = p.loadURDF("plane.urdf", [self.visual_offset,self.visual_offset,0])

            # Draw the floor for the grid
            self.floor_height = 0.2
            self.draw_floor()

            # Draw the agent
            self.agent_scale = 0.5
            self.pybullet_agent_id = p.loadURDF("cube.urdf", [self.agent_pos[0], self.agent_pos[1], self.floor_height + self.agent_scale/2], useFixedBase=False, globalScaling=self.agent_scale)
            p.changeVisualShape(self.pybullet_agent_id, -1, rgbaColor=[0.1, 0.1, 0.8, 1])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        
        if self.render_mode == 'human':
             # For resetting position, we don't want smooth movement in any case
            self.update_visual_objects(force_teletransport=True)
        
        obs = self.get_observation()
        info = {}
        return obs, info
        
    def update_visual_objects(self, force_teletransport = False):
        """
        Reresh the visual objects in the environment.
        Parameters:
            smooth_movement (bool, optional): Whether to smoothly move the agent or not. Defaults to True.
        """

        # If smooth_movement is enabled and smooth is True, move the agent smoothly
        # Otherwise, reset the agent's position and orientation (teletransportation)
        # Render the environment
        if self.smooth_movement and not force_teletransport:
            # Move the agent smoothly
            self.move_agent_smoothly(self.agent_pos[0], self.agent_pos[1])
        else:
            # Teletransport the agent
            p.resetBasePositionAndOrientation(self.pybullet_agent_id, 
                [self.agent_pos[0], self.agent_pos[1], self.floor_height + self.agent_scale/2],
                [0, 0, 0, 1])
        #self.render()
        

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid Action: {action}"

        # Define movement vectors
        movement = {
            0: np.array([0, 1]),  # North
            1: np.array([0, -1]),   # South
            2: np.array([1, 0]),   # East
            3: np.array([-1, 0])   # West
        }
        
        # Update agent's position
        new_pos = self.agent_pos + movement[action]
        
        # Check boundaries
        new_pos = np.clip(new_pos, [0, 0], [self.n_columns-1, self.n_rows-1])
        
        self.agent_pos = new_pos
        
        # Update visualization
        if self.render_mode == 'human':
            self.update_visual_objects()
        
        reward = self.calculate_reward()       
        obs = self.get_observation()
        terminated = False
        truncated = False
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def get_observation(self):
        observation = np.zeros((self.n_columns, self.n_rows), dtype=np.int32)
        observation[self.agent_pos[0], self.agent_pos[1]] = 1
        return observation.flatten()
    
    def calculate_reward(self):
        # No reward in this basic environment
        reward = 0
        return reward

    def move_agent_smoothly(self, target_x, target_y, step_size=0.01, sleep_time=0.002):
        """
        Moves the agent smoothly towards the target position.
        Parameters:
            target_x (float): The x-coordinate of the target position.
            target_y (float): The y-coordinate of the target position.
            step_size (float, optional): The step size for each movement. Defaults to 0.01.
            sleep_time (float, optional): The sleep time between each movement. Defaults to 0.002.
        """

        # Get the current position and orientation of the object
        current_pos, current_orientation = p.getBasePositionAndOrientation(self.pybullet_agent_id)
        current_orientation = [0,0,0,1] # Always keep the same orientation

        # Extract current (x, y, z) position
        current_x, current_y, current_z = current_pos

        # Calculate the direction vector to the target position
        direction = np.array([target_x - current_x, target_y - current_y])
        distance = np.linalg.norm(direction)

        # Normalize the direction vector
        direction_normalized = direction / distance if distance > 0 else np.array([0, 0])

        # Incrementally move the object towards the target position
        while distance > step_size:
            # Compute the new position
            current_x += step_size * direction_normalized[0]
            current_y += step_size * direction_normalized[1]
            
            # Update the object's position in PyBullet
            p.resetBasePositionAndOrientation(self.pybullet_agent_id, [current_x, current_y, current_z], current_orientation)
            
            # Recalculate distance to the target
            distance = np.linalg.norm([target_x - current_x, target_y - current_y])
            
            # Pause for visualization purposes
            time.sleep(sleep_time)
            p.stepSimulation()

        # Final update to ensure the object reaches exactly the target position
        p.resetBasePositionAndOrientation(self.pybullet_agent_id, [target_x, target_y, current_z], current_orientation)
        p.stepSimulation()

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode == 'human':
            p.stepSimulation()
    
    def close(self):
        if self.render_mode == 'human':
            p.disconnect()

    def draw_floor(self):
        """
        Draws a floor in the environment.
        This function creates the a floor using the specified dimensions and color.
        The bottom-left corner is positioned at coors (0,0).
        Parameters:
        - self: The instance of the GridBase class.
        Returns:
        - None
        """

        half_height = self.floor_height / 2.0
        pos_x = self.n_columns / 2 - 1
        pos_y = self.n_rows / 2 - 1
        color = [0, 0.4, 0, 0.8]  # Green color

        # Create visual shape for the floow
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.n_columns / 2, self.n_rows / 2, half_height],
            rgbaColor=color,
            specularColor=[0, 0, 0]
        )

        # Create collision shape for the box
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[self.n_columns / 2, self.n_rows / 2, half_height]
        )

        # Create the multi-body object
        perimeter_id = p.createMultiBody(
            baseMass=0,  # Static object
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=[pos_x + self.visual_offset, pos_y + self.visual_offset, half_height]
        )



# Example usage
if __name__ == "__main__":
    env = GridBaseEnv(n_rows=6, n_columns=10, render_mode='human')
    obs, info = env.reset()
    
    for _ in range(100):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        if env.render_mode == 'human':
            time.sleep(0.5)
        if terminated or truncated:
            break
    
    env.close()
