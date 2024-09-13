import math
import os
import numpy as np
import pybullet as p

def create_stadium(side_length):
    """
    Creates a soccer stadium in PyBullet.

    Args:
        side_length (float): The length of the longer side of the stadium, the other will be half this length.

    Returns:
        tuple: A tuple containing the PyBullet goal ids for the right and left goals.
    """
    pass
    base_height = 0.4
    _create_turf(side_length, base_height)
    pybullet_wall_ids = _create_perimeter(side_length, thickness=0.1, height=0.4, base_height=base_height, color=[0.7, 0.7, 0.7, 1])
    # Add the goals
    pybullet_goal_right_id = _create_goal([side_length/2 - 0.10, 0, base_height], p.getQuaternionFromEuler([0, 0, 0]))
    pybullet_goal_left_id = _create_goal([-side_length/2 + 0.10, 0, base_height], p.getQuaternionFromEuler([0, 0, math.pi]))

    return pybullet_goal_right_id, pybullet_goal_left_id, pybullet_wall_ids

def create_player(position = [0,0,0], color = [0.8, 0.1, 0.1, 1]):
    player_id = p.loadURDF("cube.urdf", position, useFixedBase=False, globalScaling=0.4)
    p.changeVisualShape(player_id, -1, rgbaColor=color)
    p.changeVisualShape(player_id, linkIndex=1, rgbaColor=[0,0,0,1])
    p.changeDynamics(player_id, -1, restitution=0.8) # Bounciness
    p.changeDynamics(player_id, -1, mass=1, lateralFriction=1, rollingFriction=1, spinningFriction=1)

    return player_id

def _create_player(position = [0,0,0], color = [0.8, 0.1, 0.1, 1]):
    position = [0, 0, 0.5]
    cube_size = 0.4
    cube_collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size/2] * 3)
    cube_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size/2] * 3, rgbaColor=color)
    player_id = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=cube_collision_id,
            baseVisualShapeIndex=cube_visual_id,
            basePosition=position)
    
    small_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[cube_size/10] * 3)
    small_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[cube_size/10] * 3, rgbaColor=[1, 1, 1, 1])

    # Add another link with a joint connecting it to the base
    link_id = p.createMultiBody(
        baseMass=0.0001,
        baseCollisionShapeIndex=small_collision_shape_id,
        baseVisualShapeIndex=small_visual_shape_id,
        basePosition=[0, 0, cube_size]  # Position on top of the base link
    )

    # Add a joint connecting the two links
    joint_id = p.createConstraint(
        parentBodyUniqueId=player_id,
        parentLinkIndex=-1,
        childBodyUniqueId=link_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 1],
        parentFramePosition=[cube_size/2, 0, 0],
        childFramePosition=[0, 0, -cube_size/10]
    )

    p.changeVisualShape(player_id, -1, rgbaColor=color)
    p.changeVisualShape(player_id, linkIndex=1, rgbaColor=[0,0,0,1])
    p.changeDynamics(player_id, -1, restitution=0.8) # Bounciness
    p.changeDynamics(player_id, -1, mass=1, lateralFriction=1, rollingFriction=1, spinningFriction=1)

    return player_id



def create_ball(position):
        pybullet_ball_id = p.loadURDF("soccerball.urdf", position, useFixedBase=False, globalScaling=0.3)

        lateral_friction = 1.0
        spinning_friction = 0.05
        rolling_friction = 0.05
        p.changeDynamics(bodyUniqueId=pybullet_ball_id, linkIndex=-1,
                         lateralFriction=lateral_friction, spinningFriction=spinning_friction, rollingFriction=rolling_friction, 
                         mass = 0.1, restitution=0.8) # Restitution is the bounciness of the ball
        return pybullet_ball_id


def _create_turf(side_length, height = 0.4):
    half_size = side_length / 2.0 + 0.8 # Add some extra space around the field
    half_height = height / 2.0
    color = [0, 0.4, 0, 0.9]  # Green color

    # Create visual shape for the box
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half_size, half_size/2, half_height],
        rgbaColor=color,
        specularColor=[0, 0, 0]
    )

    # Create collision shape for the box
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half_size, half_size/2, half_height]
    )

    # Create the multi-body object
    turf_id = p.createMultiBody(
        baseMass=0,  # Static object
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[0, 0, half_height]
    )

    p.changeDynamics(turf_id, -1, lateralFriction=0.5)
    return turf_id


def _create_perimeter(length, thickness, height, base_height, color):
    """
    Creates a rectangular structure with walls, with gaps in the center of the shorter sides.

    Args:
        length (float): The length of the rectangle (double the width).
        width (float): The width of the rectangle (half the length).
        thickness (float): The thickness of the walls.
        height (float): The height of the walls.
        base_height (float): The base height at which the walls are positioned.
        color (tuple): A tuple representing the color of the walls in RGB format (r, g, b).
    """
    # Define half dimensions for easier calculations
    width = length / 2
    half_length = length / 2
    half_width = width / 2
    half_thickness = thickness / 2
    half_height = height / 2
    gap_size = width * 0.40
    gap_size = 2 # Length of the goal
    segment_length = (width - gap_size) / 2
    corner_radius = 1

    # Create collision shape for wall segments
    long_wall_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[half_length - corner_radius, half_thickness, half_height])
    short_wall_shape = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[segment_length / 2 - corner_radius/2, half_thickness, half_height])

    # Define the visual shape for the walls
    long_wall_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                        halfExtents=[half_length-corner_radius, half_thickness, half_height],
                                        rgbaColor=[color[0], color[1], color[2], 1])
    short_wall_visual = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=[segment_length / 2-corner_radius/2, half_thickness, half_height],
                                            rgbaColor=[color[0], color[1], color[2], 1])

    # Position and orientation for the walls
    positions = [
        [0, -half_width + half_thickness, base_height + half_height],   # Front wall
        [0, +half_width - half_thickness, base_height + half_height],  # Back wall
        [half_length - half_thickness, half_width - segment_length / 2 - corner_radius/2, base_height + half_height],  # Right wall (segment 1)
        [half_length - half_thickness, -half_width + segment_length / 2 + corner_radius/2, base_height + half_height],  # Right wall (segment 2)
        [-half_length + half_thickness, half_width - segment_length / 2 - corner_radius/2, base_height + half_height],  # Left wall (segment 1)
        [-half_length + half_thickness, -half_width + segment_length / 2 + corner_radius/2, base_height + half_height]  # Left wall (segment 2)
    ]
    orientations = [
        p.getQuaternionFromEuler([0, 0, 0]),                    # Front wall
        p.getQuaternionFromEuler([0, 0, 0]),                    # Back wall
        p.getQuaternionFromEuler([0, 0, 1.5708]),  # Right wall (90 degrees rotation around Z-axis)
        p.getQuaternionFromEuler([0, 0, 1.5708]),  # Right wall (90 degrees rotation around Z-axis)
        p.getQuaternionFromEuler([0, 0, 1.5708]),  # Left wall (90 degrees rotation around Z-axis)
        p.getQuaternionFromEuler([0, 0, 1.5708])   # Left wall (90 degrees rotation around Z-axis)
    ]

    # Create walls
    pybullet_wall_ids = []

    # Create walls
    # Long walls
    for i in range(2):
        id = p.createMultiBody(baseMass=0,
                        baseCollisionShapeIndex=long_wall_shape,
                        baseVisualShapeIndex=long_wall_visual,
                        basePosition=positions[i],
                        baseOrientation=orientations[i])
        #p.setCollisionFilterGroupMask(id, -1, 2, 1)
        p.changeDynamics(id, -1, restitution=0.8) # Bounciness
        pybullet_wall_ids.append(id)

    # Short walls with gaps
    for i in range(2, 6):
        id = p.createMultiBody(baseMass=0,
                        baseCollisionShapeIndex=short_wall_shape,
                        baseVisualShapeIndex=short_wall_visual,
                        basePosition=positions[i],
                        baseOrientation=orientations[i])
        #p.setCollisionFilterGroupMask(id, -1, 2, 1)
        p.changeDynamics(id, -1, restitution=0.8) # Bounciness
        pybullet_wall_ids.append(id)

    # Add the 4 corners
    pybullet_corners_1 = _create_curved_corner([half_length - half_thickness - corner_radius, half_width - half_thickness - corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=0, color=color)
    pybullet_corners_2 =_create_curved_corner([half_length - half_thickness - corner_radius, -half_width + half_thickness + corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=-math.pi/2, color=color)
    pybullet_corners_3 =_create_curved_corner([-half_length + half_thickness + corner_radius, half_width - half_thickness - corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=math.pi/2, color=color)
    pybullet_corners_4 =_create_curved_corner([-half_length + half_thickness + corner_radius, -half_width + half_thickness + corner_radius, base_height + half_height], 2*half_height, radius=1, orientation=math.pi, color=color)

    pybullet_wall_ids.extend(pybullet_corners_1)
    pybullet_wall_ids.extend(pybullet_corners_2)
    pybullet_wall_ids.extend(pybullet_corners_3)
    pybullet_wall_ids.extend(pybullet_corners_4)

    return pybullet_wall_ids


def _create_curved_corner(position, height, radius=1, orientation=0, color = [0,0,0,1], num_segments=30):
    """
    Create a 90-degree circular corner with a given position, radius, number of segments, and orientation.

    Parameters:
    - position: (x, y, z) coordinates where the corner's center should be placed
    - radius: Radius of the circular corner (default = 1)
    - num_segments: Number of cylinder segments to approximate the curve (default = 10)
    - orientation: Orientation of the corner (0, π/2, π, 3π/2) in radians (default = 0)
    """
    angle_step = np.pi / 2 / num_segments  # Divide 90 degrees into segments
    
    pybullet_object_ids = []

    # Loop to create cylinder segments
    for i in range(num_segments):
        # Calculate the angle for each segment
        angle = i * angle_step
        # Apply the orientation by rotating the entire corner
        x = position[0] + radius * np.cos(angle + orientation)
        y = position[1] + radius * np.sin(angle + orientation)
        z = position[2]
        
        # Create a small cylinder segment to approximate the curve
        cylinder_collision_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=height)
        # Apply the same orientation offset to the segment's local orientation
        cylinder_id = p.createMultiBody(baseCollisionShapeIndex=cylinder_collision_id,
                        basePosition=[x, y, z],
                        baseOrientation=p.getQuaternionFromEuler([0, 0, angle + orientation + np.pi / 2]))
        p.changeVisualShape(cylinder_id, -1, rgbaColor=color)
        #p.setCollisionFilterGroupMask(cylinder_id, -1, 2, 1)
        p.changeDynamics(cylinder_id, -1, restitution=0.8) # Bounciness
        pybullet_object_ids.append(cylinder_id)

    return pybullet_object_ids


def _create_goal(position, orientation):
    script_dir = os.path.dirname(__file__)
    goal_path = os.path.join(script_dir, "./meshes/goal.obj")

    scaling_factor = [1, 1, 1]
    visual_shape_id = p.createVisualShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=scaling_factor)

    # Add a collision shape for physics simulation (optional)
    collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_MESH, fileName=goal_path, meshScale=scaling_factor, flags=p.GEOM_FORCE_CONCAVE_TRIMESH)

    # Create a multibody object that combines both visual and collision shapes
    goal_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape_id, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)
    #goal_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=position, baseOrientation=orientation)

    p.changeDynamics(goal_id, -1, restitution=0.8) # Bounciness

    return goal_id
