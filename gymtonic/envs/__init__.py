from gymnasium.envs.registration import register

"""
Register custom Gymnasium environments for the Gymtonic package.

gymtonic/GridTarget-v0: GridTargetEnv with smooth_movement set to True.
gymtonic/GridTargetDiff-v0: GridTargetEnv with smooth_movement set to True.

"""
register(
    id='gymtonic/SoccerSingle-v0',
    entry_point='gymtonic.envs.soccer_single_v0:SoccerSingleEnv',
    max_episode_steps=500
)

register(
    id='gymtonic/SoccerSingleDiscrete-v0',
    entry_point='gymtonic.envs.soccer_single_discrete_v0:SoccerSingleDiscreteEnv',
    max_episode_steps=500
)

register(
    id='gymtonic/SoccerSingleRaycast-v0',
    entry_point='gymtonic.envs.soccer_single_ray_v0:SoccerSingleRaycastEnv',
    max_episode_steps=500
)


register(
    id='gymtonic/GridTargetSimple-v0',
    entry_point='gymtonic.envs.grid_target_simple_v0:GridTargetSimpleEnv',
    max_episode_steps=100,
    kwargs=dict(smooth_movement=True)
)

register(
    id='gymtonic/GridTarget-v0',
    entry_point='gymtonic.envs.grid_target_v0:GridTargetEnv',
    max_episode_steps=100,
    kwargs=dict(smooth_movement=True)
)

register(
    id='gymtonic/GridTargetDirectional-v0',
    entry_point='gymtonic.envs.grid_target_directional_v0:GridTargetDirectionalEnv',
    max_episode_steps=100,
    kwargs=dict(smooth_movement=True)
)

register(
    id='gymtonic/BlockPush-v0',
    entry_point='gymtonic.envs.block_push_v0:BlockPush',
    max_episode_steps=500
)
