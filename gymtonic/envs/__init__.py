from gymnasium.envs.registration import register

register(
    id='gymtonic/GridTarget-v0',
    entry_point='gymtonic.envs.grid_target_v0:GridTargetEnv',
    max_episode_steps=100,
    kwargs=dict(smooth_movement=True)
)

register(
    id='gymtonic/GridTargetDiff-v0',
    entry_point='gymtonic.envs.grid_target_diff_v0:GridTargetEnv',
    max_episode_steps=100,
    kwargs=dict(smooth_movement=True)
)
