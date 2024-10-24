# Grid Target Simple

![Grid Target](images/grid_target.gif)


|   |   |
|---|---|
| Action Space | Discrete(4) |
| Observation Space | Discrete(`n_rows` x `n_columns`) |
| Import | `gymnasium.make("gymtonic/GridTargetSimple_v0")` | 


### Description
This is a basic toy environment to experiment with Pybullet-based scenarios and Discrete observation spaces. An agent that moves on a grid to reach a target that appears at random positions. The number of columns (along axis X) and rows (along axis Y) of the grid is configurable (default is 5 x 5).

The agent always starts at position (0,0) of the grid.

### Action Space
There are four discrete actions available: go North, Soutch, East and West.

### Observation Space
The state is a value representing the position of the target in the grid. If the position is (`x`,`y`) the state would be `x * n_columns + y * n_rows`. For example, in a 3 x 6 grid, if the position of the target is (2,3) the state would be 2x6+3 = 15. 

### Rewards
Reward is +1 if the agent reaches the location of the target.

### Starting State
The agent starts always at position (0,0).

### Episode Termination
The episode finishes if:
1) the agent catches the target
2) 100 steps are reached (episode truncated)

### Arguments
The size of the grid can be configured with `n_rows` and `n_columns`. Default is 5 x 5.

For example, to create a 6 x 10 grid:
```python
import gymnasium as gym
import gymtonic

env = gym.make('gymtonic/GridTargetSimple-v0', n_rows=6, n_columns=10, render_mode='human')
```

### Version History
- v0: Initial version

<!-- ### References -->

### Credits
Created by Inaki Vazquez
