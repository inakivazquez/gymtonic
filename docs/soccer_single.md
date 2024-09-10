# Soccer Single

```{figure} ./images/soccer_single.gif 
:width: 200px
:name: soccer_single
```


|   |   |
|---|---|
| Action Space | Box(-1, +1, (2,), float32) |
| Observation Shape | (7,) |
| Observation High | [2*math.pi, perimeter_side/2, perimeter_side/2, perimeter_side, perimeter_side, perimeter_side, perimeter_side ] |
| Observation Low | [-2*math.pi, -perimeter_side/2, -perimeter_side/2, -perimeter_side, -perimeter_side, -perimeter_side, -perimeter_side ] |
| Import | `gymnasium.make("gymtonic/GridTargetDirectional_v0")` | 


### Description
This environment represents a soccer pitch and an agent that must learn to score a goal. The agent and the ball appear at random positions on the field.
GOAL DIRECTION?

### Action Space
The action space has two continuous dimenions. The first one [-1, 1] represents how to rotate the agent (scaled to pi/6), the second, also in the range [-1, 1], represents the force to move in the forward direction where the agent is oriented (negative values move backwards). 

### Observation Space
The state is an 7-dimensional vector representing:
- The current orientation of the agent
- Absolute position (x,y) of the agent
- Relative position (x,y) of the ball relative to the agent
- Relative position (x,y) of the goal line relative to the agent

### Rewards
Reward are:
- +100 for scoring a goal
- +0.01 every time the agent kicks the ball
- -0.01 per step (time penalty)

### Starting State
The agent and the ball start at random positions.

### Episode Termination
The episode finishes if:
1) the agent scores a goal
2) the ball goes outside the stadium
2) 500 steps are reached (episode truncated)

### Arguments
The size of the pitch can be configured with `perimeter_side` (length of the pitch, width is half that value). Default is 10.

The maximum speed of the player can be configured with the `max_speed` parameter. Default is 1.

### Version History
- v0: Initial version

<!-- ### References -->

### Credits
Created by Inaki Vazquez
