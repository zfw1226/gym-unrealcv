This tutorial will show you how to configure and modify the environments without coding.
You can configure most of the common-used arguments about the environment in [the register file](../gym_unrealcv/__init__.py) and [the json files](../gym_unrealcv/envs/setting).

In the register file, a specific environment is registered with env_id, 
entry_point and a set of arguments, shown as below.
```buildoutcfg
register(
    id='Search-RrDoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.json',
              'reset_type' : 'waypoint', # 'waypoint', 'random', 'testpoint'
              'test': False, # True, False
              'action_type' : 'discrete', # 'discrete', 'continuous'
              'observation_type': 'color', # 'color', 'depth', 'rgbd'
              'reward_type': 'bbox', #'distance','bbox','bbox_distance'
              'docker': use_docker # True, False
              },
    max_episode_steps = 1000000 
)
```
In this case, you can configure the type of observation, action, reward, the resetting mode and if using docker to run the environemnt. 
The comment indicates the candidate variable.

Besides, more details of the environment is in the `setting_file`, a json file in `/gym_unrealcv/envs/setting`. 
For example, [search_rr_door41.json](../gym_unrealcv/envs/setting/search_rr_door41.json) with comments is shown as below:

```buildoutcfg
{
	"env_name": "RealisticRendering",
	# path to the binary of unreal binary, it must be under "gym_unrealcv/envs/UnrealEnv"
	"env_bin": "RealisticRendering_RL_3.10/RealisticRendering/Binaries/Linux/RealisticRendering",
	"cam_id": 0,
	"height": 40, # height of camera (cm)
	"pitch": 0, # the pitch angle of camera 
	"maxsteps": 100, # the max steps in an episode 
	"trigger_th": 0.6, # when trigger value is lagger than trigger_th, the virual button is trigger.
	"reward_th": 0.1, # if reward < reward_th, reward=0
	"reward_factor": 10, # reward = reward_factor * bbox_size/img_size
	"collision_th": 50, # param for waypoint reset mode, the min distance between the waypoint and the collision point
	"waypoint_th": 200, # param for waypoint reset mode, the min distance between two waypoints
	
	# the list of target objects
	"targets": [
		"SM_Door_39"
	],
	
	# the list of the coordinate of start point for testing
	"test_xy": [
		[-106.195, 437.424],
		[27.897, -162.437 ],
		[10.832, 135.126  ],
		[67.903, 26.995   ],
		[-23.558, -187.354],
		[-80.312, 254.579 ]
	],
	
	# define the discrete actions [liner velocity, angular velocity, trigger]
	"discrete_actions": [
		[50, 0, 0],
		[30, 15, 0],
		[30, -15, 0],
		[15, 30, 0],
		[15, -30, 0],
		[0, 0, 1]
	],
	
	# the range of each action dimension,[liner velocity, angular velocity, trigger]
	"continous_actions": {
		"high": [100, 45, 1],
		"low":  [0,  -45, 0]
	}
}
```
