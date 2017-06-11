from gym.envs.registration import register
import logging
logger = logging.getLogger(__name__)
register(
    id='Unrealcv-Simple-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSimple'
)
register(
    id='Unrealcv-Search-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch'
)

# adding reward by distance
register(
    id='Unrealcv-Search-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_v1'
)

# target driven and reward by distance
register(
    id='Unrealcv-Search-v2',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_v2'
)

