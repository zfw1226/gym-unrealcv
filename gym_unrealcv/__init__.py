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


# single target, reward by trigger
register(
    id='Unrealcv-Search-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_v4',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml'}
)

register(
    id='Unrealcv-Search-v2',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_v4',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml'}
)

register(
    id='Unrealcv-Search-v3',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_v4',
    kwargs = {'setting_file' : 'search_rr_door41.yaml'}
)

# multi targets, reward by trigger
register(
    id='Unrealcv-Search-v4',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_v4',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml'},
)

register(
    id='Unrealcv-Search-v5',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_v4',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml'}
)

