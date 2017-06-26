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
    id='Search-RrPlant-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml'}
)

register(
    id='Search-RrPlant-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml'}
)

register(
    id='Search-RrDoor-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml'}
)

register(
    id='Search-Arch1Door-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml'}
)

# multi targets
register(
    id='Search-RrMultiPlants-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml'},
)

register(
    id='Search-Arch1MultiDoors-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml'}
)




