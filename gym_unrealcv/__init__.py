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


#RrPlant7
register(
    id='Search-RrPlantDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : False,
              'discrete_action': True
              }
)
register(
    id='Search-RrPlantContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : False,
              'discrete_action': False
              }
)

register(
    id='Search-RrPlantDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : True,
              'discrete_action': True
              }
)
register(
    id='Search-RrPlantContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant7.yaml',
              'test' : True,
              'discrete_action': False
              }
)


# RrPlant8
register(
    id='Search-RrPlantDiscrete-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': False,
              'discrete_action': True}
)
register(
    id='Search-RrPlantContinuous-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': False,
              'discrete_action': False}
)
register(
    id='Search-RrPlantDiscreteTest-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': True,
              'discrete_action': True}
)
register(
    id='Search-RrPlantContinuousTest-v1',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant8.yaml',
              'test': True,
              'discrete_action': False}
)

# RrDoor41
register(
    id='Search-RrDoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': False,
              'discrete_action': True}
)
register(
    id='Search-RrDoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': False,
              'discrete_action': False}
)
register(
    id='Search-RrDoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': True,
              'discrete_action': True}
)
register(
    id='Search-RrDoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_door41.yaml',
              'test': True,
              'discrete_action': False}
)


#Arch1Door1
register(
    id='Search-Arch1DoorDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml',
              'test': False,
              'discrete_action': True}
)
register(
    id='Search-Arch1DoorContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml',
              'test': False,
              'discrete_action': False}
)
register(
    id='Search-Arch1DoorDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml',
              'test': True,
              'discrete_action': True}
)
register(
    id='Search-Arch1DoorContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door1.yaml',
              'test': True,
              'discrete_action': False}
)


# RrMultiPlants
register(
    id='Search-RrMultiPlantsDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': False,
              'discrete_action': True}
)
register(
    id='Search-RrMultiPlantsContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': False,
              'discrete_action': False}
)

register(
    id='Search-RrMultiPlantsDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': True,
              'discrete_action': True},
)
register(
    id='Search-RrMultiPlantsContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_rr_plant78.yaml',
              'test': True,
              'discrete_action': False},
)


# Arch1MultiDoors
register(
    id='Search-Arch1MultiDoorsDiscrete-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': False,
              'discrete_action': True}
)
register(
    id='Search-Arch1MultiDoorsContinuous-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': False,
              'discrete_action': False}
)
register(
    id='Search-Arch1MultiDoorsDiscreteTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': True,
              'discrete_action': True}
)
register(
    id='Search-Arch1MultiDoorsContinuousTest-v0',
    entry_point='gym_unrealcv.envs:UnrealCvSearch_base',
    kwargs = {'setting_file' : 'search_arch1_door12.yaml',
              'test': True,
              'discrete_action': False}
)