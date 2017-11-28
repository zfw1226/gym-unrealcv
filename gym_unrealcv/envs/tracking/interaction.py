from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
class Tracking(UnrealCv):
    def __init__(self, env, cam_id = 0, port = 9000,
                 ip = '127.0.0.1'):

        super(Tracking, self).__init__(env=env, port = port,ip = ip , cam_id=cam_id,resolution=(160,120))