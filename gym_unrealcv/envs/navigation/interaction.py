from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
class Navigation(UnrealCv):
    def __init__(self, env, cam_id = 0, port = 9000,
                 ip = '127.0.0.1' , targets = None):

        super(Navigation, self).__init__(env=env, port = port,ip = ip , cam_id=cam_id,resolution=(160,120))

        if targets == 'all':
            self.targets = self.get_objects()
            self.color_dict = self.build_color_dic(self.targets)
        elif targets is not None:
            self.targets = targets
            self.color_dict = self.build_color_dic(self.targets)

#nav = Navigation(env='test')