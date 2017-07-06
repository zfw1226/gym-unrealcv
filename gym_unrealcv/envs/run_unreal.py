import os
from multiprocessing import Process
import run_docker
import sys
import getpass
import gym_unrealcv

ENV = dict(
    #RealisticRendering = '/RealisticRendering-Linux-0.3.6/LinuxNoEditor/RealisticRendering/Binaries/Linux/RealisticRendering',
    RealisticRendering = 'RealisticRendering_RL/RealisticRendering/Binaries/Linux/RealisticRendering',
    ArchinteriorsVol2Sceen1 = 'ArchinteriorsVol2Scene1/ArchinteriorsVol2Scene1/Binaries/Linux/ArchinteriorsVol2Scene1',
    UrbanCity = '/UrbanCity/Binaries/Linux/UrbanCity'
)

class RunUnreal():
    def __init__(self,envname='RealisticRendering', docker = False,):
        if docker :
            self.docker = run_docker.RunDocker()
        else:
            path2env = self.get_path2UnrealEnv()
            print path2env
            username = getpass.getuser()
            cmd= 'sudo chown {USER} {ENV_PATH} -R'
            os.system(cmd.format(USER=username, ENV_PATH=path2env))
            path2binary = os.path.join(path2env,ENV[envname])
            print path2binary
            p = Process(target=self.run_proc,args=(path2binary,))
            p.start()
            print 'Running docker-free env'


    def get_path2UnrealEnv(self):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath,'envs/UnrealEnv')

    def run_proc(self, path2env):
        os.system('export Display=:0.0')
        cmd = 'nohup {path2env}'
        os.system(cmd.format(path2env=path2env))
        return os.getpid()


run = RunUnreal()