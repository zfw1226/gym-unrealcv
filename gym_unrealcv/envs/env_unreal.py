import os
from multiprocessing import Process
import run_docker
import getpass


# api for running unrealenv

class RunUnreal():
    def __init__(self,ENV_BIN ='ArchinteriorsVol2Scene1/ArchinteriorsVol2Scene1/Binaries/Linux/ArchinteriorsVol2Scene1'):

        self.env_bin = ENV_BIN
        self.path2env = self.get_path2UnrealEnv()
        self.path2binary = os.path.join(self.path2env, self.env_bin)

    def start(self,docker):
        if docker :
            self.docker = run_docker.RunDocker(self.path2env)
            env_ip = self.docker.start(ENV_BIN= self.env_bin)
        else:
            self.pid = []
            self.modify_permission(self.path2env)
            self.env = Process(target=self.run_proc,args=(self.path2binary,))
            self.env.start()
            env_ip = '127.0.0.1'
            print 'Running docker-free env'
        return env_ip

    def get_path2UnrealEnv(self):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath,'envs/UnrealEnv')

    def run_proc(self, path2env):
        os.system('export Display=:0.0')
        cmd = 'nohup {path2env}'
        os.system(cmd.format(path2env=path2env))
        self.pid.append(os.getpid())


    def modify_permission(self,path):
        cmd = 'sudo chown {USER} {ENV_PATH} -R'
        username = getpass.getuser()
        os.system(cmd.format(USER=username, ENV_PATH=path))


#run = RunUnreal()