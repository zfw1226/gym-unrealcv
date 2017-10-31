import getpass
import os
import time
from multiprocessing import Process

import gym_unrealcv.envs.utils.run_docker


# api for running unrealenv

class RunUnreal():
    def __init__(self,ENV_BIN ='ArchinteriorsVol2Scene1/ArchinteriorsVol2Scene1/Binaries/Linux/ArchinteriorsVol2Scene1'):

        self.env_bin = ENV_BIN
        self.path2env = self.get_path2UnrealEnv()
        self.path2binary = os.path.join(self.path2env, self.env_bin)

    def start(self,docker):
        port = self.read_port(self.path2binary)
        if docker :
            self.docker = gym_unrealcv.envs.utils.run_docker.RunDocker(self.path2env)
            env_ip = self.docker.start(ENV_BIN= self.env_bin)
            print 'Running nvidia-docker env'
        else:
            env_ip = '127.0.0.1'
            while not self.isPortFree(env_ip, port):
                port += 1
            print port
            self.write_port(self.path2binary,port)
            self.pid = []
            self.modify_permission(self.path2env)
            self.env = Process(target=self.run_proc,args=(self.path2binary,))
            self.env.start()

            print 'Running docker-free env'

        print 'Please wait for a while to launch env......'
        time.sleep(10)
        return env_ip,port

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

    def read_port(self,bin_path):
        s = bin_path.split('/')
        s[-1] = 'unrealcv.ini'
        delimiter = '/'
        ini_path = delimiter.join(s)
        with open(ini_path,'r') as f:
            s=f.read()
            ss = s.split()
        return int(ss[1][-4:])

    def write_port(self,bin_path,port):
        s = bin_path.split('/')
        s[-1] = 'unrealcv.ini'
        delimiter = '/'
        ini_path = delimiter.join(s)
        with open(ini_path,'r') as f:
            s=f.read()
            ss = s.split('\n')
            print ss
        with open(ini_path, 'w') as f:
            print ss[1]
            ss[1] = 'Port={port}'.format(port = port)
            d = '\n'
            s_new = d.join(ss)
            f.write(s_new)

    def isPortFree(self, ip, port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((ip, port))
        except Exception, e:
            sock.close()
            print e
            return False
        sock.close()
        return True


#run = RunUnreal()