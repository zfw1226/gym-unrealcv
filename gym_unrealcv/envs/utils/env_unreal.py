import getpass
import os
import time
from multiprocessing import Process

# api for running unrealenv


class RunUnreal():
    def __init__(self, ENV_BIN, ENV_MAP=None):

        self.env_bin = ENV_BIN
        self.env_map = ENV_MAP
        self.path2env = self.get_path2UnrealEnv()
        self.path2binary = os.path.join(self.path2env, self.env_bin)
        assert os.path.exists(self.path2binary), \
            'Please load env binary in UnrealEnv and Check the env_bin in setting file!'

    def start(self, docker, resolution=(160, 160)):
        # check binary exist
        port = self.read_port(self.path2binary)
        self.write_resolution(self.path2binary, resolution)
        self.use_docker = docker
        if self.use_docker:
            import gym_unrealcv.envs.utils.run_docker
            self.docker = gym_unrealcv.envs.utils.run_docker.RunDocker(self.path2env)
            env_ip = self.docker.start(ENV_BIN=self.env_bin)
            print ('Running nvidia-docker env')
        else:
            env_ip = '127.0.0.1'
            while not self.isPortFree(env_ip, port):
                port += 1
                self.write_port(self.path2binary, port)
            #self.modify_permission(self.path2env)
            self.env = Process(target=self.run_proc, args=(self.path2binary, self.env_map))
            self.env.start()
            print ('Running docker-free env, pid:{}'.format(self.env.pid))

        print ('Please wait for a while to launch env......')
        time.sleep(10)
        return env_ip, port

    def get_path2UnrealEnv(self):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/UnrealEnv')

    def run_proc(self, path2env, map):
        # os.system('export Display=:0.0')
        cmd = 'exec nohup {path2env} '
        cmd_exe = cmd.format(path2env=path2env)
        if map is not None:
            cmd_exe += map
        print (cmd_exe)
        os.system(cmd_exe)

    def close(self):
        if self.use_docker:
            self.docker.close()
        else:
            import signal
            os.kill(self.env.pid+1, signal.SIGTERM)

    def modify_permission(self, path):
        cmd = 'sudo chown {USER} {ENV_PATH} -R'
        username = getpass.getuser()
        os.system(cmd.format(USER=username, ENV_PATH=path))

    def read_port(self, bin_path):
        s = bin_path.split('/')
        s[-1] = 'unrealcv.ini'
        delimiter = '/'
        ini_path = delimiter.join(s)
        if os.path.exists(ini_path):
            with open(ini_path, 'r') as f:
                s=f.read()
                ss = s.split()
            return int(ss[1][-4:])
        else:
            return 9000

    def write_port(self, bin_path, port):
        s = bin_path.split('/')
        s[-1] = 'unrealcv.ini'
        delimiter = '/'
        ini_path = delimiter.join(s)
        with open(ini_path, 'r') as f:
            s=f.read()
            ss = s.split('\n')
        with open(ini_path, 'w') as f:
            print (ss[1])
            ss[1] = 'Port={port}'.format(port = port)
            d = '\n'
            s_new = d.join(ss)
            f.write(s_new)

    def write_resolution(self, bin_path, resolution):
        s = bin_path.split('/')
        s[-1] = 'unrealcv.ini'
        delimiter = '/'
        ini_path = delimiter.join(s)
        if os.path.exists(ini_path):
            with open(ini_path, 'r') as f:
                s = f.read()
                ss = s.split('\n')
            with open(ini_path, 'w') as f:
                ss[2] = 'Width={width}'.format(width=resolution[0])
                ss[3] = 'Height={height}'.format(height=resolution[1])
                d = '\n'
                s_new = d.join(ss)
                f.write(s_new)

    def isPortFree(self, ip, port):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((ip, port))
        except Exception as e:
            sock.close()
            print (e)
            return False
        sock.close()
        return True
