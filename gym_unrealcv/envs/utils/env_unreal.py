import getpass
import os
import time
# from multiprocessing import Process
import sys
import subprocess
import atexit
import signal
# api for launching UE4 binary


class RunUnreal():
    def __init__(self, ENV_BIN, ENV_MAP=None):

        self.env_bin = ENV_BIN
        self.env_map = ENV_MAP
        self.path2env = self.get_path2UnrealEnv()
        self.path2binary = os.path.abspath(os.path.join(self.path2env, self.env_bin))
        self.path2unrealcv = os.path.join(os.path.split(self.path2binary)[0], 'unrealcv.ini')
        assert os.path.exists(self.path2binary), \
            'Please load env binary in UnrealEnv and Check the env_bin in setting file!'

    def start(self, docker, resolution=(160, 160)):
        port = self.read_port()
        self.write_resolution(resolution)
        self.use_docker = docker
        if self.use_docker:
            import gym_unrealcv.envs.utils.run_docker
            self.docker = gym_unrealcv.envs.utils.run_docker.RunDocker(self.path2env)
            env_ip = self.docker.start(ENV_BIN=self.env_bin)
            print('Running nvidia-docker env')
        else:
            env_ip = '127.0.0.1'
            while not self.isPortFree(env_ip, port):
                port += 1
                self.write_port(port)
            #self.modify_permission(self.path2env)
            cmd_exe = os.path.abspath(self.path2binary)
            if self.env_map is not None:
                cmd_exe += map

            self.env = subprocess.Popen([cmd_exe], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, start_new_session=True)

            signal.signal(signal.SIGTERM, self.signal_handler)
            signal.signal(signal.SIGINT, self.signal_handler)

            print('Running docker-free env, pid:{}'.format(self.env.pid))

        print('Please wait for a while to launch env......')
        time.sleep(5)
        return env_ip, port

    def get_path2UnrealEnv(self): # get path to UnrealEnv
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs', 'UnrealEnv')

    def run_proc(self, path2env, map): # run env in a new process
        # os.system('export Display=:0.0')
        if 'linux' in sys.platform:
            cmd = 'exec nohup {path2env} '  # linux
        elif 'win' in sys.platform:
            cmd = 'start /b  {path2env} '  # win
        cmd_exe = cmd.format(path2env=os.path.abspath(path2env))
        if map is not None:
            cmd_exe += map
        print(cmd_exe)
        os.system(cmd_exe)

    def close(self): # close env
        if self.use_docker:
            self.docker.close()
        else:
            self.env.terminate()
            self.env.wait()

    def signal_handler(self):
        self.close()
    def modify_permission(self, path):
        cmd = 'sudo chown {USER} {ENV_PATH} -R'
        username = getpass.getuser()
        os.system(cmd.format(USER=username, ENV_PATH=path))

    def read_port(self): # read port number from unrealcv.ini
        if os.path.exists(self.path2unrealcv): # check unrealcv.ini exist
            with open(self.path2unrealcv, 'r') as f:
                s = f.read() # read unrealcv.ini
                ss = s.split()
            return int(ss[1][-4:]) # return port number
        else:
            return 9000 # default port number

    def write_port(self, port): # write port number to unrealcv.ini
        with open(self.path2unrealcv, 'r') as f:
            s = f.read()
            ss = s.split('\n')
        with open(self.path2unrealcv, 'w') as f:
            print(ss[1])
            ss[1] = 'Port={port}'.format(port=port)
            d = '\n'
            s_new = d.join(ss)
            f.write(s_new)

    def write_resolution(self, resolution): # set unrealcv camera resolution by writing unrealcv.ini
        if os.path.exists(self.path2unrealcv):
            with open(self.path2unrealcv, 'r') as f:
                s = f.read()
                ss = s.split('\n')
            with open(self.path2unrealcv, 'w') as f:
                ss[2] = 'Width={width}'.format(width=resolution[0])
                ss[3] = 'Height={height}'.format(height=resolution[1])
                d = '\n'
                s_new = d.join(ss)
                f.write(s_new)

    def isPortFree(self, ip, port): # check port is free
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if 'linux' in sys.platform:
                sock.bind((ip, port))
            elif 'win' in sys.platform:
                sock.bind((ip, port))
                sock.connect((ip, port))
        except Exception as e:
            sock.close()
            print(e) # print error message
            return False
        sock.close()
        return True
