import os
import docker
import time
import sys


class RunDocker():
    def __init__(self, path2env, IMAGE = 'zfw1226/unreal-gpu:v0.1',):
       self.docker_client = docker.from_env()
       self.check_image(target_images=IMAGE)
       os.system('xhost +')
       self.image = IMAGE
       self.path2env = path2env

    def start(self,
              ENV_BIN = '/RealisticRendering_RL/RealisticRendering/Binaries/Linux/RealisticRendering',
              ENV_DIR_DOCKER='/unreal',
              ):

        path2binary = os.path.join(self.path2env,ENV_BIN)
        if not os.path.exists(path2binary):
            print ('Did not find unreal environment, Please move your binary file to env/UnrealEnv')
            sys.exit()

        docker_cmd = 'nvidia-docker run -d -it --env="DISPLAY=:0.0" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --volume="{ENV_DIR_HOST}:{ENV_DIR_DOCKER}:rw" {IMAGE} '
        run_cmd = 'bash -c "chown unrealcv {ENV_DIR_DOCKER} -R && su unrealcv -c {ENV_DIR_BIN_DOCKER}"'
        cmd = docker_cmd.format(ENV_DIR_HOST=self.path2env, ENV_DIR_DOCKER=ENV_DIR_DOCKER, IMAGE=self.image) + \
              run_cmd.format(ENV_DIR_DOCKER=ENV_DIR_DOCKER, ENV_DIR_BIN_DOCKER = os.path.join(ENV_DIR_DOCKER,ENV_BIN))

        print (cmd)
        os.system(cmd)
        self.container = self.docker_client.containers.list()

        return self.get_ip()

    def get_ip(self):
        return self.container[0].attrs['NetworkSettings']['Networks']['bridge']['IPAddress']

    def get_path2UnrealEnv(self):
        import gym_unrealcv
        gympath = os.path.dirname(gym_unrealcv.__file__)
        return os.path.join(gympath, 'envs/UnrealEnv')

    def close(self):
        self.container[0].remove(force=True)

    def check_image(self, target_images='zfw1226/unreal-gpu:v0.1'):

        images = self.docker_client.images.list()

        # Check the existence of image
        Found_Img = False
        for i in range(len(images)):
            if images[i].tags.count(target_images) > 0:
                Found_Img = True
        # Download image
        if Found_Img == False:
            print ('Do not found images,Downloading')
            self.docker_client.images.pull(target_images)
        else:
            print ('Found images')





