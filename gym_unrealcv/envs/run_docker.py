import os
import docker
import time
import sys

ENV = dict(
    RealisticRendering = '/RealisticRendering_RL/RealisticRendering/Binaries/Linux/RealisticRendering',
    ArchinteriorsVol2Sceen1 = '/ArchinteriorsVol2Sceen1/Binaries/Linux/ArchinteriorsVol2Sceen1',
    UrbanCity = '/UrbanCity/Binaries/Linux/UrbanCity'
)


class RunDocker():
    def __init__(self, IMAGE = 'zfw1226/unreal-gpu:v0.1',):
       self.docker_client = docker.from_env()
       self.check_image(target_images = IMAGE)
       os.system('xhost +')
       self.image = IMAGE

    def start(self,
              ENV_NAME = 'RealisticRendering',
              ENV_DIR_DOCKER='/unreal',
              ENV_DIR_HOST = 'UnrealEnv'):

        HOST_DIR = self.get_abspath(ENV_DIR_HOST)
        binary = HOST_DIR + ENV[ENV_NAME]
        print binary
        if not os.path.exists(binary):
            sh_dir = HOST_DIR + '/' + ENV_NAME + '.sh'
            print sh_dir
            if os.path.exists(sh_dir):
                os.system('sh ' + sh_dir)
            else :
                print 'Did not find unreal environment, Please move your binary file to env/UnrealEnv'
                sys.exit()


        docker_cmd = 'nvidia-docker run  -d -it  --env="DISPLAY=:0.0"     --env="QT_X11_NO_MITSHM=1"   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"     --volume="{ENV_DIR_HOST}:{ENV_DIR_DOCKER}:rw"    {IMAGE} '
        run_cmd = 'bash -c "chown unrealcv {ENV_DIR_DOCKER} -R && su unrealcv -c {ENV_DIR_DOCKER}{ENV_DIR_BIN}"'
        cmd = docker_cmd.format(ENV_DIR_HOST=HOST_DIR, ENV_DIR_DOCKER=ENV_DIR_DOCKER, IMAGE=self.image) + run_cmd.format(
            ENV_DIR_DOCKER=ENV_DIR_DOCKER, ENV_DIR_BIN=ENV[ENV_NAME])
        print cmd
        os.system(cmd)
        time.sleep(5)
        self.container = self.docker_client.containers()
        return self.get_ip(),HOST_DIR


    def get_ip(self):
        return self.container[0]['NetworkSettings']['Networks']['bridge']['IPAddress']
    def get_abspath(self, relativepath):
        paths = sys.path
        for p in paths:
           if p[-20:].find('gym-unrealcv') > 0:
               gympath = p

       
        return os.path.join(gympath,'gym_unrealcv/envs',relativepath)

    def close(self):
        #.container[0].stop()
        self.container[0].remove(force = True)

    def check_image(self,target_images='zfw1226/unreal-gpu:v0.1'):
        images = self.docker_client.images()

        # Check the existence of image
        Found_Img = False
        for i in range(len(images)):
            if images[i]['RepoTags'].count(target_images) > 0:
                Found_Img = True
        # Download image
        if Found_Img == False:
            print 'Do not found images,Downloading'
            #pdb.set_trace()
            self.docker_client.pull(target_images)
        else:
            print 'Found images'
