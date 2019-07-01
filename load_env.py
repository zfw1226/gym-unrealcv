import os
import argparse

source = dict(
    RandomRoom='https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/RandomRoom.zip',
    DuelingRoom='https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/DuelingRoom.zip',
    SplineCharacterA='https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/SplineCharacterA.zip',
    SplineCharacterF='https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/SplineCharacterA.zip',
    Arch1='https://www.cs.jhu.edu/~qiuwch/release/unrealcv/ArchinteriorsVol2Scene1-Linux-0.3.10.zip',
    RealisticRoom='https://s3-us-west-1.amazonaws.com/unreal-rl/RealisticRendering_RL_3.10.zip',
    Textures='https://www.cs.jhu.edu/~qiuwch/unrealcv/binaries/Textures.zip',
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env", nargs='?', default='RobotArm-Discrete-v0',
                        help='Select the environment to run')
    args = parser.parse_args()
    cmd_load = 'wget '+source[args.env]
    os.system(cmd_load)
    name_zip = source[args.env].split('/')[-1]
    cmd_unzip = 'unzip -n {zipfile} -d {dir}'.format(zipfile=name_zip, dir='gym_unrealcv/envs/UnrealEnv')
    os.system(cmd_unzip)
    os.system('rm ' + name_zip)