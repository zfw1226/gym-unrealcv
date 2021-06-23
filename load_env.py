import os
import argparse

binary_list = dict(
    # for searching
    RealisticRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/RealisticRendering_RL_3.10.zip',
    Arch1='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/ArchinteriorsVol2Scene1-Linux-0.3.10.zip',
    # env with spline target for tracking
    SplineCharacterA='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SplineCharacterA.zip',  # env for end-to-end active object tracking (icml 2018, tpami)
    SplineCharacterF='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SplineCharacterF.zip',  # env for end-to-end active object tracking (icml 2018, tpami)
    # training env for tracking
    RandomRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/RandomRoom.zip', # env for end-to-end active object tracking (icml 2018, tpami)
    DuelingRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/DuelingRoom.zip', # env for ad-vat (iclr 2019)
    MCRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/MCRoom.zip',  # env for pose-assisted multi-camera tracking (aaai 2020)
    Textures='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/Textures.zip', # textures for environment augmentation
    # realistic testing env for tracking
    UrbanCity='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/UrbanCity_2P.zip', # env for ad-vat (iclr 2019)
    SnowForest='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SnowForest_2P.zip', # env for ad-vat (iclr 2019)
    Garage='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/Garage_2P.zip', # env for ad-vat (iclr 2019)
    # multi camera
    UrbanTree='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/urban_cam.zip', # env for pose-assisted multi-camera tracking (aaai 2020)
    Garden='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/neighborhood.zip', # env for pose-assisted multi-camera tracking (aaai 2020)
    # Arm env
    Arm='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/arm-0610.zip'
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env", nargs='?', default='RobotArm-Discrete-v0',
                        help='Select the environment to run')
    args = parser.parse_args()
    cmd_load = 'wget '+binary_list[args.env]
    os.system(cmd_load)
    name_zip = binary_list[args.env].split('/')[-1]
    cmd_unzip = 'unzip -n {zipfile} -d {dir}'.format(zipfile=name_zip, dir='gym_unrealcv/envs/UnrealEnv')
    os.system(cmd_unzip)
    os.system('rm ' + name_zip)