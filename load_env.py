import os
import argparse
import wget
import zipfile
import sys
import shutil

binary_linux = dict(
    # for searching
    RealisticRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/RealisticRendering_RL_3.10.zip',
    Arch1='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/ArchinteriorsVol2Scene1-Linux-0.3.10.zip',
    # env with spline target for tracking
    SplineCharacterA='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SplineCharacterA.zip',  # env for end-to-end active object tracking (icml 2018, tpami)
    SplineCharacterF='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SplineCharacterF.zip',  # env for end-to-end active object tracking (icml 2018, tpami)
    # training env for tracking
    RandomRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/RandomRoom.zip', # env for end-to-end active object tracking (icml 2018, tpami)
    DuelingRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/DuelingRoom.zip', # env for ad-vat (iclr 2019)
    FlexibleRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/FlexibleRoom.zip', # env with distractors and obstacles (icml 2021)
    MCRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/MCRoom.zip',  # env for pose-assisted multi-camera tracking (aaai 2020)
    Textures='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/Textures.zip', # textures for environment augmentation
    # realistic testing env for tracking
    UrbanCity='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/UrbanCity_2P.zip', # env for ad-vat (iclr 2019)
    SnowForest='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/SnowForest_2P.zip', # env for ad-vat (iclr 2019)
    Garage='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/Garage_2P.zip', # env for ad-vat (iclr 2019)
    UrbanCityMulti='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/UrbanCity_Multi.zip', # env for distraction robustness (icml 2021)
    GarageMulti='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/ParkingLot_Multi.zip', # env for distraction robustness (icml 2021)
    # multi camera
    UrbanTree='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/urban_cam.zip', # env for pose-assisted multi-camera tracking (aaai 2020)
    Garden='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/neighborhood.zip', # env for pose-assisted multi-camera tracking (aaai 2020)
    # Arm env
    Arm='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/arm-0610.zip',
    test='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/testfolder.zip'  # this is an empty file used for testing the script
)

binary_win = dict(
    FlexibleRoom='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/FlexibleRoom_Win_v0.zip',
    UrbanCity='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/urbancity_win_v1.zip',
    Garage='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/parkinglot_win_v1.zip',
    Garden='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/neighborhood_win_v1.zip',
    test='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/testfolder.zip',
    Textures='https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/Textures.zip'
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env", nargs='?', default='RealisticRoom',
                        help='Select the binary to download')
    if 'win' in sys.platform:
        binary_all = binary_win
    elif 'linux' in sys.platform:
        binary_all = binary_linux
    args = parser.parse_args()

    if args.env in binary_all:
        filename = wget.download(binary_all[args.env])  # download the binary
        with zipfile.ZipFile(filename, "r") as z:
            z.extractall()  # extract the zip file
        folder = filename[:-4]
        target = 'gym_unrealcv/envs/UnrealEnv'
        shutil.move(folder, target)
        os.remove(filename)
    else:
        print(f"{args.env} is not available to your platform")
        exit()

