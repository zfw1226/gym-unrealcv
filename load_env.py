import os
import argparse
import zipfile
import sys
import shutil
aliyun = 'https://gym-unrealcv.oss-cn-beijing.aliyuncs.com/'
binary_linux = dict(
    # for searching
    RealisticRoom='RealisticRendering_RL_3.10.zip',
    Arch1='ArchinteriorsVol2Scene1-Linux-0.3.10.zip',
    # env with spline target for tracking
    SplineCharacterA='SplineCharacterA.zip',  # env for end-to-end active object tracking (icml 2018, tpami)
    SplineCharacterF='SplineCharacterF.zip',  # env for end-to-end active object tracking (icml 2018, tpami)
    # training env for tracking
    RandomRoom='RandomRoom.zip', # env for end-to-end active object tracking (icml 2018, tpami)
    DuelingRoom='DuelingRoom.zip', # env for ad-vat (iclr 2019)
    FlexibleRoom='FlexibleRoom.zip', # env with distractors and obstacles (icml 2021)
    MCRoom='MCRoom.zip',  # env for pose-assisted multi-camera tracking (aaai 2020)
    Textures='Textures.zip', # textures for environment augmentation
    # realistic testing env for tracking
    UrbanCity='UrbanCity_2P.zip', # env for ad-vat (iclr 2019)
    SnowForest='SnowForest_2P.zip', # env for ad-vat (iclr 2019)
    Garage='Garage_2P.zip', # env for ad-vat (iclr 2019)
    UrbanCityMulti='UrbanCity_Multi.zip', # env for distraction robustness (icml 2021)
    GarageMulti='ParkingLot_Multi.zip', # env for distraction robustness (icml 2021)
    # multi camera
    UrbanTree='urban_cam.zip', # env for pose-assisted multi-camera tracking (aaai 2020)
    Garden='neighborhood.zip', # env for pose-assisted multi-camera tracking (aaai 2020)
    # Arm env
    Arm='arm-0610.zip',
    test='testfolder.zip'  # this is an empty file used for testing the script
)

binary_win = dict(
    FlexibleRoom='FlexibleRoom_Win_v0.zip',
    UrbanCity='urbancity_win_v1.zip',
    Garage='parkinglot_win_v1.zip',
    Garden='neighborhood_win_v1.zip',
    Textures='Textures.zip'
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("-e", "--env", nargs='?', default='Textures',
                        help='Select the binary to download')
    parser.add_argument("-cloud", "--cloud", nargs='?', default='modelscope',
                        help='Select the cloud to download the binary, modelscope or aliyun')
    args = parser.parse_args()
    if 'win' in sys.platform:
        binary_all = binary_win
    elif 'linux' in sys.platform:
        binary_all = binary_linux

    if args.env in binary_all:
        target_name = binary_all[args.env]
    else:
        print(f"{args.env} is not available to your platform")
        exit()

    if args.cloud == 'modelscope':
        cmd = f"modelscope download --dataset zfw1226/UnrealEnv --include {target_name} --local_dir ."
        try:
            os.system(cmd)
        except:
            print('Please install modelscope first: pip install modelscope')
            exit()
        filename = target_name
    elif args.cloud == 'aliyun':
        import wget
        filename = wget.download(os.path.join(aliyun, binary_all[args.env]))  # download the binary from aliyun

    with zipfile.ZipFile(filename, "r") as z:
        z.extractall()  # extract the zip file
    if 'Textures' in filename:
        folder ='textures'
    else:
        folder = filename[:-4]
    target = 'gym_unrealcv/envs/UnrealEnv'
    shutil.move(folder, target)
    os.remove(filename)

