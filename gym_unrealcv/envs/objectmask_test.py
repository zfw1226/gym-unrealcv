from unrealcv_cmd import  UnrealCv
import time
import run_docker
import yaml


def IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # print xA,xB,yA,yB

    if xB - xA > 0 and yB - yA > 0:
        interArea = (xB - xA + 1) * (yB - yA + 1)
    else:
        interArea = 0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


f = open('rr_bbox_all.yaml')
setting = yaml.load(f)
ENV_NAME = setting['env_name']
TARGETS = setting['targets']
cam_id = setting['cam_id']

#start docker
docker = run_docker.RunDocker()
env_ip, env_dir = docker.start(ENV_NAME=ENV_NAME)
print env_ip
print env_dir
#connect unrealcv
unrealcv = UnrealCv(cam_id, ip=env_ip, targets=TARGETS, env=env_dir)

num = setting['num']
print 'num:' + str(num)
count_match = 0
count_mismatch = 0
for i in range(int(num)):
    info = setting[i]
    x,y,z = info['position']
    unrealcv.set_position(cam_id,x,y,z)
    roll,yaw,pitch = info['rotation']
    unrealcv.set_rotation(cam_id,roll,yaw,pitch)
    object_mask = unrealcv.read_image(cam_id, 'object_mask')
    bboxes = unrealcv.get_bboxes_obj(object_mask, TARGETS)
    for (obj,box) in bboxes.items():
        (xmin, ymin), (xmax, ymax) = box
        boxarea = (ymax - ymin) * (xmax - xmin)
        if boxarea > 0:
            box = dict()
            box['bbox'] = [float(xmin),float(ymin),float(xmax),float(ymax)]
            match = False
            if info.has_key(obj) :
                if IOU(box['bbox'],info[obj]['bbox']) > 0.9:
                    match = True
            if match:
                count_match += 1
            else:
                count_mismatch += 1

print str(count_match) + 'objects matched!'
print str(count_mismatch) + 'objects mismatched!'

docker.close()









