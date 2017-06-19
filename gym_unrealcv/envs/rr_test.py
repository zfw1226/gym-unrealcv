from unrealcv_cmd import  UnrealCv
import time
import run_docker
import yaml
'''
A toolkit for sampling  bounding box ground truth in environment
sample image every 10s, boxes(percent) are saved in yaml format.
'''

'''TARGETS = [ 'SM_CoffeeTable_14', 'Couch_13','SM_Couch_1seat_5','Statue_48','SM_TV_5', 'SM_DeskLamp_5',
 'SM_Plant_7', 'SM_Plant_8', 'SM_Door_37', 'SM_Door_39', 'SM_Door_41']'''

ENV_NAME = 'RealisticRendering'
cam_id = 0


#start docker
docker = run_docker.RunDocker()
env_ip, env_dir = docker.start(ENV_NAME=ENV_NAME)
print env_ip
print env_dir
#connect unrealcv
unrealcv = UnrealCv(cam_id, ip=env_ip, targets='all', env=env_dir)

TARGETS = unrealcv.targets

setting = dict()
#TARGETS = unrealcv.get_objects()
setting['targets'] = TARGETS
setting['env_name'] = ENV_NAME
setting['cam_id'] = cam_id
setting['num'] = 5
for i in range(setting['num']):
    info =dict()
    info['position'] = unrealcv.get_position(cam_id)
    info['rotation'] = unrealcv.get_rotation(cam_id)
    object_mask = unrealcv.read_image(cam_id, 'object_mask')
    info['width'] = object_mask.shape[0]
    info['height'] = object_mask.shape[1]
    bboxes = unrealcv.get_bboxes_obj(object_mask, TARGETS)
    for (obj,box) in bboxes.items():
        (xmin, ymin), (xmax, ymax) = box
        boxarea = (ymax - ymin) * (xmax - xmin)
        if boxarea > 0:
            box = dict()
            box['bbox'] = [float(xmin),float(ymin),float(xmax),float(ymax)]
            box['area'] = float(boxarea)
            info[obj] = box
    print info
    setting[i] = info
    lit = unrealcv.read_image(cam_id, 'lit')
    time.sleep(10)

print setting
f = open('rr_bbox_all.yaml','w')
f.write(yaml.dump(setting))
f.close()

docker.close()

def save_setting(dict,filename):
    f = open(filename,'w')
    f.write(yaml.dump(dict))
    f.close()