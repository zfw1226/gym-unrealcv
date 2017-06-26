from unrealcv_cmd import  UnrealCv
import time
unrealcv = UnrealCv()
#unrealcv.set_position(0,-530,-3720,90)
unrealcv.set_rotation(0,0,-180,350)
for i in range(10000):
    unrealcv.move(0,0,20)
    time.sleep(0.2)
