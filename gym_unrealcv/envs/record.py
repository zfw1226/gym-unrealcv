from unrealcv_cmd import  UnrealCv
unrealcv = UnrealCv()
unrealcv.set_position(0,-700,-6000,300)
unrealcv.set_rotation(0,0,90,0)
for i in range(10000):
    unrealcv.move(0,0,20)
