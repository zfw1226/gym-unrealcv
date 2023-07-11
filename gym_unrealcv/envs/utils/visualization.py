import numpy as np
import matplotlib.pyplot as plt

def map_render(camera_pos, target_pos, choose_ids, target_move, camera_move, scale_rate, pose_rate):

    length = 600
    coordinate_delta = np.mean(np.array(camera_pos)[:, :2], axis=0)
    img = np.zeros((length + 1, length + 1, 3)) + 255
    num_cam = len(camera_pos)

    camera_position_origin = np.array([camera_pos[i][:2] for i in range(num_cam)])
    target_position_origin = np.array(target_pos[:2])

    lengths = []
    for i in range(num_cam):
        length = np.sqrt(sum(np.array(camera_position_origin[i] - coordinate_delta)) ** 2)
        lengths.append(length)
    pose_scale = max(lengths)

    pose_scale = pose_scale * pose_rate
    target_position = length * (np.array([scale_rate + (target_position_origin[0] - coordinate_delta[0]) / pose_scale, scale_rate +
                                          (target_position_origin[1] - coordinate_delta[0]) / pose_scale])) / 2 + np.array(target_move)

    camera_position = []
    for i in range(num_cam):
        position_transfer = length * (np.array([scale_rate + (camera_position_origin[i][0] - coordinate_delta[0]) / pose_scale,
                                                scale_rate + (camera_position_origin[i][1] - coordinate_delta[1]) / pose_scale])) / 2 + np.array(camera_move)
        camera_position.append(position_transfer)

    abs_angles = [camera_pos[i][4] for i in range(num_cam)]

    color_dict = {'red': [255, 0, 0], 'black': [0, 0, 0], 'blue': [0, 0, 255], 'green': [0, 255, 0],
                  'darkred': [128, 0, 0], 'yellow': [255, 255, 0], 'deeppink': [255, 20, 147]}

    # plot camera
    for i in range(num_cam):
        img[int(camera_position[i][1])][int(camera_position[i][0])][0] = color_dict["black"][0]
        img[int(camera_position[i][1])][int(camera_position[i][0])][1] = color_dict["black"][1]
        img[int(camera_position[i][1])][int(camera_position[i][0])][2] = color_dict["black"][2]

    # plot target
    img[int(target_position[1])][int(target_position[0])][0] = color_dict['blue'][0]
    img[int(target_position[1])][int(target_position[0])][1] = color_dict['blue'][1]
    img[int(target_position[1])][int(target_position[0])][2] = color_dict['blue'][2]

    plt.cla()
    plt.imshow(img.astype(np.uint8))

    # get camera's view space positions
    visua_len = 60
    for i in range(num_cam):
        theta = abs_angles[i] + 90.0
        dx = visua_len * math.sin(theta * math.pi / 180)
        dy = - visua_len * math.cos(theta * math.pi / 180)
        plt.arrow(camera_position[i][0], camera_position[i][1], dx, dy, width=0.1, head_width=8,
                  head_length=8, length_includes_head=True)

        plt.annotate(str(i), xy=(camera_position[i][0], camera_position[i][1]),
                     xytext=(camera_position[i][0], camera_position[i][1]), fontsize=10, color='blue')

        # top-left
        if int(choose_ids[i]) == 0:
            plt.annotate('cam {0} use pose'.format(i), xy=(camera_position[i][0], camera_position[i][1]),  xytext=(350, (1 + i) * 50 + 250), fontsize=10, color='red')
        else:
            plt.annotate('cam {0} use vision'.format(i), xy=(camera_position[i][0], camera_position[i][1]), xytext=(350, (1 + i) * 50 + 250), fontsize=10,
                         color='blue')

    plt.plot(target_position[0], target_position[1], 'ro')
    plt.title("Top-view")
    plt.xticks([])
    plt.yticks([])
    plt.pause(0.01)