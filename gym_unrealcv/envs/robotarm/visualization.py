import cv2
def show_info(info ):
    cv_img = info['Color'].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    height = cv_img.shape[0]
    width = cv_img.shape[1]

    cv2.putText(cv_img, 'Reward:' + str(round(info['Reward'], 3)), (int(3*width/10), int(9*height/10)), font, 0.5 * cv_img.shape[1]/640.0, (255, 255, 255), 1)

    #action
    action_x = int(5*width/10)
    action_y = int(9*height/10)
    cv2.putText(cv_img, 'Action:' + str(info['Action']), (action_x, action_y), font, 0.5 * cv_img.shape[1]/640.0, (255, 255, 255), 1)

    step_x = int(7*width/10)
    step_y = int(9*height/10)
    cv2.putText(cv_img, 'Steps:' + str(info['Steps']), (step_x, step_y), font, 0.5 * cv_img.shape[1]/640.0, (255, 255, 255), 1)

    collision_x = int(1*width/10)
    collision_y = int(9*height/10)
    rad = int(15 * cv_img.shape[1] / 640.0)
    if info['Collision']:
        cv2.putText(cv_img, 'Collision', (collision_x, collision_y), font, 0.5 * cv_img.shape[1]/640.0, (0, 0, 255), 1)
        cv2.circle(cv_img, (collision_x + int(width * 0.05), collision_y - rad*2), rad, (0, 0, 255), -1)
    else:
        cv2.putText(cv_img, 'Collision', (collision_x, collision_y), font, 0.5 * cv_img.shape[1]/640.0, (0, 255, 0), 1)
        cv2.circle(cv_img, (collision_x + int(width * 0.05), collision_y - rad*2), rad, (0, 255, 0), -1)

    for box in info['Bbox']:
            xmin = int(box[0][0] * width)
            xmax = int(box[1][0] * width)
            ymin = int(box[0][1] * height)
            ymax = int(box[1][1] * height)
            cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (50, 255, 50))

    cv2.imshow('info_show', cv_img)
    cv2.waitKey(10)
