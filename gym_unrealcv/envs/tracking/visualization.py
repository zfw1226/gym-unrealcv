import cv2
def show_info(info, action_type='discrete'):
    cv_img = info['Color'].copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    height = cv_img.shape[0]
    width = cv_img.shape[1]

    cv2.putText(cv_img, 'Reward:' + str(round(info['Reward'], 3)), (int(3*width/10), int(9*height/10)), font, 0.5, (255, 255, 255), 2)

    #action
    if action_type =='discrete':
        action_x = int(5*width/10)
        action_y = int(9*height/10)
        cv2.putText(cv_img, 'Action', (action_x, action_y), font, 0.5, (255, 255, 255), 2)
        color = [(255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)]
        if info['Action'] == 0:
            color[0] = (0, 0, 255)
        elif info['Action'] == 1:
            color[1] = (0, 0, 255)
        elif info['Action'] == 2:
            color[2] = (0, 0, 255)
        elif info['Action'] == 3:
            color[3] = (0, 0, 255)
        elif info['Action'] == 4:
            color[4] = (0, 0, 255)

        cv2.circle(cv_img, (action_x + int(width*0.04),    action_y - int(height*0.15)), 8, color[0], -1) #forward
        cv2.circle(cv_img, (action_x+10 + int(width*0.04), action_y - int(height*0.10)), 8, color[1], -1) #right
        cv2.circle(cv_img, (action_x-10 + int(width*0.04), action_y - int(height*0.10)), 8, color[2], -1) #left
        cv2.circle(cv_img, (action_x+20 + int(width*0.04), action_y - int(height*0.05)), 8, color[3], -1)  # right
        cv2.circle(cv_img, (action_x-20 + int(width*0.04), action_y - int(height*0.05)), 8, color[4], -1)  # left
    else:
        action_x = int(5*width/10)
        action_y = int(9*height/10)
        (velocity, angle) = info['Action']
        cv2.putText(cv_img, 'V:{}'.format(velocity), (action_x, action_y- int(height*0.07)), font, 0.5, (255, 255, 255), 2)
        cv2.putText(cv_img, 'A:{}'.format(angle), (action_x, action_y - int(height*0.15)), font, 0.5, (255, 255, 255), 2)


    collision_x = int(1*width/10)
    collision_y = int(9*height/10)
    if info['Collision']:
        cv2.putText(cv_img, 'Collision', (collision_x, collision_y), font, 0.5, (0, 0, 255), 2)
        cv2.circle(cv_img, (collision_x + int(width * 0.05), collision_y - 30), 15, (0, 0, 255), -1)
    else:
        cv2.putText(cv_img, 'Collision', (collision_x, collision_y), font, 0.5, (0, 255, 0), 2)
        cv2.circle(cv_img, (collision_x + int(width * 0.05), collision_y - 30), 15, (0, 255, 0), -1)


    cv2.imshow('info_show', cv_img)
    cv2.waitKey(1)

