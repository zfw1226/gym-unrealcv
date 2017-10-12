import cv2
import numpy as np
import keras.backend as K

class preprocessor():
    def __init__(self,observation_space, length = 3, size = (84,84) ):
        self.length = length
        self.previous = np.zeros((1,length,size[0],size[1]))
        self.size = size
        self.image_channels = observation_space.shape[2]
        self.image_high = observation_space.high
        self.image_low = observation_space.low
        self.image_range = self.image_high - self.image_low

    def resize(self,image):
        cv_image = cv2.resize(image, self.size)
        return cv_image

    def color2gray(self,image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray_image

    def reshape(self,image):
        reshape_image = image.reshape(1, self.image_channels, self.size[0], self.size[1])
        if K.image_dim_ordering() == 'tf':
            reshape_image= reshape_image.transpose(0, 2, 3, 1)
        return reshape_image

    def normalize(self,image):
        normalized_image = (image - self.image_low) / (self.image_range)
        return  normalized_image

    def process_gray(self,image, reset= False):
        resize_image = self.resize(image)
        gray_image = self.color2gray(resize_image)
        gray_image = gray_image/255.0
        if reset:
            #print 'reset'
            for i in range(self.length):
                self.previous[0][i] = gray_image
        else:
            #print 'update'
            self.previous = np.insert(self.previous, 0, gray_image, axis=1)
            self.previous = np.delete(self.previous, -1, axis=1)


        #print self.previous.shape
        if K.image_dim_ordering() == 'tf':
            processed = self.previous.transpose(0, 2, 3, 1)
        else:
            processed = self.previous
        return processed

        # print img_processed.shape
        return img_processed

