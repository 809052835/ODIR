import os
import cv2
import random
import pandas as pd
import numpy as np
import re
import time
from generator.DataClean import DataClean

'''测试用：生成器生成左眼图片，右眼图片给CNN模型做预测'''

class data_generator:
    def __init__(self,val_dir,BATCHSIZE,image_size,mode):
        self.mode = mode
        self.index = 0
        self.batch_size = BATCHSIZE
        self.image_size = image_size
        self.load_images_labels(val_dir)

    def load_images_labels(self,val_dir):
        left_path = []
        right_path = []
        for i, imgs_name in enumerate(sorted(os.listdir(val_dir), key=lambda x: int(x.split('_')[0]))):

            imgs_path = os.path.join(val_dir,imgs_name)
            if 'left' in imgs_name:
                left_path.append(imgs_path)
            elif 'right' in imgs_name:
                right_path.append(imgs_path)

        images = left_path+right_path
        self.num_of_images = len(images)
        self.images = images
        print('test images    : %d' % self.num_of_images)

    def get_mini_batch(self):
        while True:
            batch_images = []
            for i in range(self.batch_size):
                if (self.index == self.num_of_images):
                    break
                img = cv2.imread(self.images[self.index], cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
                batch_images.append(img)
                self.index += 1
            batch_images = np.array(batch_images) / 255.0
            yield batch_images


