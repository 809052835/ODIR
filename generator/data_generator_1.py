import os
import cv2
import random
import pandas as pd
import numpy as np
import re
import time
from generator.DataClean import DataClean
'''生成器生成图片以及这张图片的标签'''

class data_generator:

    def __init__(self,annotation,train_dir,BATCHSIZE,image_size,mode):
        self.mode = mode
        self.data = DataClean(annotation)
        self.index = 0
        self.batch_size = BATCHSIZE
        self.image_size = image_size
        self.load_images_labels(train_dir)

    def load_images_labels(self,train_dir):
        if self.mode=='train':
            print(self.data.head())
            print("Traing data generating...")
        elif self.mode=='valid':
            print("Valid data generating...")
        elif self.mode=='test':
            print('Testing data generating...')

        left_label = self.data['left_label'].values
        right_label = self.data['right_label'].values

        images = []  # 存放图片路径
        labels = []  # 存放标签信息

        img_cvt = set()  #训练集的图片路径
        for i, imgs_name in enumerate(sorted(os.listdir(train_dir), key=lambda x: int(x.split('_')[0]))):
            img_cvt.add(imgs_name)  #记录训练集中所有图片的名称，带有路径


        '''左眼'''
        for i, img_name in enumerate(self.data['左眼眼底图像']):
            fname = img_name.split('.')[0].strip('t')
            for j in img_cvt:  # 候选目录里面的图片名称，有后缀
                img_cvt_name = j.split('t')[0]  # 候选图片前面的几个字符
                if fname == img_cvt_name:   #如果训练集中的候选图片的前缀和标签对应图片的前缀相同，则赋予标签
                    imgPath = os.path.join(train_dir, j)
                    images.append(imgPath)
                    labels.append(np.array(left_label[i]))

        '''右眼'''
        for i, img_name in enumerate(self.data['右眼眼底图像']):
            fname = img_name.split('.')[0].strip('t')
            for j in img_cvt:  # 候选目录里面的图片名称，有后缀
                img_cvt_name = j.split('t')[0]  # 候选图片前面的几个字符
                if fname == img_cvt_name:   #如果训练集中的候选图片的前缀和标签对应图片的前缀相同，则赋予标签
                    imgPath = os.path.join(train_dir, j)
                    images.append(imgPath)
                    labels.append(np.array(right_label[i]))

        self.num_of_images = len(labels)
        self.images = images
        self.labels = labels

        '''一开始先随机打乱'''
        c = list(zip(self.images, self.labels))  # 将图像和标签合并成一个list
        random.shuffle(c)  # 一起打乱
        self.images, self.labels = zip(*c)  # 这样保证images和labels是一一对应的

        if self.mode=='train':
            print('training images: %d' % self.num_of_images)
        elif self.mode=='valid':
            print('valid images   : %d' % self.num_of_images)
        elif self.mode=='test':
            print('test images    : %d' % self.num_of_images)

    def get_mini_batch(self):
        while True:
            batch_images = []
            batch_labels = []

            for i in range(self.batch_size):
                if (self.index == self.num_of_images):
                    if self.mode == 'train':
                        c = list(zip(self.images, self.labels))  # 将图像和标签合并成一个list
                        random.shuffle(c)  # 一起打乱
                        self.images, self.labels = zip(*c)  # 这样保证images和labels是一一对应的
                    self.index = 0

                img = cv2.imread(self.images[self.index], cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.image_size[0], self.image_size[1]))
                batch_images.append(img)
                batch_labels.append(self.labels[self.index])
                self.index += 1
            batch_images = np.array(batch_images)/255.0
            batch_labels = np.array(batch_labels)
            yield batch_images,batch_labels
