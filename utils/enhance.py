import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'  #只显示warning 和 error

def enhance_gray(img_dir,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, img_name in enumerate(sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[0]))):
        # fname = img_name.split('.')[0]
        img_path = os.path.join(img_dir, img_name)
        new_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # --------对灰度图进行直方图均衡化------
        b, green_fundus, r = cv2.split(new_img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced_green_fundus = clahe.apply(green_fundus)
        # print(os.path.join(save_dir, img_name))
        cv2.imwrite(os.path.join(save_dir, img_name), contrast_enhanced_green_fundus)

def enhance_color(img_dir,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i, img_name in enumerate(sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[0]))):
        # fname = img_name.split('.')[0]
        img_path = os.path.join(img_dir, img_name)
        new_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # -----对彩色图像进行直方图均衡化-------
        # 将RGB图像转换到YCrCb空间中
        ycrcb = cv2.cvtColor(new_img,cv2.COLOR_BGR2YCR_CB)
        #将YCR_CB图像通道分离
        channels = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(channels[0],channels[0])
        #将处理后的通道和没有处理的两个通道合并，命名为ycrcb
        cv2.merge(channels,ycrcb)
        #将ycrcb图像转换成RGB图像
        equalization = cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR)
        cv2.imwrite(os.path.join(save_dir, img_name), equalization)



if __name__ == '__main__':
    oriImgDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train_aug'
    saveDir_g = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train_aug_equ_g'
    saveDir_c = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train_aug_equ_c'
    enhance_gray(oriImgDir, saveDir_g)
    enhance_color(oriImgDir,saveDir_c)

    oriImgDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug'
    saveDir_g = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug_equ_g'
    saveDir_c = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug_equ_c'
    enhance_gray(oriImgDir, saveDir_g)
    enhance_color(oriImgDir, saveDir_c)