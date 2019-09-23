import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img,save_img

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'  #只显示warning 和 error

def data_augmentation(img_dir,save_dir):
    datagen = ImageDataGenerator(
        zca_epsilon=False,
        zca_whitening=0,
        rotation_range = 40,    #在0~45度的范围内随机旋转
        shear_range=0.1, #shear_range就是错切变换，效果就是让所有点的x坐标(或者y坐标)保持不变，
                        # 而对应的y坐标(或者x坐标)则按比例发生平移，且平移的大小和该点到x轴(或y轴)的垂直距离成正比。
        width_shift_range= 0.05,     #水平偏移的幅度
        height_shift_range= 0.05,    #垂直偏移的幅度
        channel_shift_range=10,
        brightness_range=(0.15,1.3),

        horizontal_flip =False,     #图像和左右位置有关
        vertical_flip=False,        #一般眼底图没有上下翻转过来的
        fill_mode='constant'
    )

    for i, img_name in enumerate(sorted(os.listdir(img_dir), key=lambda x: int(x.split('_')[0]))):
        fname = img_name.split('.')[0]
        # print(fname)
        img_path = os.path.join(img_dir,img_name)
        img = load_img(img_path)
        save_img(os.path.join(save_dir,img_name),img)
        img = img_to_array(img)
        # print(img.shape)
        # print(type(img))
        x=img.reshape((1,)+ img.shape)

        num = 0
        for batch in datagen.flow(x,batch_size=1,save_to_dir=save_dir,save_prefix=fname,save_format='jpg'):
            num+=1
            if num>=7:  #控制每一张图片生成的假样本的个数
                break


if __name__ == '__main__':
    oriImgDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train'
    saveDir   = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train_aug'
    data_augmentation(oriImgDir,saveDir)

    oriImgDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val'
    saveDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug'
    data_augmentation(oriImgDir, saveDir)


