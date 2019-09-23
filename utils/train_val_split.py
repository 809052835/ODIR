import os
import random
import shutil

def train_val_split(oriDir,trainDir,valDir,val_size=0.1):  #这个函数的做法是把训练集中的图片剪切过去验证集
    assert  os.path.exists(oriDir)
    if (not os.path.exists(trainDir)):
        os.mkdir(trainDir)
    if (not os.path.exists(valDir)):
        os.mkdir(valDir)

    left = []
    right = []
    for i, fname in enumerate(sorted(os.listdir(oriDir), key=lambda x: int(x.split('_')[0]))):
        if 'left' in fname:
            left.append(fname)
        elif 'right' in fname:
            right.append(fname)
        else:
            assert True
    c = list(zip(left, right))  # 将左眼和右眼并成一个list
    random.shuffle(c)  # 一起打乱
    left,right = zip(*c)  # 这样保证左右眼是一一对应的

    ori_num_of_images = len(left)+len(right)     #ori中图片数量
    val_num_of_images = int(val_size * ori_num_of_images/2) #验证集中需要划分的图片数量

    left_valid = left[ :val_num_of_images]
    left_train = left[val_num_of_images:]

    right_valid = right[:val_num_of_images]
    right_train = right[val_num_of_images:]

    valid_images = left_valid+right_valid
    train_images = left_train+right_train

    for img in valid_images:
        val_img_path = os.path.join(oriDir,img)
        shutil.copy2(val_img_path,valDir)  #将图片复制到验证集文件夹
    for img in train_images:
        train_img_path = os.path.join(oriDir,img)
        shutil.copy2(train_img_path,trainDir)

if __name__ == '__main__':
    oriDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v2'
    trainDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train'
    valDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val'
    train_val_split(oriDir,trainDir,valDir)
