from generator.data_generator_2 import data_generator
from user_defined.losses import myLoss,Focal_loss_v1
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import pickle
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''将验证集的数据给基模型预测，得到左右眼在三个模型的预测结果'''

def predict_val(testDir,model_path):
    BATCHSIZE =16
    INPUT_SHAPE = (224, 224, 3)

    '''创建生成器'''
    val_gen = data_generator(testDir,BATCHSIZE,INPUT_SHAPE,'test')

    '''loadModel and predict'''
    val_model = load_model(model_path,custom_objects={'Focal_loss_v1':Focal_loss_v1})
    steps = val_gen.num_of_images // val_gen.batch_size
    prediction = val_model.predict_generator(val_gen.get_mini_batch(),verbose=0,steps=steps)
    print(prediction.shape)

    minMax = MinMaxScaler()
    pred_std = minMax.fit_transform(prediction)
    mid = int(pred_std.shape[0]/2)
    left = pred_std[:mid,:]
    right = pred_std[mid:,:]
    pred_Matrix = np.hstack([left,right])
    print('left.shape',left.shape)
    print('right.shape',right.shape)
    print('pred_Matrix.shape',pred_Matrix.shape)
    return pred_Matrix

if __name__ == '__main__':
    valDir_g = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug_equ_g'
    model_path_g = r'H:\Qiulin\ODIR\model\v6\vgg16_g_v2.h5'
    pred_Matrix_g = predict_val(valDir_g,model_path_g)

    valDir_c = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug_equ_c'
    model_path_c = r'H:\Qiulin\ODIR\model\v6\vgg16_c_v1.h5'
    pred_Matrix_c = predict_val(valDir_c, model_path_c)

    valDir_bv = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug_bv'
    model_path_bv = r'H:\Qiulin\ODIR\model\v6\vgg16_bv_v1.h5'
    pred_Matrix_bv = predict_val(valDir_bv, model_path_bv)

    '''将三个模型生成的结果合并起来'''
    merge_pred = np.hstack([pred_Matrix_g,pred_Matrix_c,pred_Matrix_bv])
    pred_header = [
                   'GN_L', 'GD_L', 'GG_L', 'GC_L', 'GA_L', 'GH_L', 'GM_L', 'GO_L',
                   'GN_R', 'GD_R', 'GG_R', 'GC_R', 'GA_R', 'GH_R', 'GM_R', 'GO_R',
                   'CN_L', 'CD_L', 'CG_L', 'CC_L', 'CA_L', 'CH_L', 'CM_L', 'CO_L',
                   'CN_R', 'CD_R', 'CG_R', 'CC_R', 'CA_R', 'CH_R', 'CM_R', 'CO_R',
                   'BVN_L', 'BVD_L', 'BVG_L', 'BVC_L', 'BVA_L', 'BVH_L', 'BVM_L','BVO_L',
                   'BVN_R', 'BVD_R', 'BVG_R', 'BVC_R', 'BVA_R', 'BVH_R', 'BVM_R','BVO_R'
                   ]
    df = pd.DataFrame(merge_pred, columns=pred_header)
    df.to_csv(r'H:\Qiulin\ODIR\predict\v6\merge_pred.csv', encoding='utf-8', index=False)

    '''提取病人的标签信息'''
    final_label = []    #记录病人label
    annotation = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training-Chinese_modified_v1.xlsx'
    excelData = pd.read_excel(annotation)
    label = excelData.filter(items=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    label_orig = label.values
    id = excelData.filter(items=['编号'])
    id = id.values  #病人编号
    for i, imgs_name in enumerate(sorted(os.listdir(valDir_g), key=lambda x: int(x.split('_')[0]))):    #根据病人编号来提取标签
        if 'left' in imgs_name:
            Id = int(imgs_name.split('_')[0])
            for key, value in enumerate(id):
                if Id==int(value):  #如果文件夹中病人编号和注释中的编号一致
                    final_label.append(label_orig[key]) #提取注释中的病人标签
                    # print(label_orig[key])
    final_label = np.array(final_label) #将标签转化为数组
    # print(final_label.shape)
    label_header = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    df = pd.DataFrame(final_label,columns=label_header)
    df.to_csv(r'H:\Qiulin\ODIR\predict\v6\final_label.csv',encoding='utf-8',index=False)










