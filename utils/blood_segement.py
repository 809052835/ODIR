import os
import cv2
import numpy as np
import pandas as pd
from math import sqrt,pow

def blood_vessel(oriDir,saveDir):
    for i, img_name in enumerate(sorted(os.listdir(oriDir), key=lambda x: int(x.split('_')[0]))):
        img_path = os.path.join(oriDir, img_name)
        ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        b, green_fundus, r = cv2.split(ori_img)
        inverted_green = 255 - green_fundus
        clear_img = cv2.GaussianBlur(inverted_green,ksize=(3,3),sigmaX=0.45,sigmaY=0.45)
        kernel = np.ones((7,7),np.uint8)
        top_hat = cv2.morphologyEx(clear_img,cv2.MORPH_TOPHAT,kernel)

        sort_TH = np.sort(top_hat.flatten())
        max = np.max(sort_TH)
        min = np.min(sort_TH)
        num = int(sort_TH.shape[0]*0.04)
        lower = sort_TH[num]
        upper = sort_TH[-num]
        top_hat[top_hat <= lower] = min #对顶帽操作之后的图片做对比度增强
        top_hat[top_hat >= upper] = max
        print('min', np.min(top_hat))
        print('max', np.max(top_hat))
        enhance = top_hat

        hist,grayValue = calHist(enhance)  #得到直方图，即灰度概率分布，grayValue就是灰度值
        t_globle = CDF(hist)   #根据累积概率分布函数得到灰度阈值分割的最佳阈值
        binary,output = cv2.threshold(enhance,t_globle,255,cv2.THRESH_BINARY)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))  #十字结构
        eroded1 = cv2.erode(output, kernel1)
        cv2.imwrite(os.path.join(saveDir, img_name), eroded1)


'''计算直方图并返回'''
def calHist(grayIMG):
    grayValue = set()
    hist = {}
    newhist = {}
    for i in grayIMG.flatten(): #统计每一个灰度值出现的次数
        hist[i] = hist.get(i,0) + 1

    for i in range(256):    #将字典按照键从小到大排序
        if i in hist:
            newhist[i] = hist[i]
    for key in newhist.keys():
        grayValue.add(key)
    return newhist,grayValue

'''计算累积分布函数，返回图像分割的最佳阈值'''
def CDF(hist):
    cdf = {}    #累积概率分布函数
    for key,value in hist.items():
        temp = dict((k,v) for k,v in hist.items() if k<=key)
        sum = 0
        for i in temp:
            sum = sum + temp[i]
        cdf[key] = sum

    delta = max(cdf.keys())-min(cdf.keys())
    T = {}  #记录灰度阈值分割的候选阈值，然后比较哪一个最大
    for key,value in cdf.items():
        T[key] = abs(cdf[key] + delta*(1-key) -1 )/sqrt((1+pow(delta,2)))
    threshold = max(T,key=T.get)
    MaxGray = max(cdf.keys())
    if threshold == MaxGray:
        threshold = threshold - 1
    return threshold

'''计算灰度共生矩阵
    grayIMG:输入的灰度图像
    dx:x轴方向的dixtance
    dy:y轴方向的dixtance
    gray_level：灰度级，默认为256'''
def GLCM(grayIMG,dx,dy,gray_level):
    # rows_max = 0
    # cols_max = 0
    # max_gray = grayIMG.max()    #求出灰度的最大值
    height,width = grayIMG.shape    #灰度图的大小
    arr = grayIMG.astype(np.float64)    #将灰度图的数据类型从整形转换成float类型
    # arr = arr*(gray_level - 1)//max_gray    #将灰度值归一化
    ret = np.zeros((gray_level+1,gray_level+1),dtype=np.int32)  #创建灰度共生矩阵
    for j in range(height-abs(dy)):
        for i in range(width-abs(dx)):
            rows = arr[j][i].astype(int)
            cols = arr[j + dy][i + dx].astype(int)
            # if rows>rows_max:
            #     rows_max=rows
            # if cols>cols_max:
            #     cols_max=cols
            ret[rows][cols] +=1
    if dx >= dy:
        ret = ret / float(height * (width - 1))  # 归一化, 水平方向或垂直方向
    else:
        ret = ret / float((height - 1) * (width - 1))  # 归一化, 45度或135度方向
    # print('sum_ret',np.sum(ret))  #检验一下是不是ret的和是1
    # print('cols_max',cols_max)
    # print('rows_max',rows_max)
    return ret

'''计算一个像素属于背景的概率，返回P_bg'''
def sum_Pxy(P_xy,threshold):
    width,height = P_xy.shape
    P_bg = 0
    for i in range(width):
        for j in range(height):
            P_bg = P_bg + P_xy[i][j]
            if j>=threshold:
                break
        if i>=threshold:
            break
    return P_bg

def corr(P_xy,grayIMG,P_bg,t_globle):
    n=0
    width,height = grayIMG.shape
    mask = np.zeros((width,height))
    for i in range(1,width-1,1):
        for j in range(1,height-1,1):
            Ew = 0
            corr = 0
            for u in range(3):
                for v in range(3):
                    x = y = grayIMG[i-1+u,j-1+v]
                    if x>t_globle:
                        Ew += 1
                    gx = grayIMG[i-1+u , j-1:j+2]   #x轴方向的的三个元素
                    gy = grayIMG[i-1:i+2 , j-1+v]   #Y轴方向的三个元素
                    ux = np.mean(gx)
                    uy = np.mean(gy)
                    thetaX = np.std(gx)
                    thetaY = np.std(gy)
                    a = (x-ux)*(y-uy)
                    b = thetaX * thetaY
                    if a==0 or b==0:
                        continue
                    corr = corr + P_xy[x,y]*(a/b)
            t_sd = int(P_bg*corr*9)
            # print('Ew  :', Ew)
            # print('t_sd:',t_sd)
            if Ew<t_sd:     #属于组织，用黑色0表示
                mask[i,j] = 0
                n +=1
            else:   #属于背景，用黑色；来表示
                mask[i,j] = 255
    print('n',n)
    return mask


if __name__ == '__main__':
    oriPath = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train_aug'
    saveDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train_aug_bv'
    blood_vessel(oriPath,saveDir)

    oriPath = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug'
    saveDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug_bv'
    blood_vessel(oriPath, saveDir)