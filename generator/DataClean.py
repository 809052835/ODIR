import pandas as pd
import numpy as np
import pickle
import cv2
import os
import re

def DataClean(annotation):
    excelData = pd.read_excel(annotation)

    Left_Diagnostic_Keywords = excelData.filter(items=['左眼诊断关键词'])
    Left_Diagnostic_Keywords = Left_Diagnostic_Keywords.values

    Right_Diagnostic_Keywords = excelData.filter(items=['右眼诊断关键词'])
    Right_Diagnostic_Keywords = Right_Diagnostic_Keywords.values

    label = excelData.filter(items=['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'])
    label_orig = label.values

    left_label = []
    right_label = []

    for i, key in enumerate(Left_Diagnostic_Keywords):
        left_keywords = key[0]   #将nparray转成str
        left_init_label = np.array((0, 0, 0, 0, 0, 0, 0, 0))

        if '正常' in left_keywords:   #如果有正常这个关键词，就是正常的
            left_label.append(np.array((1, 0, 0, 0, 0, 0, 0, 0)))
            continue

        if '外眼像' in left_keywords or '无眼底图像' in left_keywords or '无眼底图片' in left_keywords: #"外眼像"和"无眼底图像"不属于8个类别中的任何一类。
            left_label.append(np.array((0, 0, 0, 0, 0, 0, 0, 0)))
            continue

        keywords = re.split('[,，]',left_keywords)
        for j in range(len(keywords)):
            # print(keywords[j])
            if '糖尿病' in keywords[j]:
                left_init_label = np.bitwise_or(left_init_label, np.array((0, 1, 0, 0, 0, 0, 0, 0)))
            elif '青光眼' in keywords[j]:
                left_init_label = np.bitwise_or(left_init_label, np.array((0, 0, 1, 0, 0, 0, 0, 0)))
            elif '白内障' in keywords[j]:
                left_init_label = np.bitwise_or(left_init_label, np.array((0, 0, 0, 1, 0, 0, 0, 0)))
            elif '老年黄斑病变' in keywords[j]:
                left_init_label = np.bitwise_or(left_init_label, np.array((0, 0, 0, 0, 1, 0, 0, 0)))
            elif '高血压' in keywords[j]:
                left_init_label = np.bitwise_or(left_init_label, np.array((0, 0, 0, 0, 0, 1, 0, 0)))
            elif '近视' in keywords[j]:
                left_init_label = np.bitwise_or(left_init_label, np.array((0, 0, 0, 0, 0, 0, 1, 0)))
            elif  "镜头污点" not in keywords[j] and "视盘不可见" not in keywords[j] \
                and "图像质量差" not in keywords[j] and '图片质量差' not in keywords[j] and "图片偏位" not in keywords[j]:
                left_init_label = np.bitwise_or(left_init_label, np.array((0, 0, 0, 0, 0, 0, 0, 1)))

        left_label.append(left_init_label)

    for i, key in enumerate(Right_Diagnostic_Keywords):
        Right_keywords = key[0]   #将nparray转成str
        Right_init_label = np.array((0, 0, 0, 0, 0, 0, 0, 0))

        if '正常' in Right_keywords:   #如果有正常这个关键词，就是正常的
            right_label.append(np.array((1, 0, 0, 0, 0, 0, 0, 0)))
            continue

        if '外眼像' in Right_keywords or '无眼底图像' in Right_keywords or '无眼底图片' in Right_keywords : #"外眼像"和"无眼底图像"不属于8个类别中的任何一类。
            right_label.append(np.array((0, 0, 0, 0, 0, 0, 0, 0)))
            continue

        keywords = re.split('[,，]', Right_keywords)
        for j in range(len(keywords)):
            if '糖尿病' in keywords[j]:
                Right_init_label = np.bitwise_or(Right_init_label, np.array((0, 1, 0, 0, 0, 0, 0, 0)))
            elif '青光眼' in keywords[j]:
                Right_init_label = np.bitwise_or(Right_init_label, np.array((0, 0, 1, 0, 0, 0, 0, 0)))
            elif '白内障' in keywords[j]:
                Right_init_label = np.bitwise_or(Right_init_label, np.array((0, 0, 0, 1, 0, 0, 0, 0)))
            elif '老年黄斑病变' in keywords[j]:
                Right_init_label = np.bitwise_or(Right_init_label, np.array((0, 0, 0, 0, 1, 0, 0, 0)))
            elif '高血压' in keywords[j]:
                Right_init_label = np.bitwise_or(Right_init_label, np.array((0, 0, 0, 0, 0, 1, 0, 0)))
            elif '近视' in keywords[j]:
                Right_init_label = np.bitwise_or(Right_init_label, np.array((0, 0, 0, 0, 0, 0, 1, 0)))
            elif  "镜头污点" not in keywords[j] and "视盘不可见" not in keywords[j] \
                and "图像质量差" not in keywords[j] and '图片质量差' not in keywords[j] and "图片偏位" not in keywords[j]:
                Right_init_label = np.bitwise_or(Right_init_label, np.array((0, 0, 0, 0, 0, 0, 0, 1)))

        right_label.append(Right_init_label)

    num = 0
    id = excelData.filter(items=['编号'])
    id = id.values
    for i, key in enumerate(id):
        predict = np.bitwise_or(left_label[i], right_label[i])
        if predict[0] == 1:
            if np.sum(predict[1:], axis=0) > 0:  # 如果有其他病
                predict[0] = 0  # 正常位置置为0
        if np.sum(predict[:], axis=0) == 0:
            predict[0] = 1 #如果左右眼都无法确定（比如图像质量差），那么就是正常的
        if not np.array_equal(label_orig[i], predict):
            print(key)
            print(left_label[i])
            print(right_label[i])
            print(predict)
            print(label_orig[i])
            print('\n')
            num = num + 1
    if num!=0:
        print('关键词提取错误个数:', num)

    excelData['left_label'] = left_label
    excelData['right_label'] = right_label
    # print(excelData.head())
    return excelData





