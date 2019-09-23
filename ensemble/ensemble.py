from ensemble.lgb_xgb_cat import lgb_binary
import  pandas as pd
import numpy as np


if __name__ == '__main__':
    # X_train, X_val, y_train, y_val = train_test_split(x, yy0, test_size=0.2,random_state=2019)
    merge_pred_path_val = r'H:\Qiulin\ODIR\predict\v6\merge_pred.csv'
    merge_pred_val = pd.read_csv(merge_pred_path_val)   #三个模型预测值
    x0 = merge_pred_val.values

    merge_pred_path_test = r'H:\Qiulin\ODIR\predict\v6\merge_pred.csv'
    merge_pred_test = pd.read_csv(merge_pred_path_test)  # 三个模型预测值
    x_test = merge_pred_test.values


    label_path = r'H:\Qiulin\ODIR\predict\v6\final_label.csv'
    label = pd.read_csv(label_path)
    y = label.values    #病人的标签

    "N 0"
    x = x0.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y0 = y[:, 0:1]
    yy0 = []
    for i in y0:
        yy0.append(int(i))
    y_train = yy0[:2520]
    y_val = yy0[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_0 = x_test.tolist()
    predict0 = lgb_model.predict(x_test_0)

    'D 1'
    x = x0[:,[1,9,17,25,33,41]]
    x = x.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y1 = y[:,1:2]
    yy1 = []
    for i in y1:
        yy1.append(int(i))
    y_train = yy1[:2520]
    y_val = yy1[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_1 = x_test[:,[1,9,17,25,33,41]]
    x_test_1 = x_test_1.tolist()
    predict1 = lgb_model.predict(x_test_1)

    'G 2'
    x = x0[:, [2,10,18,26,34,42]]
    x = x.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y2 = y[:, 2:3]
    yy2 = []
    for i in y2:
        yy2.append(int(i))
    y_train = yy2[:2520]
    y_val = yy2[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_2 = x_test[:, [2,10,18,26,34,42]]
    x_test_2 = x_test_2.tolist()
    predict2 = lgb_model.predict(x_test_2)

    'C 3'
    x = x0[:, [3,11,19,27,35,43]]
    x = x.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y3 = y[:, 3:4]
    yy3 = []
    for i in y3:
        yy3.append(int(i))
    y_train = yy3[:2520]
    y_val = yy3[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_3 = x_test[:, [3,11,19,27,35,43]]
    x_test_3 = x_test_3.tolist()
    predict3 = lgb_model.predict(x_test_3)

    'A 4'
    x = x0[:, [4,12,20,28,36,44]]
    x = x.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y4 = y[:, 4:5]
    yy4 = []
    for i in y4:
        yy4.append(int(i))
    y_train = yy4[:2520]
    y_val = yy4[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_4 = x_test[:, [4,12,20,28,36,44]]
    x_test_4 = x_test_4.tolist()
    predict4 = lgb_model.predict(x_test_4)

    'H 5'
    x = x0[:, [5,13,21,29,37,45]]
    x = x.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y5 = y[:, 5:6]
    yy5 = []
    for i in y5:
        yy5.append(int(i))
    y_train = yy5[:2520]
    y_val = yy5[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_5 = x_test[:, [5,13,21,29,37,45]]
    x_test_5 = x_test_5.tolist()
    predict5 = lgb_model.predict(x_test_5)

    'M 6'
    x = x0[:, [6,14,22,30,38,46]]
    x = x.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y6 = y[:, 6:7]
    yy6 = []
    for i in y6:
        yy6.append(int(i))
    y_train = yy6[:2520]
    y_val = yy6[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_6 = x_test[:, [6,14,22,30,38,46]]
    x_test_6 = x_test_6.tolist()
    predict6 = lgb_model.predict(x_test_6)

    'O 7'
    x = x0[:, [7,15,23,31,39,47]]
    x = x.tolist()
    X_train = x[:2520]
    X_val = x[2520:]
    y7 = y[:, 7:8]  #其他病
    yy7 = []
    for i in y7:
        yy7.append(int(i))
    y_train = yy7[:2520]
    y_val = yy7[2520:]
    lgb_model = lgb_binary(X_train, X_val, y_train, y_val, merge_pred_val.columns, True)
    x_test_7 = x_test[:, [7,15,23,31,39,47]]
    x_test_7 = x_test_7.tolist()
    predict7 = lgb_model.predict(x_test_7)


