
from sklearn import metrics
import keras
import numpy as np
from generator.data_generator_1 import data_generator

'''自定义auc callback'''

class Roc_Callback(keras.callbacks.Callback):
    def __init__(self, predict_batch_size=64, include_on_batch=False):
        super(Roc_Callback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_batch_begin(self, batch, logs={}):
        pass

    def on_batch_end(self, batch, logs={}):
        pass

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')
        if not ('final_score' in self.params['metrics']):
            self.params['metrics'].append('final_score')
        logs['final_score'] = float('-inf')

    def on_train_end(self, logs={}):
        pass

    def on_epoch_begin(self, epoch, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        BatchSize = 64
        INPUT_SHAPE = (224, 224, 3)
        annotation = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training-Chinese_modified_v1.xlsx'
        ODIR_valDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_val_aug_bv'
        val_gen = data_generator(annotation,ODIR_valDir,BatchSize,INPUT_SHAPE,mode='valid')

        x_val, y_val = next(val_gen.get_mini_batch())
        predict = self.model.predict_on_batch(x_val)
        steps = val_gen.num_of_images // val_gen.batch_size #有多少个steps
        for i in range(steps):
            if i == 0:
                continue
            datas, labels = next(val_gen.get_mini_batch())
            pre = self.model.predict_on_batch(datas)
            predict = np.vstack((predict, pre))
            y_val = np.vstack((y_val, labels))

        th = 0.5
        gt = y_val.flatten()
        pr = predict.flatten()
        kappa = metrics.cohen_kappa_score(gt, pr > th)
        f1 = metrics.f1_score(gt, pr > th, average='micro')
        auc = metrics.roc_auc_score(gt, pr)
        final_score = (kappa + f1 + auc) / 3.0
        print('val_kappa:      %.3f' % kappa)
        print('val_f1   :      %.3f' % f1)
        print('val_auc  :      %.3f' % auc)
        print('val_final_score:%.3f' % final_score)
        logs['final_score'] = final_score