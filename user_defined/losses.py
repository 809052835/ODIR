import keras.backend as K
import tensorflow as tf
import numpy as np


#keras自带交叉熵的loss
def loss_v0(y_true, y_pred,BatchSize):
   loss_weight = np.array((1, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 1))
   # loss_weight = np.array((1, 1, 1, 1, 1, 1, 1, 1))
   loss_weight = K.variable(loss_weight)
   b = K.binary_crossentropy(y_true, y_pred)
   # print('b.shape',b.shape)
   loss_0 = K.mean(b * loss_weight, axis=-1)
   # print('loss_0.shape', loss_0.shape)
   return loss_0

def myLoss(y_true, y_pred):
   batchSize = 32
   loss = loss_v0(y_true, y_pred, batchSize)
   return loss


#考虑将kappa加入到优化目标中
def kappa_loss(y_pred, y_true, eps=1e-5):
   y_true = tf.to_float(y_true)
   y_pred = tf.clip_by_value(tf.sign(y_pred - 0.5), 0, 1)

   total = tf.reduce_sum(tf.ones_like(y_true), axis=1)
   p0 = tf.reduce_sum(tf.where(tf.equal(y_true, y_pred), tf.ones_like(y_pred), tf.zeros_like(y_pred)),
                      axis=1) / total
   a1 = tf.reduce_sum(y_true, axis=1)
   a0 = tf.subtract(total, a1)
   b1 = tf.reduce_sum(y_pred, axis=1)
   b0 = tf.subtract(total, b1)
   pe = (tf.multiply(a0, b0) + tf.multiply(a1, b1)) / (tf.multiply(total, total))

   kappa = tf.subtract(p0, pe) / tf.subtract(tf.ones_like(pe), pe)
   kappa = tf.add(kappa, tf.ones_like(kappa)) / 2
   k_loss = - (1 - kappa) * tf.log(kappa + eps)
   k_loss = tf.clip_by_value(k_loss, 0, 1e5)
   return k_loss

#focal_loss带gamma
def Focal_loss_v1(y_true, y_pred, gamma=0):
   alpha = tf.constant([0.5,0.6,0.7,0.7,0.7,0.75,0.7,0.6], dtype=tf.float32)   #正样本分类的代价
   one_alpha = tf.subtract(tf.ones_like(alpha,dtype=tf.float32),alpha)        #负样本分类的代价
   '''正样本的loss'''
   a1 = tf.pow(tf.subtract(tf.ones_like(y_pred),y_pred),gamma)
   a2 = tf.log(y_pred)
   x1 = tf.multiply(alpha,y_true)
   x2 = tf.multiply(x1,a1)
   x3 = tf.multiply(x2,a2)
   '''负样本的loss'''
   b1 = tf.subtract(tf.ones_like(y_true,dtype=tf.float32),y_true)
   b2 = tf.pow(y_pred,gamma)
   b3 = tf.log(tf.subtract(tf.ones_like(y_pred,dtype=tf.float32),y_pred))
   y1 = tf.multiply(one_alpha,b1)
   y2 = tf.multiply(y1, b2)
   y3 = tf.multiply(y2,b3)

   y = tf.add(x3,y3)
   loss = K.mean(-y, axis=-1)
   return loss

# focal_loss不带gamma
# def Focal_loss_v2(y_true, y_pred):
#    alpha = tf.constant([0.5,0.6,0.7,0.7,0.7,0.7,0.7,0.6], dtype=tf.float32)  # 正样本分类的代价
#    one_alpha = tf.subtract(tf.ones_like(alpha, dtype=tf.float32), alpha)  # 负样本分类的代价
#    '''正样本的loss'''
#    a1 = tf.log(tf.clip_by_value(y_pred,0.000001,1))
#    x1 = tf.multiply(alpha, y_true)
#    x2 = tf.multiply(x1, a1)
#    '''负样本的loss'''
#    b1 = tf.subtract(tf.ones_like(y_true, dtype=tf.float32), y_true)
#    b2 = tf.log(tf.clip_by_value(tf.subtract(tf.ones_like(y_pred, dtype=tf.float32), y_pred),0.000001,1))
#    y1 = tf.multiply(one_alpha, b1)
#    y2 = tf.multiply(y1, b2)
#
#    y = tf.add(x2, y2)
#    loss = K.mean(-y, axis=-1)
#    return loss

# focal_loss带kappa
# def Focal_loss_v3(y_true, y_pred):
#    loss1 = Focal_loss_v1(y_true, y_pred)
#    k_loss = kappa_loss(y_pred, y_true, eps=1e-5)
#    loss = loss1 + k_loss * 0.0001
#    return loss



