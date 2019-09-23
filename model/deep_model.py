from keras.models import Sequential
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.layers.normalization import BatchNormalization
from keras.applications.xception import Xception
from keras.layers import Dense, Dropout, Activation, Flatten,GlobalAveragePooling2D
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, Add, Input, ZeroPadding2D
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'  #只显示warning 和 error

def get_VGG16(inputshape):
    base_model = VGG16(input_shape=inputshape, weights=None, include_top=False, pooling=None)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(8, activation='sigmoid')(x)
    model = Model(inputs=base_model.input,outputs=predictions)
    print(model.summary())
    return model

def get_VGG19(inputshape):
    base_model = VGG19(input_shape=inputshape, weights=None, include_top=False, pooling=None)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(8, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    print(model.summary())
    return model


