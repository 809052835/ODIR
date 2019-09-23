import os

from keras.models import load_model
from user_defined.losses import myLoss,Focal_loss_v1
from user_defined.metric_c import Roc_Callback
from model.deep_model import get_VGG16
from generator.data_generator_1 import data_generator

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'  #只显示warning 和 error

if __name__ == '__main__':
    BatchSize = 32
    epochs = 25
    INPUT_SHAPE = (224, 224, 3)
    model_savePath = r'H:\Qiulin\ODIR\model\v5\vgg16_c_v1.h5'
    annotation = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training-Chinese_modified_v1.xlsx'
    ODIR_trainDir = r'H:\Qiulin\ODIR\ODIR_dataset\ODIR-5K_training_cvt\v3_train_aug_equ_c'

    '''generator'''
    train_gen = data_generator(annotation,ODIR_trainDir,BatchSize,INPUT_SHAPE,mode='train')

    '''training'''
    model = get_VGG16(INPUT_SHAPE)

    #定义callback需要做的事情
    cb = [
        Roc_Callback(),  # include it before EarlyStopping!
        # EarlyStopping(monitor='final_score', patience=10, verbose=2, mode='max')
        ModelCheckpoint(model_savePath, monitor='final_score', verbose=1, save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='final_score', factor=0.1,patience=8, verbose=1,mode='max',min_delta=0.001)
    ]

    # define loss optimizer
    optimizer = Adam(lr=0.001)  #0.001不要变了
    model.compile(loss=Focal_loss_v1,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    # input data to model and train
    steps = train_gen.num_of_images // train_gen.batch_size

    model.fit_generator(generator=train_gen.get_mini_batch(), steps_per_epoch=steps,
                        epochs=epochs,
                        callbacks=cb,
                        validation_data=None,
                        validation_steps=None,
                        verbose=2)

