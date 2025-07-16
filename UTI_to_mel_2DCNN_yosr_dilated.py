'''
Training a model over ultrasound and audio recording dataset  - Ibrahim Ibrahimov, 2025

'''

import os
# to disable TensorFlow debug/info logs (to clean up the output)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# data pre-processing tasks
import signal_processing
import numpy as np
import cv2
import datetime
import pickle
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# for getting mel-spec, using WaveGlow_functions by Zainko Csaba
# import WaveGlow_functions

import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

#tf.config.set_visible_devices([], 'GPU')


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
         print(e)

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpu = tf.config.list_physical_devices('GPU')
print(gpu)



data_dir = "/data/UltraSuite_TaL/TaL80/core/"

speakers = ['01fi','02fe','03mn','04me']


framesPerSec = 81.5
n_lines = 64
n_pixels = 842
n_melband = 80

n_pixels_reduced = 128


# passing over all possible speakers to repeat the full pipeline
for speaker in speakers:

    # collecting all the file names into a list for further processing
    dir_data = os.path.join(data_dir,speaker)
    files_all = []
    exclude_ids = {'004', '005', '006', '007', '008'}

for speaker in speakers:
    dir_data = os.path.join(data_dir, speaker)

    files_all = [
        os.path.join(dir_data, file[:-4])
        for file in sorted(os.listdir(dir_data))
        if file.endswith('.ult')
           and not file.endswith('_sync.ult')
           and file[:3] not in exclude_ids
    ]

    # train-dev-test division
    files = dict()
    files['train'] = files_all[:int(0.9*len(files_all))]
    files['valid'] = files_all[int(0.9*len(files_all)):]

    # to have "train" "valid" "test" sets collected into different parts of the dictionaries
    ult_data_all = dict()
    mel_data_all = dict()

    for train_valid in ['train', 'valid']:
        ult_data_all[train_valid] = np.empty((0, n_lines, n_pixels_reduced))
        mel_data_all[train_valid] = np.empty((0, n_melband))

        for file in files[train_valid]:
            param_path = file + '.param'

            # (if not done yet) synchronising ultrasound and wav data based on beep sound & saving synchrosized ult and wav files; and receiving synch-ed mel data
            if not os.path.isfile(param_path[:-6] + '_sync.ult'):
                ult_path = file + '.ult'
                wav_path = file + '.wav'

                ult_params = signal_processing.read_param(param_path)

                sync_ult, sync_wav, sync_mel = signal_processing.sync_ult_wav_mel(ult_path,wav_path,param_path)
            
            ult_path = file + '_sync.ult'
            wav_path = file + '_sync.wav'
            
            ult_data, wav_data, mel_data = signal_processing.get_ult_wav_mel(ult_path, wav_path, param_path)
            
            # trimming ultrasound and mel data according to the minimum length among them
            ultmel_len = min(len(ult_data), len(mel_data))
            
            # ult - 300 frames
            # mel - 348 frammes
            # min(300,348) - 300
            ult_data = ult_data[:ultmel_len] 
            mel_data = mel_data[:ultmel_len]

            print(file, ult_data.shape, mel_data.shape)

            # bicubic interpolation on ultrasound data [from (,64,842) to (,64,128)] and normalization to [0,1] range
            ult_resized = np.zeros((ultmel_len, n_lines, n_pixels_reduced))
            for i in range(ultmel_len):
                ult_resized[i] = cv2.resize(ult_data[i], (n_pixels_reduced, n_lines), interpolation=cv2.INTER_CUBIC) / 255
            
            # to resize ultrasound from [0,1] to [-1,1]: resized_ult = ult *2 -1
            ult_resized = ult_resized * 2 - 1

            # collecting all ultrasound data together
            ult_data_all[train_valid] = np.concatenate([ult_data_all[train_valid], ult_resized], axis = 0)


            # collecting all mel-spec data together
            mel_data_all[train_valid] = np.concatenate([mel_data_all[train_valid], mel_data], axis = 0)

            print(ult_data_all[train_valid].shape)
            print(mel_data_all[train_valid].shape)


    # mel_all = np.fliplr(np.rot90(mel_all, axes=(1, 0)))
    # ult_all_2d = np.resize(ult_all, (ult_all.shape[0]*n_lines, n_pixels_reduced))

    # reshape ult for CNN
    ult_data_all['train'] = np.reshape(ult_data_all['train'], (-1, n_lines, n_pixels_reduced, 1))
    ult_data_all['valid'] = np.reshape(ult_data_all['valid'], (-1, n_lines, n_pixels_reduced, 1))

    melspec_scaler = StandardScaler(with_mean=True, with_std=True)
    mel_data_all['train'] = melspec_scaler.fit_transform(mel_data_all['train'])
    mel_data_all['valid'] = melspec_scaler.fit_transform(mel_data_all['valid']) 

    '''
    2D-CNN training part is adapted from Tamas Gabor Csapo's implementation: https://github.com/BME-SmartLab/UTI-to-STFT
    '''


    ### single training without cross-validation
    # convolutional model, improved version
    model=Sequential()
    # https://github.com/keras-team/keras/issues/11683
    # https://github.com/keras-team/keras/issues/10417
    model.add(InputLayer(input_shape=ult_data_all['train'].shape[1:]))
    # add input_shape to first hidden layer
    model.add(Conv2D(filters=30,kernel_size=(13,13),strides=(1,1), dilation_rate = (2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001), input_shape=ult_data_all['train'].shape[1:]))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=60,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None) ,kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=90,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=120,kernel_size=(13,13),strides=(1,1),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(1000,activation='swish', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None),kernel_regularizer=regularizers.l1(0.000005)))
    model.add(Dropout(0.2))
    model.add(Dense(n_melband,activation='linear'))
    # compile model
    model.compile(SGD(learning_rate=0.01,  momentum=0.1, nesterov=True),loss='mean_squared_error', metrics=['mean_squared_error'])


    print(model.summary())

    # early stopping to avoid over-training
    # csv logger
    current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format( date=datetime.datetime.now() )
    print(current_date)
    model_name = 'models/2D_CNN_dilated_yosr_version_' + speaker + '_' + current_date
    log_dir = os.path.join("logs2", speaker + "_" + current_date)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)


    # callbacks
    earlystopper = EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0001, patience=11, verbose=1, mode='auto')
    lrr = ReduceLROnPlateau(monitor='val_mean_squared_error', patience=2, verbose=1, factor=0.5, min_lr=0.0001)
    logger = CSVLogger(model_name + '.csv', append=True, separator=';')
    checkp = ModelCheckpoint(model_name + '_weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks = [earlystopper, lrr, logger, checkp, tensorboard_callback]
    
    

    # save model
    model_json = model.to_json()
    with open(model_name + '.json', "w") as json_file:
        json_file.write(model_json)

    # serialize scalers to pickle
    pickle.dump(melspec_scaler, open(model_name + '_melspec_scaler.sav', 'wb'))

    # Run training

  
    history = model.fit(ult_data_all['train'], mel_data_all['train'],
                    epochs=10, batch_size=128, shuffle=True, verbose=1,
                    validation_data=(ult_data_all['valid'], mel_data_all['valid']),
                    callbacks=[earlystopper, lrr, logger, checkp, tensorboard_callback])


    # here the training of the DNN is finished


 

