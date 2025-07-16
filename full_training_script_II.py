'''
Training a model over ultrasound and audio recording dataset - Yosr Selllami, 2025
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import signal_processing
import numpy as np
import cv2
import datetime
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plti

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
from tensorflow.keras.layers import InputLayer, Dropout, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import InputLayer, Conv2D, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint, TensorBoard
from tensorflow.keras import layers as KL




print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpu = tf.config.list_physical_devices('GPU')
print(gpu)


data_dir = "/data/UltraSuite_TaL/TaL80/core/"
speakers = ['03mn']

framesPerSec = 81.5
n_lines = 64
n_pixels = 842
n_melband = 80
n_pixels_reduced = 128

'''implementing the keras customer layer DCNv2'''

class DCNv2(KL.Layer):
    def __init__(self, filters,
                 kernel_size,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = (1, 1, 1, 1)
        self.dilation = (1, 1)
        self.deformable_groups = 1
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        super(DCNv2, self).__init__(**kwargs)
    def get_config(self):
        config = super(DCNv2, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
            "kernel_regularizer": self.kernel_regularizer,
            "bias_regularizer": self.bias_regularizer,
        })
        return config

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name='kernel',
            shape=(kh, kw, in_channels, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
            dtype='float32',
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
                dtype='float32',
            )

        self.offset_kernel = self.add_weight(
            name='offset_kernel',
            shape=self.kernel_size + (input_shape[-1], 3 * self.deformable_groups * kh * kw),
            initializer='zeros',
            trainable=True,
            dtype='float32')

        self.offset_bias = self.add_weight(
            name='offset_bias',
            shape=(3 * kh * kw * self.deformable_groups,),
            initializer='zeros',
            trainable=True,
            dtype='float32',
        )
        self.ks = kh * kw
        self.ph, self.pw = (kh - 1) // 2, (kw - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype='int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1),
                                             tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis=-1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])
        super(DCNv2, self).build(input_shape)

    def call(self, x):
        offset = tf.nn.conv2d(x, self.offset_kernel, strides=self.stride, padding='SAME')
        offset += self.offset_bias

        bs, ih, iw, ic = tf.unstack(tf.shape(x))
        ih_f = tf.cast(ih + 1, tf.float32)
        iw_f = tf.cast(iw + 1, tf.float32)

        oyox, mask = offset[..., :2 * self.ks], offset[..., 2 * self.ks:]
        mask = tf.nn.sigmoid(mask)
        oyox = tf.cast(oyox, tf.float32)
        mask = tf.cast(mask, tf.float32)


        grid_yx = tf.stack(tf.meshgrid(tf.range(iw), tf.range(ih))[::-1], axis=-1)
        grid_yx = tf.reshape(grid_yx, [1, ih, iw, 1, 2]) + self.phw + self.patch_yx
        grid_yx = tf.cast(grid_yx, 'float32') + tf.reshape(oyox, [bs, ih, iw, -1, 2])

        grid_iy0ix0 = tf.floor(grid_yx)
        grid_iy1ix1 = tf.clip_by_value(grid_iy0ix0 + 1.0, 0.0, tf.stack([ih_f, iw_f]))
        grid_iy0ix0 = tf.clip_by_value(grid_iy0ix0, 0.0, tf.stack([ih_f, iw_f]))

        grid_iy1, grid_ix1 = tf.split(grid_iy1ix1, 2, axis=-1)
        grid_iy0, grid_ix0 = tf.split(grid_iy0ix0, 2, axis=-1)
        grid_yx = tf.clip_by_value(grid_yx, 0.0, tf.stack([ih_f, iw_f]))

        batch_index = tf.tile(tf.reshape(tf.range(bs), [bs, 1, 1, 1, 1, 1]), [1, ih, iw, self.ks, 4, 1])
        grid = tf.reshape(tf.concat([
            grid_iy1, grid_ix1,
            grid_iy1, grid_ix0,
            grid_iy0, grid_ix1,
            grid_iy0, grid_ix0
        ], axis=-1), [bs, ih, iw, self.ks, 4, 2])
        grid = tf.concat([batch_index, tf.cast(grid, 'int32')], axis=-1)

        delta = tf.reshape(tf.concat([grid_yx - grid_iy0ix0, grid_iy1ix1 - grid_yx], axis=-1), [bs, ih, iw, self.ks, 2, 2])
        w = tf.expand_dims(delta[..., 0], -1) * tf.expand_dims(delta[..., 1], -2)

        x = tf.pad(x, [[0, 0], [self.ph, self.ph], [self.pw, self.pw], [0, 0]])
        map_sample = tf.gather_nd(x, grid)
        map_bilinear = tf.reduce_sum(tf.reshape(w, [bs, ih, iw, self.ks, 4, 1]) * map_sample, axis=-2) * tf.expand_dims(mask, -1)

        map_all = tf.reshape(map_bilinear, [bs, ih, iw, -1])
        output = tf.nn.conv2d(map_all, tf.reshape(self.kernel, [1, 1, -1, self.filters]), strides=self.stride, padding='SAME')

        if self.use_bias:
            output += self.bias

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)


'''loop for passing through all the speakers'''
# Files to exclude 
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


    files = {
        'train': files_all[:int(0.9 * len(files_all))],
        'valid': files_all[int(0.9 * len(files_all)):]
    }

    ult_data_all = {key: np.empty((0, n_lines, n_pixels_reduced)) for key in ['train', 'valid']}
    mel_data_all = {key: np.empty((0, n_melband)) for key in ['train', 'valid']}

    for train_valid in ['train', 'valid']:
        for file in files[train_valid]:
            param_path = file + '.param'

            if not os.path.isfile(file + '_sync.ult'):
                ult_path = file + '.ult'
                wav_path = file + '.wav'
                _ = signal_processing.sync_ult_wav_mel(ult_path, wav_path, param_path)

            ult_path = file + '_sync.ult'
            wav_path = file + '_sync.wav'

            ult_data, wav_data, mel_data = signal_processing.get_ult_wav_mel(ult_path, wav_path, param_path)

            ultmel_len = min(len(ult_data), len(mel_data))
            ult_data = ult_data[:ultmel_len]
            mel_data = mel_data[:ultmel_len]

            ult_resized = np.zeros((ultmel_len, n_lines, n_pixels_reduced))
            for i in range(ultmel_len):
                ult_resized[i] = cv2.resize(ult_data[i], (n_pixels_reduced, n_lines), interpolation=cv2.INTER_CUBIC) / 255
            ult_resized = ult_resized * 2 - 1

            ult_data_all[train_valid] = np.concatenate([ult_data_all[train_valid], ult_resized], axis=0)
            mel_data_all[train_valid] = np.concatenate([mel_data_all[train_valid], mel_data], axis=0)

    ult_data_all['train'] = np.reshape(ult_data_all['train'], (-1, n_lines, n_pixels_reduced, 1))
    ult_data_all['valid'] = np.reshape(ult_data_all['valid'], (-1, n_lines, n_pixels_reduced, 1))

    melspec_scaler = StandardScaler()
    mel_data_all['train'] = melspec_scaler.fit_transform(mel_data_all['train'])
    mel_data_all['valid'] = melspec_scaler.transform(mel_data_all['valid'])



  # Building the model using the custom DCNv2 (Deformable Convolution v2) layers

    model = Sequential()
    model.add(InputLayer(input_shape=ult_data_all['train'].shape[1:]))

    model.add(DCNv2(filters=30, kernel_size=(3, 3),
                kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2))

    model.add(Conv2D(filters=60,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None) ,kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(filters=90,kernel_size=(13,13),strides=(2,2),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=120,kernel_size=(13,13),strides=(1,1),activation="swish", padding="same",kernel_initializer=keras.initializers.he_uniform(seed=None), kernel_regularizer=regularizers.l1(0.00001)))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1000,activation='swish', kernel_initializer=keras.initializers.he_uniform(seed=None), bias_initializer=keras.initializers.he_uniform(seed=None),kernel_regularizer=regularizers.l1(0.000005)))
    model.add(Dropout(0.2))
    model.add(Dense(n_melband,activation='linear'))

    model.compile(optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True, clipvalue=1.0),
              loss='mean_squared_error', metrics=['mean_squared_error'])

    model.summary()


    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = 'models/deformable_II_version_{speaker}_{current_date}'

    log_dir = os.path.join("logsdeformable", f"{speaker}_{current_date}")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    earlystopper = EarlyStopping(monitor='val_mean_squared_error', patience=11, verbose=1)
    lrr = ReduceLROnPlateau(monitor='val_mean_squared_error', patience=2, verbose=1, factor=0.5, min_lr=0.0001)
    logger = CSVLogger(model_name + '.csv', append=True, separator=';')
    checkp = ModelCheckpoint(model_name + '_weights_best.h5', monitor='val_loss', verbose=1, save_best_only=True)

    with open(model_name + '.json', "w") as json_file:
        json_file.write(model.to_json())

    pickle.dump(melspec_scaler, open(model_name + '_melspec_scaler.sav', 'wb'))

    history = model.fit(ult_data_all['train'], mel_data_all['train'],
                        epochs=10, batch_size=8, shuffle=True, verbose=1,
                        validation_data=(ult_data_all['valid'], mel_data_all['valid']),
                        callbacks=[earlystopper, lrr, logger, checkp, tensorboard_callback])

    with open(model_name + '_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

  
