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
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import model_from_json




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
exclude_ids = {'004', '005', '006', '007', '008'}
data_dir = "/data/UltraSuite_TaL/TaL80/core/"
mel_dir = "/home/yosr/data/Hifi_Gan/mel_spectrogram/"
model_dir = "/home/yosr/data/models/"
speakers = ['01fi','02fe','03mn','04me']

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
           shape=tuple(self.kernel_size) + (input_shape[-1], 3 * self.deformable_groups * kh * kw),
           initializer='zeros',
           trainable=True,
           dtype='float32'
)

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

os.makedirs(mel_dir, exist_ok=True)


for speaker in speakers:
    print(f"\nTesting speaker: {speaker}")

    dir_data = os.path.join(data_dir, speaker)
    test_files = [
        os.path.join(dir_data, file[:-4])
        for file in sorted(os.listdir(dir_data))
        if file.endswith('.ult') and not file.endswith('_sync.ult') and file[:3] in exclude_ids
    ]

    if not test_files:
        print(f"No test files for speaker {speaker}, skipping.")
        continue

    # === Load trained model ===
    try:
       model_dir = "/home/yosr/data/models/"
       model_base = sorted([
            f.replace('.json', '') for f in os.listdir("models")
            if f.startswith(f"deformable_II_version_{speaker}") and f.endswith(".json")
        ])[-1]

    except IndexError:
        print(f"No trained model found for {speaker}, skipping.")
        continue

    print(f"?? Loading model: {model_base}")
    with open(os.path.join("models", model_base + ".json")) as f:
       model = model_from_json(f.read(), custom_objects={
        'L1': regularizers.l1,
        'DCNv2': DCNv2
    })
    model.load_weights(os.path.join("models", model_base + "_weights_best.h5"))


    with open(os.path.join(model_dir, model_base + "_melspec_scaler.sav"), "rb") as f:

        melspec_scaler = pickle.load(f)

    # === Inference on each file ===
    for file in test_files:
        file_id = os.path.basename(file)[:3]

        ult_path = file + '_sync.ult'
        wav_path = file + '_sync.wav'
        param_path = file + '.param'

        if not os.path.exists(ult_path):
            _ = signal_processing.sync_ult_wav_mel(file + '.ult', file + '.wav', param_path)

        ult_data, _, mel_data = signal_processing.get_ult_wav_mel(ult_path, wav_path, param_path)
        L = min(len(ult_data), len(mel_data))

        if L == 0:
            print(f"Empty data for {speaker} file {file_id}, skipping.")
            continue

        ult = np.array([
            cv2.resize(ult_data[i], (n_pixels_reduced, n_lines), interpolation=cv2.INTER_CUBIC)
            for i in range(L)
        ]) / 255.0
        ult = ult * 2 - 1
        X_test = ult.reshape((-1, n_lines, n_pixels_reduced, 1))
        y_true = mel_data[:L]

        try:
            y_scaled = melspec_scaler.transform(y_true)
        except Exception as e:
            print(f"Scaler transform failed for {speaker} file {file_id}: {e}")
            continue

        y_pred = model.predict(X_test)

        # === Save output ===
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        npy_path = os.path.join(mel_dir, f"{speaker}_{file_id}_{timestamp}.npy")
        png_path = os.path.join(mel_dir, f"{speaker}_{file_id}_{timestamp}.png")

        np.save(npy_path, y_pred)

        plt.figure(figsize=(10, 4))
        plt.imshow(y_pred.T, aspect='auto', origin='lower', interpolation='none')
        plt.title(f"Mel-Spectrogram - {speaker} {file_id}")
        plt.xlabel("Time Frame")
        plt.ylabel("Mel Bands")
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

        print(f"Saved: {npy_path}")
        print(f"Saved: {png_path}")
