import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import signal_processing
import numpy as np
import cv2
import pickle
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers as KL

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

data_dir = "/data/UltraSuite_TaL/TaL80/core/"
speakers = ['01fi', '02fe', '03mn', '04me']
exclude_ids = {'004', '005', '006', '007', '008'}
n_lines = 64
n_pixels_reduced = 128

# === Custom Layer ===
class DCNv2(KL.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(DCNv2, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.stride = (1, 1, 1, 1)
        self.dilation = (1, 1)
        self.deformable_groups = 1

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name='kernel',
            shape=(kh, kw, in_channels, self.filters),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                trainable=True,
            )

        self.offset_kernel = self.add_weight(
            name='offset_kernel',
            shape=(kh, kw, in_channels, 3 * self.deformable_groups * kh * kw),
            initializer='zeros',
            trainable=True,
        )

        self.offset_bias = self.add_weight(
            name='offset_bias',
            shape=(3 * kh * kw * self.deformable_groups,),
            initializer='zeros',
            trainable=True,
        )

        self.ks = kh * kw
        self.ph, self.pw = (kh - 1) // 2, (kw - 1) // 2
        self.phw = tf.constant([self.ph, self.pw], dtype='int32')
        self.patch_yx = tf.stack(tf.meshgrid(tf.range(-self.phw[1], self.phw[1] + 1),
                                             tf.range(-self.phw[0], self.phw[0] + 1))[::-1], axis=-1)
        self.patch_yx = tf.reshape(self.patch_yx, [-1, 2])

    def call(self, x
        # For compatibility, just return a placeholder operation:
        offset = tf.nn.conv2d(x, self.offset_kernel, strides=self.stride, padding='SAME')
        output = tf.nn.conv2d(x, self.kernel, strides=self.stride, padding='SAME')
        if self.use_bias:
            output += self.bias
        return output

    def get_config(self):
        config = super(DCNv2, self).get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "use_bias": self.use_bias,
            "kernel_initializer": keras.initializers.serialize(self.kernel_initializer),
            "bias_initializer": keras.initializers.serialize(self.bias_initializer),
            "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            "bias_regularizer": keras.regularizers.serialize(self.bias_regularizer),
        })
        return config




# === Main Testing Loop ===
for speaker in speakers:
    print(f"\n?? Testing speaker: {speaker}")
    dir_data = os.path.join(data_dir, speaker)

    test_files = [
        os.path.join(dir_data, file[:-4])
        for file in sorted(os.listdir(dir_data))
        if file.endswith('.ult') and not file.endswith('_sync.ult') and file[:3] in exclude_ids
    ]

    if not test_files:
        print(f"?? No test files for speaker {speaker}")
        continue

    try:
        model_base = sorted([
            f.replace('.json', '') for f in os.listdir("models")
            if f.startswith(f"deformable_II_version_{speaker}") and f.endswith(".json")
        ])[-1]
    except IndexError:
        print(f"? No trained model found for {speaker}, skipping.")
        continue

    with open(os.path.join("models", model_base + ".json")) as f:
        model = keras.models.model_from_json(f.read(), custom_objects={'DCNv2': DCNv2, 'L1': regularizers.l1})
    model.load_weights(os.path.join("models", model_base + "_weights_best.h5"))

    with open(os.path.join("models", model_base + "_melspec_scaler.sav"), "rb") as f:
        melspec_scaler = pickle.load(f)

    mse_list = []

    for file in test_files:
        file_id = os.path.basename(file)[:3]
        ult_path = file + '_sync.ult'
        wav_path = file + '_sync.wav'
        param_path = file + '.param'

        if not os.path.exists(ult_path):
            _ = signal_processing.sync_ult_wav_mel(file + '.ult', file + '.wav', param_path)

        ult_data, _, mel_data = signal_processing.get_ult_wav_mel(ult_path, wav_path, param_path)
        L = min(len(ult_data), len(mel_data))

        ult = np.array([cv2.resize(ult_data[i], (n_pixels_reduced, n_lines)) for i in range(L)]) / 255
        ult = ult * 2 - 1
        X_test = ult.reshape((-1, n_lines, n_pixels_reduced, 1))

        y_true = mel_data[:L]
        if y_true.shape[0] == 0:
            continue

        y_scaled = melspec_scaler.transform(y_true)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_scaled, y_pred)
        mse_list.append(mse)

    if mse_list:
        avg_mse = np.mean(mse_list)
        print(f"? The average MSE value for the speaker {speaker} is {avg_mse:.6f}")
    else:
        print(f"?? No valid test results for speaker {speaker}")
