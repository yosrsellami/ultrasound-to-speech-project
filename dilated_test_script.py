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
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense, InputLayer
from tensorflow.keras import regularizers

# === CONFIGURATION ===
data_dir = "/data/UltraSuite_TaL/TaL80/core/"
mel_dir = "/home/yosr/data/Hifi_Gan/mel_spectogram_dilated/"
os.makedirs(mel_dir, exist_ok=True)

speakers = ['01fi', '02fe', '03mn', '04me']
exclude_ids = {'004', '005', '006', '007', '008'}

n_lines = 64
n_pixels_reduced = 128
n_melband = 80

# === PROCESS EACH SPEAKER ===
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
       f.replace('.json', '') for f in os.listdir(model_dir)
       if f.startswith(f"2D_CNN_dilated_yosr_version_{speaker}") and f.endswith(".json")
       ])[-1]

    except IndexError:
        print(f"No trained model found for {speaker}, skipping.")
        continue

    print(f"Loading model: {model_base}")
    with open(os.path.join(model_dir, model_base + ".json")) as f:
      model = model_from_json(f.read(), custom_objects={'L1': regularizers.l1})
      model.load_weights(os.path.join(model_dir, model_base + "_weights_best.h5"))


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
        # Save ground truth .wav for MCD reference
        ref_wav_dir = "/home/yosr/data/Hifi_Gan/reference_wav/"
        os.makedirs(ref_wav_dir, exist_ok=True)

        original_synced_wav = wav_path


        import shutil
        ref_filename = f"{speaker}_{file_id}.wav"
        shutil.copyfile(original_synced_wav, os.path.join(ref_wav_dir, ref_filename))





