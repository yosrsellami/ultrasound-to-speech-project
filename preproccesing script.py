import os
import signal_processing

speakers = ['02fe', '03mn', '04me']
data_dir = "/data/UltraSuite_TaL/TaL80/core/"

for speaker in speakers:
    dir_data = os.path.join(data_dir, speaker)
    for file in sorted(os.listdir(dir_data)):
        if file.endswith('.ult') and not file.endswith('_sync.ult'):
            base = os.path.join(dir_data, file[:-4])
            ult_path = base + '.ult'
            wav_path = base + '.wav'
            param_path = base + '.param'
            try:
                print(f"Syncing {base}")
                signal_processing.sync_ult_wav_mel(ult_path, wav_path, param_path)
            except Exception as e:
                print(f"Failed: {e}")
