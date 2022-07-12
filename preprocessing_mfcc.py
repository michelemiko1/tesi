
import os
import librosa
import json
import numpy as np

#DATASET_PATH = r"G:\tesi_4maggio_22\dataset_michele\dataset\train"
#JSON_PATH = r"G:\tesi_4maggio_22\dataset_michele\log_mel_spec_10effects_1500segments.json"

DATASET_PATH = r"G:\tesi_4maggio_22\dataset_michele\dataset\test"
JSON_PATH = r"G:\tesi_4maggio_22\dataset_michele\log_mel_spec_TEST.json"


SAMPLE_RATE = 22050
SEGMENT_LENGTH = SAMPLE_RATE * 2
NUM_SEGMENTS = 5


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):

    data = {
        "mapping": [],  # ["chorus", "delay", "distortion", "overdrive", "phaser", "reverb", "unprocessed"]
        "mfcc": [],
        "labels": []
    }

    # loop through all effect folders

    for i, (path, folders, files) in enumerate(os.walk(dataset_path)):

        if path is not dataset_path:

            # path = G:\tesi_4maggio_22\dataset_michele\dataset\chorus
            # folders = []
            # files = ['les_bridge_fing01_chorus_0.21.wav', 'les_bridge_fing02_chorus_0.38.wav', ... ]

            # save label
            path_components = os.path.split(path)  # get touple
            effect_name = path_components[-1]
            data["mapping"].append(effect_name)
            print(f"processing {effect_name}", end=" ")

            # how many audio files in each folder
            length = len(files)
            print(f"( samples in {effect_name} folder: {length} )")

            for file in files:

                # load audio file
                file_path = os.path.join(path, file)
                signal, _ = librosa.load(file_path, sr=SAMPLE_RATE)

                # divide audio into segments

                #print(f"total file length: {len(signal)}")
                for index in range(NUM_SEGMENTS):

                    start = index * SEGMENT_LENGTH
                    end = start + SEGMENT_LENGTH
                    # print("segment_index: ", index)
                    # print(f"initial sample: {start}, finish sample: {end}")

                    segment = signal[start:end]

                    # save mfcc of each segment
                    #mfcc = librosa.feature.mfcc(y=segment, sr=SAMPLE_RATE, n_fft=n_fft,
                    #                            n_mfcc=n_mfcc, hop_length=hop_length)
                    #mfcc = mfcc.T
                    #data["mfcc"].append(mfcc.tolist())
                    #data["labels"].append(i - 1)

                    # ATTENZIONE!! STO USANDO LOG MEL SPECTROGRAM E NON MFCC

                    mel_spectrogram = librosa.feature.melspectrogram(y=segment, sr=SAMPLE_RATE, n_fft=n_fft,
                                                                     hop_length=hop_length, n_mels=32)

                    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
                    log_mel_spectrogram = log_mel_spectrogram.T
                    data["mfcc"].append(log_mel_spectrogram.tolist())
                    data["labels"].append(i - 1)




    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":

    save_mfcc(DATASET_PATH, JSON_PATH)
