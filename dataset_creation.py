


from pedalboard import Pedalboard, Chorus, Reverb, Distortion, Phaser, Delay, load_plugin
from pedalboard.io import AudioFile
import os
import numpy as np
import librosa
import pyloudnorm as pyln

# time only training set creation 9.40 (minutes)

TEST_AUDIO_PATH = r'G:\tesi_4maggio_22\dataset_prove\audio_1.wav'
TEST_OUTPUT_PATH = r'G:\tesi_4maggio_22\dataset_prove\test_1.wav'
TEST_AUDIO_PATH_STEREO = r'G:\tesi_4maggio_22\dataset_prove\audio_stereo.wav'

UNPROCESSED_PATH = r'G:\tesi_4maggio_22\dataset_michele\dataset\test\unprocessed'
PROCESSED_PATH = r'G:\tesi_4maggio_22\dataset_michele\dataset\test'

LOUDNESS_LEVEL = -23.0
SAMPLE_RATE = 44100
DATASET_LENGTH = 300


def load_audio_file(file_path):
    with AudioFile(file_path, 'r') as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate
    return audio, samplerate


def export_audio(processed_audio, processed_path, sr):
    with AudioFile(processed_path, 'w', sr, processed_audio.shape[0]) as f:
        f.write(processed_audio)


def generate_random_array(min, max, length):
    plugin_levels = np.random.uniform(min, max, length)
    return plugin_levels


def print_loudness_value(audio_file, sr, reshape=True):
    if reshape:
        audio_file_reshape = np.reshape(audio_file, np.shape(audio_file)[1])
    else:
        audio_file_reshape = audio_file
    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio_file_reshape)
    print(f"the loudness value is: ", loudness)


def normalize_loudness(audio_file, sr, loudness_level):

    # reshape audio file (1, 507150) -> (507150,)
    # [how pedalboardloads -> how librosa loads]
    audio_file_reshape = np.reshape(audio_file, np.shape(audio_file)[1])
    #print("input shape: ", np.shape(audio_file), " modified shape to apply loudness norm: ", np.shape(audio_file_reshape))

    meter = pyln.Meter(sr)  # create BS.1770 meter
    loudness = meter.integrated_loudness(audio_file_reshape)

    # loudness normalize to -12 dB LUFS
    loudness_normalized_audio = pyln.normalize.loudness(audio_file_reshape, loudness, loudness_level)

    audio_file_normalized = np.reshape(loudness_normalized_audio, (1, np.shape(audio_file)[1]))
    #print("output shape from loudness norm: ", np.shape(loudness_normalized_audio), " reconstructed shape: ", np.shape(audio_file_normalized))

    return audio_file_normalized



def test_tremolo():
    # name="Depth" discrete raw_value=0.5 value=50.0% range=(0.0, 100.0, 0.1)> sensati: 0.3 - 0.6
    # name="Rate" discrete raw_value=0.788257 value=4.000 Hz range=(0.01, 20.0, ~0.02654342105263153)>

    tremolo_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\modulation\MTremolo.vst3")

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)

    print(tremolo_vst3.parameters)
    normalized_depth = 0.5
    real_depth = normalized_depth * 100
    tremolo_vst3.depth = real_depth
    tremolo_vst3.rate = 8.0
    board = Pedalboard([tremolo_vst3])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"depth_{normalized_depth:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


def test_vibrato():
    # name="Depth" discrete raw_value=0.4 value=40.0% range=(0.0, 100.0, 0.1)> sensati: 0.1 - 0.2
    # name="Rate" discrete raw_value=0.788257 value=4.000 Hz range=(0.01, 20.0, ~0.02654342105263153)> rate sensata = 0.4, 0.6, 0.8

    vibrato_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\modulation\MVibrato.vst3")

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)

    print(vibrato_vst3.parameters)
    normalized_depth = 0.1
    real_depth = normalized_depth * 100
    vibrato_vst3.depth = real_depth
    #vibrato_vst3.rate = 8.0
    board = Pedalboard([vibrato_vst3])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"depth_{normalized_depth:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


def test_flanger():
    # name="Depth" discrete raw_value=0.75 value="100% wet, 100% dry" (401 valid string values)> (default 100%, 100%)
    # --> sensati 0.5 - 1
    # name="Range" discrete raw_value=0.282843 value="0.80 ms" (259 valid string values)>
    # name="Rate" discrete raw_value=0.394129 value=0.2000 Hz range=(0.01, 20.0, ~0.02654342105263153)>

    flanger_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\modulation\MFlanger.vst3")

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)

    wet_value_normalized = 1
    wet_value_int = int(wet_value_normalized * 100)
    wet_value_string = f'{wet_value_int}% wet, 100% dry'
    print(wet_value_string)

    flanger_vst3.depth = wet_value_string
    print("flanger depth value: ", flanger_vst3.depth)


    board = Pedalboard([flanger_vst3])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"depth_{wet_value_normalized:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


def test_reverb():
    # if you set dry_level=0.5 and wet_level=0.0 -> input==output (testato in controfase su Reaper)
    # wet_levels range: 0.0 - 1.0 -> useful range: 0.2 - 0.5

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)
    wet_level = np.random.uniform(0.2, 0.5)
    print("wet: ", wet_level)
    board = Pedalboard([Reverb(wet_level=wet_level, dry_level=0.5)])
    effected_audio = board(audio_file, sr)

    export_audio(effected_audio, TEST_OUTPUT_PATH, sr)


def test_chorus():
    # mix values 0.0 - 1.0

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)
    mix_level = np.random.uniform(0.2, 0.5)
    print("mix: ", mix_level)
    board = Pedalboard([Chorus(mix=mix_level)])
    effected_audio = board(audio_file, sr)

    export_audio(effected_audio, TEST_OUTPUT_PATH, sr)


def test_distortion():
    # drive_db values 0.0 - (infinite?) -> sensati 20 - 40

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)
    drive_level = 20
    print("drive: ", drive_level)

    board = Pedalboard([Distortion(drive_db=drive_level)])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"drive_db_{drive_level:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


def test_phaser():
    # mix values: 0 - 1 (dopo 1 rimane uguale) -> sensati: 0.2 - 0.5
    # testato in controfase 1 =2=10. mentre 0 è QUASI uguale all'unprocessed

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)
    mix_level = 0.5
    print("level: ", mix_level)

    board = Pedalboard([Phaser(mix=mix_level)])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"mix_{mix_level:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


def test_delay_feedback():
    # mix values: 0 - 1  -> sensati: 0.2 - 0.3 (anche 0.1 sarebbe ok)

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)
    mix_level = 0.4
    print("level: ", mix_level)

    board = Pedalboard([Delay(mix=mix_level)])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"mix_{mix_level:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


def test_overdrive():
    # range gain value =(0.0, 1.0, 0.01)

    overdrive_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\The Klone.vst3")

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)
    gain_level = 1.0
    print("level: ", gain_level)

    print(overdrive_vst3.parameters)
    overdrive_vst3.gain = gain_level
    board = Pedalboard([overdrive_vst3])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"gain_{gain_level:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


def test_delay_feedback():
    # mix values: 0 - 1  -> sensati: 0.2 - 0.3 (anche 0.1 sarebbe ok)
    # Feedback must be between 0.0 and 1.0 -> quello che utilizzo: 0.4

    audio_file, sr = load_audio_file(TEST_AUDIO_PATH)
    mix_level = 0.3
    print("level: ", mix_level)

    board = Pedalboard([Delay(mix=mix_level, feedback=0.4)])
    effected_audio = board(audio_file, sr)
    output_path = os.path.join(TEST_OUTPUT_PATH[:-11], f"feed_delay_mix_{mix_level:.2f}.wav")

    normalized_audio_file = normalize_loudness(effected_audio, sr, loudness_level=LOUDNESS_LEVEL)
    export_audio(normalized_audio_file, output_path, sr)


# add effects
def add_tremolo_normalized(unprocessed_audio, sr, mix_level, vst3):

    normalized_depth = mix_level
    real_depth = normalized_depth * 100
    vst3.depth = real_depth
    vst3.rate = 8.0          # also 0.4 or 0.6 is good

    board = Pedalboard([vst3])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_vibrato_normalized(unprocessed_audio, sr, mix_level, vst3):

    normalized_depth = mix_level
    real_depth = normalized_depth * 100
    vst3.depth = real_depth

    board = Pedalboard([vst3])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_flanger_normalized(unprocessed_audio, sr, wet_level, vst3):

    wet_value_normalized = wet_level
    wet_value_int = int(wet_value_normalized * 100)
    wet_value_string = f'{wet_value_int}% wet, 100% dry'
    vst3.depth = wet_value_string

    board = Pedalboard([vst3])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_distortion_normalized(unprocessed_audio, sr, drive_level):
    board = Pedalboard([Distortion(drive_db=drive_level)])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_reverb_normalized(unprocessed_audio, sr, wet_level):
    board = Pedalboard([Reverb(wet_level=wet_level)])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_chorus_normalized(unprocessed_audio, sr, mix_level):
    board = Pedalboard([Chorus(mix=mix_level)])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_phaser_normalized(unprocessed_audio, sr, mix_level):
    board = Pedalboard([Phaser(mix=mix_level)])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_delay_normalized(unprocessed_audio, sr, mix_level):
    board = Pedalboard([Delay(mix=mix_level)])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_overdrive_normalized(unprocessed_audio, sr, gain_level, vst3):

    vst3.gain = gain_level
    board = Pedalboard([vst3])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


def add_feedback_delay_normalized(unprocessed_audio, sr, mix_level):
    board = Pedalboard([Delay(mix=mix_level, feedback=0.4)])
    effected_audio = board(unprocessed_audio, sr)
    normalized_audio = normalize_loudness(effected_audio, sr, LOUDNESS_LEVEL)
    return normalized_audio


# export samples
def export_tremolo_sample(file, mix_value, unprocessed_audio, processed_path, sr, vst3):

    tremolo_path = os.path.join(processed_path, "tremolo")
    if not os.path.exists(tremolo_path):
        os.mkdir(tremolo_path)
        print("New folder created: ", tremolo_path)

    processed_audio = add_tremolo_normalized(unprocessed_audio, sr, mix_level=mix_value, vst3=vst3)
    sample_path = os.path.join(tremolo_path, f"{file[:-4]}_tremolo_{mix_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_vibrato_sample(file, mix_value, unprocessed_audio, processed_path, sr, vst3):

    vibrato_path = os.path.join(processed_path, "vibrato")
    if not os.path.exists(vibrato_path):
        os.mkdir(vibrato_path)
        print("New folder created: ", vibrato_path)

    processed_audio = add_vibrato_normalized(unprocessed_audio, sr, mix_level=mix_value, vst3=vst3)
    sample_path = os.path.join(vibrato_path, f"{file[:-4]}_vibrato_{mix_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_flanger_sample(file, wet_value, unprocessed_audio, processed_path, sr, vst3):

    flanger_path = os.path.join(processed_path, "flanger")
    if not os.path.exists(flanger_path):
        os.mkdir(flanger_path)
        print("New folder created: ", flanger_path)

    processed_audio = add_flanger_normalized(unprocessed_audio, sr, wet_level=wet_value, vst3=vst3)
    sample_path = os.path.join(flanger_path, f"{file[:-4]}_flanger_{wet_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_reverb_sample(file, wet_value, unprocessed_audio, processed_path, sr):

    reverb_path = os.path.join(processed_path, "reverb")
    if not os.path.exists(reverb_path):
        os.mkdir(reverb_path)
        print("New folder created: ", reverb_path)

    processed_audio = add_reverb_normalized(unprocessed_audio, sr, wet_level=wet_value)
    sample_path = os.path.join(reverb_path, f"{file[:-4]}_reverb_{wet_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_chorus_sample(file, mix_value, unprocessed_audio, processed_path, sr):

    chorus_path = os.path.join(processed_path, "chorus")
    if not os.path.exists(chorus_path):
        os.mkdir(chorus_path)
        print("New folder created: ", chorus_path)

    processed_audio = add_chorus_normalized(unprocessed_audio, sr, mix_level=mix_value)
    sample_path = os.path.join(chorus_path, f"{file[:-4]}_chorus_{mix_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_distortion_sample(file, drive_value, unprocessed_audio, processed_path, sr):

    distortion_path = os.path.join(processed_path, "distortion")
    if not os.path.exists(distortion_path):
        os.mkdir(distortion_path)
        print("New folder created: ", distortion_path)

    processed_audio = add_distortion_normalized(unprocessed_audio, sr, drive_level=drive_value)
    sample_path = os.path.join(distortion_path, f"{file[:-4]}_distortion_{drive_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_phaser_sample(file, mix_value, unprocessed_audio, processed_path, sr):

    phaser_path = os.path.join(processed_path, "phaser")
    if not os.path.exists(phaser_path):
        os.mkdir(phaser_path)
        print("New folder created: ", phaser_path)

    processed_audio = add_chorus_normalized(unprocessed_audio, sr, mix_level=mix_value)
    sample_path = os.path.join(phaser_path, f"{file[:-4]}_phaser_{mix_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_delay_sample(file, mix_value, unprocessed_audio, processed_path, sr):

    delay_path = os.path.join(processed_path, "delay")
    if not os.path.exists(delay_path):
        os.mkdir(delay_path)
        print("New folder created: ", delay_path)

    processed_audio = add_delay_normalized(unprocessed_audio, sr, mix_level=mix_value)
    sample_path = os.path.join(delay_path, f"{file[:-4]}_delay_{mix_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_overdrive_sample(file, gain_value, unprocessed_audio, processed_path, sr, vst3):

    overdrive_path = os.path.join(processed_path, "overdrive")
    if not os.path.exists(overdrive_path):
        os.mkdir(overdrive_path)
        print("New folder created: ", overdrive_path)

    processed_audio = add_overdrive_normalized(unprocessed_audio, sr, gain_level=gain_value, vst3=vst3)
    sample_path = os.path.join(overdrive_path, f"{file[:-4]}_overdrive_{gain_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def export_feedback_delay_sample(file, mix_value, unprocessed_audio, processed_path, sr):

    f_delay_path = os.path.join(processed_path, "feedback_delay")
    if not os.path.exists(f_delay_path):
        os.mkdir(f_delay_path)
        print("New folder created: ", f_delay_path)

    processed_audio = add_feedback_delay_normalized(unprocessed_audio, sr, mix_level=mix_value)
    sample_path = os.path.join(f_delay_path, f"{file[:-4]}_f_delay_{mix_value:.2f}.wav")
    export_audio(processed_audio, sample_path, sr)


def prepare_dataset(unprocessed_path, processed_path):

    reverb_wet_levels = generate_random_array(min=0.2, max=0.5, length=DATASET_LENGTH)
    chorus_mix_levels = generate_random_array(min=0.2, max=0.5, length=DATASET_LENGTH)
    distortion_drive_levels = generate_random_array(min=20, max=40, length=DATASET_LENGTH)
    phaser_mix_levels = generate_random_array(min=0.2, max=0.5, length=DATASET_LENGTH)
    delay_mix_levels = generate_random_array(min=0.2, max=0.3, length=DATASET_LENGTH)
    overdrive_gain_levels = generate_random_array(min=0.2, max=1.0, length=DATASET_LENGTH)
    f_delay_mix_levels = generate_random_array(min=0.2, max=0.3, length=DATASET_LENGTH)
    flanger_wet_levels = generate_random_array(min=0.5, max=1.0, length=DATASET_LENGTH)
    vibrato_mix_levels = generate_random_array(min=0.1, max=0.2, length=DATASET_LENGTH)
    tremolo_mix_levels = generate_random_array(min=0.3, max=0.6, length=DATASET_LENGTH)

    # load vst3 plugins
    overdrive_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\The Klone.vst3")
    flanger_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\modulation\MFlanger.vst3")
    vibrato_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\modulation\MVibrato.vst3")
    tremolo_vst3 = load_plugin(r"G:\tesi_4maggio_22\a__plugins\modulation\MTremolo.vst3")

    count = 0
    for path, folders, files in os.walk(unprocessed_path):

        # verify correct folder
        print(f'path: {path}\nfolders: {folders}\nfiles: {files}\n')

        # process each song separately
        for file in files:

            file_path = os.path.join(path, file)
            print("processing: ", file_path)

            # load audio
            unprocessed_audio, sr = load_audio_file(file_path)

            # export samples
            export_reverb_sample(file, reverb_wet_levels[count], unprocessed_audio, processed_path, sr)
            export_chorus_sample(file, chorus_mix_levels[count], unprocessed_audio, processed_path, sr)
            export_distortion_sample(file, distortion_drive_levels[count], unprocessed_audio, processed_path, sr)
            export_phaser_sample(file, phaser_mix_levels[count], unprocessed_audio, processed_path, sr)
            export_delay_sample(file, delay_mix_levels[count], unprocessed_audio, processed_path, sr)
            export_overdrive_sample(file, overdrive_gain_levels[count], unprocessed_audio, processed_path, sr, overdrive_vst3)
            export_feedback_delay_sample(file, f_delay_mix_levels[count], unprocessed_audio, processed_path, sr)
            export_flanger_sample(file, flanger_wet_levels[count], unprocessed_audio, processed_path, sr, flanger_vst3)
            export_vibrato_sample(file, vibrato_mix_levels[count], unprocessed_audio, processed_path, sr, vibrato_vst3)
            export_tremolo_sample(file, tremolo_mix_levels[count], unprocessed_audio, processed_path, sr, tremolo_vst3)

            count += 1


if __name__ == '__main__':

    prepare_dataset(UNPROCESSED_PATH, PROCESSED_PATH)

    #test_vibrato()


    # test_delay_feedback()