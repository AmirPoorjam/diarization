import librosa
import numpy as np

class signal_info():
    number_of_chuncks = 1
    sourcefoldedr = 'C:/Amir/Codes/diarization/Python_version/challenging_data_1/'  # sys.argv[1]
    destinationfolder = 'C:/Amir/Codes/diarization/Python_version/res_chalng_1/'  # sys.argv[2]
    filename = '100487.wav'  # sys.argv[3]
    signal, fs = librosa.load((sourcefoldedr + filename), sr=None)
    signal = signal - np.mean(signal)
    maxamp = abs(signal).max()
    signal = signal / maxamp
    num_zero_padding = np.mod(signal.shape[0], number_of_chuncks)
    signal = np.append(signal, np.zeros((num_zero_padding)))
    duration = signal.shape[0]/fs
    signal = np.reshape(signal, (number_of_chuncks, int(signal.shape[0] / number_of_chuncks)), order='C')
    signal = list(signal)
