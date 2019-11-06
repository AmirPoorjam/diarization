# Diarization
# Input: sourcefoldedr, destinationfolder
# Output: diarized file separeted into two channels

# Amir H. Poorjam
import numpy as np
import librosa
import mfcc_extraction as fe
import iHmmNormalSampleGibbsStatesPosterior as iHMM
from scipy import stats
import sys
from os import listdir
import parselmouth
import amfm_decompy.basic_tools as basic

MFCCParam = {'NumFilters': 27,'NFFT': 1024,'FminHz': 0,'FMaxHz': 4000,'no': 12,'FLT': 0.015,'FST': 0.015}
hypers = {'alpha0': 10, 'gamma': 10, 'a0': 1}
FRAME_AVG = 15

sourcefoldedr = 'C:/Amir/Codes/diarization/Python_version/challenging_data_1/' # 'C:/Amir/Data/zeldis_interviews/' # sys.argv[1]
destinationfolder = 'C:/Amir/Codes/diarization/Python_version/res_chalng_1/' # sys.argv[2]

all_files = [f for f in listdir(sourcefoldedr) if f.endswith('.wav')]
total_file = len(all_files)
fidx = 0
for filename in all_files:
    print('%d / %d --> File name: %s' % (fidx+1, total_file, filename))
    # s, fs = librosa.load(sourcefoldedr+filename, sr=None)
    signal = basic.SignalObj(sourcefoldedr+filename)
    signal_for_pitch = parselmouth.Sound(sourcefoldedr+filename)
    pitch = signal_for_pitch.to_pitch(time_step=MFCCParam['FST'])
    pitch_values = iHMM.array2vector(pitch.selected_array['frequency'])
    fs = int(signal.fs)
    signal.data = signal.data - np.mean(signal.data)
    maxamp = abs(signal.data).max()
    signal.data = signal.data / maxamp
    mfcc = fe.main_mfcc_function(signal.data, fs, MFCCParam)
    # mfcc = mfcc[:,0:1]
    print(mfcc.shape)
    print(pitch_values.shape)
    if mfcc.shape[0] > pitch_values.shape[0]:
        pitch_values = iHMM.array2vector(np.append(np.zeros((1, mfcc.shape[0] - pitch_values.shape[0])), pitch_values))
    elif mfcc.shape[0] < pitch_values.shape[0]:
        pitch_values = pitch_values[0:mfcc.shape[0] + 1]

    mfcc = np.concatenate((pitch_values, mfcc), axis=1)
    frames_indx = np.arange(mfcc.shape[0])
    cell_len = int(np.ceil(len(frames_indx) / FRAME_AVG))
    if np.mod(len(frames_indx), FRAME_AVG):
        less_frms = FRAME_AVG - np.remainder(len(frames_indx), FRAME_AVG)
        frames_indx = np.concatenate((frames_indx, frames_indx[-1] + np.arange(1, less_frms + 1)))
        extended_signal = np.append(signal.data, np.zeros([int(less_frms * fs * MFCCParam['FST']), 1]))
    else:
        less_frms = 0
        extended_signal = np.copy(signal.data)

    Frm_indx = np.reshape(frames_indx, (FRAME_AVG, cell_len), order='F')
    mfccNaN = np.concatenate((mfcc, np.full((less_frms, mfcc.shape[1]), np.nan)), axis=0)
    MFCCs_matrix = iHMM.array2vector(np.nanmean(mfccNaN[Frm_indx[:, 0], :], axis=0))
    for i in range(1, cell_len):
        new_column = iHMM.array2vector(np.nanmean(mfccNaN[Frm_indx[:, i], :], axis=0))
        MFCCs_matrix = np.concatenate((MFCCs_matrix, new_column), axis=1)

    MFCCs_matrix = MFCCs_matrix.T
    Total_samples = MFCCs_matrix.shape[0]

    hypers['m0'] = iHMM.array2vector(MFCCs_matrix.mean(axis=0, dtype=np.float64))
    hypers['b0'] = iHMM.array2vector(0.0001 / (np.diagonal(np.cov(MFCCs_matrix, rowvar=False)))).T
    hypers['c0'] = 10 / MFCCs_matrix.shape[0]

    init_stat_number = 3
    random_init_states = iHMM.array2vector(np.random.random_integers(1,init_stat_number, size=Total_samples)).T
    # random_init_states = iHMM.array2vector(np.random.choice(np.arange(1, init_stat_number+1), Total_samples, p=[0.7, 0.2, 0.1])).T
    posterior = iHMM.main_ihmm_function(MFCCs_matrix, hypers, 30, random_init_states)
    states, state_uncertainty = stats.mode(posterior)
    post_prob = state_uncertainty / posterior.shape[0]
    states[post_prob < 0.75] = int(np.max(states)) + 1
    num_of_states = int(np.max(states))

    diarized_signal = []
    active_segments = np.zeros((1, num_of_states))
    for sx in range(num_of_states):
        ind_states = np.where(states == sx + 1)[1]
        segments = 0
        for zx in ind_states:
            segments = np.append(segments, np.arange((int(zx * FRAME_AVG * MFCCParam['FST'] * fs)),
                                                 (int((zx + 1) * FRAME_AVG * MFCCParam['FST'] * fs))))

        segments = np.delete(segments, [0, 0])
        zzz = extended_signal[segments]
        if zzz.size==0:
            active_segments[0, sx] = 0
            print('chunck is empty')
        else:
            active_segments[0, sx] = fe.calculate_num_vad_frames(zzz, MFCCParam, fs)
        diarized_signal.append(zzz)

    ind_vs = np.argsort(active_segments)
    varname_ch_0 = destinationfolder + filename[0:-4] + '_ch_0.wav'
    varname_ch_1 = destinationfolder + filename[0:-4] + '_ch_1.wav'
    varname_ch_2 = destinationfolder + filename[0:-4] + '_ch_2.wav'
    librosa.output.write_wav(varname_ch_0, diarized_signal[ind_vs[0,-2]], fs)
    librosa.output.write_wav(varname_ch_1, diarized_signal[ind_vs[0,-1]], fs)
    librosa.output.write_wav(varname_ch_2, diarized_signal[ind_vs[0, -3]], fs)
    fidx +=1



