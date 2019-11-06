# Diarization
# Input: sourcefoldedr, destinationfolder
# Output: diarized file separeted into two channels

# Amir H. Poorjam
import numpy as np
import librosa
import acoustic_feature_extraction as fe
import iHmmNormalSampleGibbsStatesPosterior as iHMM
from scipy import stats
import sys
from os import listdir


MFCCParam = {'NumFilters': 27,'NFFT': 1024,'FminHz': 0,'FMaxHz': 4000,'no': 12,'FLT': 0.020,'FST': 0.020, 'vad_flag':1}
hypers = {'alpha0': 10, 'gamma': 10, 'a0': 1}
FRAME_AVG = 15

sourcefoldedr = sys.argv[1]
destinationfolder = sys.argv[2]

all_files = [f for f in listdir(sourcefoldedr) if f.endswith('.wav')]
total_file = len(all_files)
fidx = 0
for filename in all_files:
    print('%d / %d --> File name: %s' % (fidx+1, total_file, filename))
    signal, fs = librosa.load(sourcefoldedr+filename, sr=None)
    signal = signal - np.mean(signal)
    maxamp = abs(signal).max()
    signal = signal / maxamp
    mfcc,vad_ind,frames = fe.main_mfcc_function(signal, fs, MFCCParam)
    frames_vad = frames[vad_ind, :]

    frames_indx = np.arange(mfcc.shape[0])
    cell_len = int(np.ceil(len(frames_indx) / FRAME_AVG))
    if np.mod(len(frames_indx), FRAME_AVG):
        less_frms = FRAME_AVG - np.remainder(len(frames_indx), FRAME_AVG)
        frames_indx = np.concatenate((frames_indx, frames_indx[-1] + np.arange(1, less_frms + 1)))
        extended_frames = np.concatenate((frames_vad, np.zeros((less_frms, frames.shape[1]))), axis=0)
    else:
        less_frms = 0
        extended_frames = frames_vad

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

    init_stat_number = 2
    # random_init_states = iHMM.array2vector(np.random.random_integers(1,init_stat_number, size=Total_samples)).T
    random_init_states = iHMM.array2vector(np.random.choice(np.arange(1, init_stat_number+1), Total_samples, p=[0.8, 0.2])).T
    posterior = iHMM.main_ihmm_function(MFCCs_matrix, hypers, 30, random_init_states)
    states, state_uncertainty = stats.mode(posterior)
    num_of_states = int(np.max(states))
    post_prob = state_uncertainty / posterior.shape[0]
    states[post_prob < 0.60] = np.nan
    states = np.reshape(np.tile(states.T, FRAME_AVG), (1, FRAME_AVG * states.shape[1]))

    diarized_signal = []
    active_segments = np.zeros((1, num_of_states))
    for sx in range(num_of_states):
        ind_states = np.where(states == sx + 1)[1]
        segments = extended_frames[ind_states, :]
        zzz = np.reshape(segments, (segments.shape[0] * segments.shape[1]))
        active_segments[0, sx] = segments.shape[0]
        diarized_signal.append(zzz)
    ind_vs = np.argsort(active_segments)

    if ind_vs.shape[1] > 1:
        client_signal = diarized_signal[ind_vs[0, -2]]
        interviewer_signal = diarized_signal[ind_vs[0, -1]]
        varname_ch_0 = destinationfolder + filename[0:-4] + '_ch_0.wav'
        varname_ch_1 = destinationfolder + filename[0:-4] + '_ch_1.wav'
        librosa.output.write_wav(varname_ch_0, client_signal, fs)
        librosa.output.write_wav(varname_ch_1, interviewer_signal, fs)
    else:
        client_signal = []
        interviewer_signal = diarized_signal[ind_vs[0, -1]]
        varname_ch_1 = destinationfolder + filename[0:-4] + '_ch_1.wav'
        librosa.output.write_wav(varname_ch_1, interviewer_signal, fs)
        print('|----- No client channel has been detected! ------')


    fidx +=1



