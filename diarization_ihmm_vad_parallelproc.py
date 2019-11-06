# Diarization
# Input: sourcefoldedr, destinationfolder, filename
# Output: diarized file separeted into two channels

# Amir H. Poorjam
####################################
import time
import multiprocessing as mp

tic = time.time()
import numpy as np
import librosa
import acoustic_feature_extraction as fe
import iHmmNormalSampleGibbsStatesPosterior as iHMM
from scipy import stats
import sys
from load_signal import signal_info

def diarization_function(chunck_index):
    # print("| Segment", str(chunck_index), "is being processed. Please wait...")
    MFCCParam = {'NumFilters': 27, 'NFFT': 1024, 'FminHz': 0, 'FMaxHz': 4000, 'no': 12, 'FLT': 0.020, 'FST': 0.020,
                 'vad_flag': 1}
    hypers = {'alpha0': 10, 'gamma': 10, 'a0': 1}
    FRAME_AVG = 15
    sig = signal_info.signal[chunck_index]
    fs = signal_info.fs
    mfcc, vad_ind, frames = fe.main_mfcc_function(sig, fs, MFCCParam)
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
    hypers['b0'] = iHMM.array2vector(0.001 / (np.diagonal(np.cov(MFCCs_matrix, rowvar=False)))).T
    hypers['c0'] = 10 / MFCCs_matrix.shape[0]

    init_stat_number = 2
    random_init_states = iHMM.array2vector(np.random.choice(np.arange(1, init_stat_number + 1), Total_samples, p=[0.7, 0.3])).T
    posterior = iHMM.main_ihmm_function(MFCCs_matrix, hypers, 30, random_init_states)
    states, state_uncertainty = stats.mode(posterior)
    num_of_states = int(np.max(states))
    # print('unique states before: ', np.unique(states))
    post_prob = state_uncertainty / posterior.shape[0]
    states[post_prob < 0.20] = np.nan
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
        client_signal      = diarized_signal[ind_vs[0,-2]]
        interviewer_signal = diarized_signal[ind_vs[0,-1]]
    else:
        client_signal = []
        interviewer_signal = diarized_signal[ind_vs[0,-1]]
    return client_signal, interviewer_signal

####################################
if __name__ == '__main__':
    sourcefoldedr = signal_info.sourcefoldedr
    destinationfolder = signal_info.destinationfolder
    filename = signal_info.filename
    mp.freeze_support()
    with mp.Pool(signal_info.number_of_chuncks) as pp:
        all_diarized_signals = pp.map(diarization_function, range(signal_info.number_of_chuncks))


    channel_0 = all_diarized_signals[0][0] # np.concatenate((all_diarized_signals[0][0], all_diarized_signals[1][0]))#,all_diarized_signals[2][0]))#, all_diarized_signals[3][0]))
    channel_1 = all_diarized_signals[0][1] # np.concatenate((all_diarized_signals[0][1],all_diarized_signals[1][1]))#,all_diarized_signals[2][1]))#, all_diarized_signals[3][1]))
    varname_ch_0 = destinationfolder + filename[0:-4] + '_ch_0.wav'
    varname_ch_1 = destinationfolder + filename[0:-4] + '_ch_1.wav'
    librosa.output.write_wav(varname_ch_0, channel_0, signal_info.fs)
    librosa.output.write_wav(varname_ch_1, channel_1, signal_info.fs)


    toc = time.time()
    print('|-------------------------------------------------')
    print('| %s file is more likely to be the client channel' % (filename[0:-4] + '_ch_0.wav'))
    print('| %s file is more likely to be the interviewer channel' % (filename[0:-4] + '_ch_1.wav'))
    print('|-------------------------------------------------')
    print('| Signal duration: %2.2f min.' % (signal_info.duration/60))
    print('| Processing time: %2.2f sec.' % (toc-tic))
    print('| Relative processing time: %2.2f %% of the signal duration.' % (100*(toc-tic)/(signal_info.duration)))
    print('|-------------------------------------------------')


