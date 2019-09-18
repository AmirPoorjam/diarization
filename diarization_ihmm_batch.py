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

MFCCParam = {'NumFilters': 27,'NFFT': 512,'FminHz': 0,'FMaxHz': 4000,'no': 12,'FLT': 0.030,'FST': 0.030}
hypers = {'alpha0': 10, 'gamma': 10, 'a0': 1}
FRAME_AVG = 10

sourcefoldedr = sys.argv[1]
destinationfolder = sys.argv[2]

all_files = [f for f in listdir(sourcefoldedr) if f.endswith('.wav')]
total_file = len(all_files)
fidx = 0
for jx in all_files:
    filename = sourcefoldedr+jx
    s, fs = librosa.load(filename, sr=16000)
    sig = s - np.mean(s)
    maxamp = abs(sig).max()
    orig_signal = sig / maxamp
    mfcc = fe.main_mfcc_function(orig_signal, fs, MFCCParam)
    frames_indx = np.arange(mfcc.shape[0])
    cell_len = int(np.ceil(len(frames_indx) / FRAME_AVG))
    if np.mod(len(frames_indx), FRAME_AVG):
        less_frms = FRAME_AVG - np.remainder(len(frames_indx), FRAME_AVG)
        frames_indx = np.concatenate((frames_indx, frames_indx[-1] + np.arange(1, less_frms + 1)))
        extended_signal = np.append(orig_signal, np.zeros([int(less_frms * fs * MFCCParam['FST']), 1]))
    else:
        less_frms = 0
        extended_signal = np.copy(orig_signal)

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

    random_init_states = iHMM.array2vector(np.random.random_integers(1,3, size=Total_samples)).T
    posterior = iHMM.main_ihmm_function(MFCCs_matrix, hypers, 20, random_init_states)
    states = stats.mode(posterior)[0]
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
        active_segments[0, sx] = fe.calculate_num_vad_frames(zzz, MFCCParam, fs)
        diarized_signal.append(zzz)

    ind_vs = np.argsort(active_segments)
    varname_ch_0 = destinationfolder + filename[0:-4] + '_ch_0.wav'
    varname_ch_1 = destinationfolder + filename[0:-4] + '_ch_1.wav'
    librosa.output.write_wav(varname_ch_0, diarized_signal[ind_vs[0,-2]], fs)
    librosa.output.write_wav(varname_ch_1, diarized_signal[ind_vs[0,-1]], fs)
    fidx +=1
    print('%d / %d processed --> File name: %s ' % (fidx,total_file,jx))



