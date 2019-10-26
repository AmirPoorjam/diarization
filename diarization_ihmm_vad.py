# Diarization
# Input: sourcefoldedr, destinationfolder, filename
# Usage: python diarization_ihmm_vad.py /path/to/source/folder/ /path/to/destination/folder/ filename.wav
# Output: diarized files separeted into two channels
# Amir H. Poorjam

import time
print('|-------------------------------------------------')
tic = time.time()
import numpy as np
import librosa
import acoustic_feature_extraction as fe
import iHmmNormalSampleGibbsStatesPosterior as iHMM
from scipy import stats
import warnings
import sys

print("| The file is being processed. Please wait...")
# Parameters for feature extraction --> FLT:frame length in time, FST:frame shift in time, no:number of cepstral coefficients, vad_flag:(0 or 1)
MFCCParam = {'NumFilters': 27,'NFFT': 1024,'FminHz': 0,'FMaxHz': 4000,'no': 12,'FLT': 0.020,'FST': 0.020, 'vad_flag':1}
hypers = {'alpha0': 10, 'gamma': 10, 'a0': 1} # iHMM hyper parameters
FRAME_AVG = 15 # average frames to avoid generating irrelevant clusters

sourcefoldedr =  sys.argv[1] # path to source folder. It should end with a forward slash "/"
destinationfolder = sys.argv[2] # path to destination folder. It should end with a forward slash "/"
filename =  sys.argv[3] # file name with extension .wav

signal, fs = librosa.load((sourcefoldedr + filename), sr=None) # load audio signal and the sampling frequency
signal = signal - np.mean(signal) # DC offset removal
maxamp = abs(signal).max()
signal = signal / maxamp # Amplitude normalization
mfcc,vad_ind,frames = fe.main_mfcc_function(signal, fs, MFCCParam) # mfcc:MFCC features, vad_ind:VAD indices, frames:signal in a form of a matrix
frames_vad = frames[vad_ind,:] # excluding non-speech frames

# prepare features for clustering
frames_indx = np.arange(mfcc.shape[0])
cell_len = int(np.ceil(len(frames_indx) / FRAME_AVG))
if np.mod(len(frames_indx), FRAME_AVG):
    less_frms = FRAME_AVG - np.remainder(len(frames_indx), FRAME_AVG)
    frames_indx = np.concatenate((frames_indx, frames_indx[-1] + np.arange(1, less_frms + 1)))
    extended_frames = np.concatenate((frames_vad,np.zeros((less_frms,frames.shape[1]))),axis=0)
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

# setting iHMM hyper parameters
hypers['m0'] = iHMM.array2vector(MFCCs_matrix.mean(axis=0, dtype=np.float64))
hypers['b0'] = iHMM.array2vector(0.001 / (np.diagonal(np.cov(MFCCs_matrix, rowvar=False)))).T
hypers['c0'] = 10 / MFCCs_matrix.shape[0]

init_stat_number = 2 # for conversations we assume that there are two speakers
criterion = 1 # to check if we have more than one channel
crtx = 0

# We assume that the interviewer speaks much more than the client. Thus, we impose different priors for clusters.
# If the first row of this matrix results in only one channel, we move on to the next row by increasing the prior
# probability of client.
init_prior = np.array([[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5]])
while criterion < 2 and crtx < 4:
    random_init_states = iHMM.array2vector(np.random.choice(np.arange(1, init_stat_number+1), Total_samples, p=init_prior[crtx,:])).T
    posterior = iHMM.main_ihmm_function(MFCCs_matrix, hypers, 30, random_init_states)
    states, state_uncertainty = stats.mode(posterior)
    num_of_states = int(np.max(states))
    post_prob = state_uncertainty / posterior.shape[0]
    states[post_prob < 0.60] = np.nan # We pick more reliable frames (those having posterior probabilities greater than 0.6)
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
    criterion = ind_vs.shape[1]
    crtx += 1

# if we end up with two channels, the one with smaller number of active frames is more likely the client channel and
# the one with larger number of active frames is more likely the interviewer channel.
if criterion > 1:
    client_signal      = diarized_signal[ind_vs[0,-2]]
    interviewer_signal = diarized_signal[ind_vs[0,-1]]
    varname_ch_0 = destinationfolder + filename[0:-4] + '_ch_0.wav'
    varname_ch_1 = destinationfolder + filename[0:-4] + '_ch_1.wav'
    librosa.output.write_wav(varname_ch_0, client_signal, fs)
    librosa.output.write_wav(varname_ch_1, interviewer_signal, fs)
    toc = time.time()
    print('|-------------------------------------------------')
    print('| %s file is more likely to be the client channel' % (filename[0:-4] + '_ch_0.wav'))
    print('| %s file is more likely to be the interviewer channel' % (filename[0:-4] + '_ch_1.wav'))

else: # If after several iterations, we still have one channel, it raises a warning and assigns the channel to the interviewer.
    client_signal = []
    interviewer_signal = diarized_signal[ind_vs[0, -1]]
    varname_ch_1 = destinationfolder + filename[0:-4] + '_ch_1.wav'
    librosa.output.write_wav(varname_ch_1, interviewer_signal, fs)
    print('|-------------------------------------------------')
    print('| %s file is more likely to be the interviewer channel' % (filename[0:-4] + '_ch_1.wav'))
    warnings.warn("Warning: No client channel has been detected! ")


print('|-------------------------------------------------')
print('| Signal duration: %2.2f min.' % (signal.size/(fs*60)))
print('| Processing time: %2.2f sec.' % (toc-tic))
print('| Relative processing time: %2.2f %% of the signal duration.' % (100*(toc-tic)/(signal.size/fs)))
print('|-------------------------------------------------')


