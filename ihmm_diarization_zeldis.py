# iHMM for Speaker Diarization (Optimized for Zeldis data set)
# Amir H. Poorjam

import numpy as np
import librosa
from os import listdir
from os import remove
import mfcc_extraction
import iHmmNormalSampleGibbsStatesPosterior as iHMM
from scipy import stats

def array2vector(array):
    array = array.reshape((array.shape[0], 1))
    return array

MFCCParam = {'NumFilters': 27,
             'NFFT': 512,
             'FminHz': 0,
             'FMaxHz': 4000,
             'no': 12,
             'FLT': 0.03, # frame length in time (sec)
             'FST': 0.03}
            
hypers = {'alpha0':10, # How many different transitions
          'gamma':10,  # How many different states
          'a0':1} 
FRAME_AVG = 10
folder = 'C:/Amir/Data/zeldis_interviews/'
all_files = [f for f in listdir(folder) if f.endswith('.wav')]
total_file = len(all_files)

all_signals = {}
fidx = 0
for jx in all_files:
    filename = folder+jx
    s, fs = librosa.load(filename,sr=16000)
    sig = s-np.mean(s)
    maxamp = abs(sig).max()
    orig_signal = sig/maxamp

    mfcc = mfcc_extraction.main_mfcc_function(orig_signal,fs,MFCCParam)    

    frames_indx = np.arange(mfcc.shape[0])
    cell_len = int(np.ceil(len(frames_indx)/FRAME_AVG))
    if np.mod(len(frames_indx),FRAME_AVG):
        less_frms = FRAME_AVG-np.remainder(len(frames_indx),FRAME_AVG)
        frames_indx = np.concatenate((frames_indx, frames_indx[-1]+np.arange(1,less_frms+1)))
        extended_signal = np.append(orig_signal,np.zeros([int(less_frms*fs*MFCCParam['FST']),1]))
    else:
        less_frms = 0
        extended_signal = np.copy(orig_signal)
    
    Frm_indx = np.reshape(frames_indx,(FRAME_AVG,cell_len),order='F')
    
    mfccNaN = np.concatenate((mfcc,np.full((less_frms,mfcc.shape[1]),np.nan)),axis=0)
    MFCCs_matrix = array2vector(np.nanmean(mfccNaN[Frm_indx[:,0],:],axis=0))
    for i in range(1,cell_len):
        new_column = array2vector(np.nanmean(mfccNaN[Frm_indx[:,i],:],axis=0))
        MFCCs_matrix = np.concatenate((MFCCs_matrix,new_column),axis=1)
        
    MFCCs_matrix = MFCCs_matrix.T
    Total_samples = MFCCs_matrix.shape[0]
            
    hypers['m0'] = array2vector(MFCCs_matrix.mean(axis=0,dtype=np.float64))
    hypers['b0'] = array2vector(0.001/(np.diagonal(np.cov(MFCCs_matrix,rowvar=False)))).T
    hypers['c0'] = 10/MFCCs_matrix.shape[0]
    
#    random_init_states = array2vector(np.random.random_integers(1,3, size=Total_samples)).T
    random_init_states = array2vector(np.ceil(np.random.uniform(size=Total_samples) * 3)).T
    posterior = iHMM.main_ihmm_function(MFCCs_matrix, hypers, 50, random_init_states)
    posterior = np.delete(posterior,0,0) 
    states = stats.mode(posterior)[0]
    num_of_states = int(np.max(states))
    
    signals =[]
    mean_segments = np.zeros((1,num_of_states))
    for sx in range(num_of_states):
        ind_states = np.where(states == sx+1)[1]
        segments = 0
        for zx in ind_states:
            segments = np.append(segments,np.arange((int(zx*FRAME_AVG*MFCCParam['FST']*fs)),(int((zx+1)*FRAME_AVG*MFCCParam['FST']*fs))))
    
        segments = np.delete(segments,[0,0])
        zzz = extended_signal[segments]
        mean_segments[0,sx] = mfcc_extraction.calculate_num_vad_frames(zzz, MFCCParam, fs)

        signals.append(zzz)
        var_name_chunk = 'C:/Amir/Codes/diarization/Python_version/results/' + str(jx)[0:-4]+'_ch_'+str(sx)+'.wav'
        librosa.output.write_wav(var_name_chunk, zzz, fs)

    varname = 'sig_' + str(fidx)
    rmx = num_of_states
    while rmx > 2:
        del signals[int(np.argmin(mean_segments))]
        remove('C:/Amir/Codes/diarization/Python_version/results/'+str(jx)[0:-4]+'_ch_'+str(np.argmin(mean_segments))+'.wav')
        mean_segments = array2vector(np.delete(mean_segments, int(np.argmin(mean_segments)))).T
        rmx -= 1
    all_signals[varname]=signals
print('Done')