## MFCC Calculation

import numpy as np
import librosa
import scipy


def Hz2Mel(f_Hz):
    Mel = (1000/np.log10(2)) * np.log10(1 + f_Hz/1000)
    return Mel

def Mel2Hz(f_Mel):
    f_Hz = 1000 * (10**((np.log10(2) * f_Mel)/1000) - 1);
    return f_Hz

def framing(sig,Segment_length,Segment_shift):
    f = librosa.util.frame(sig, Segment_length, Segment_shift).T
    f = scipy.signal.detrend(f, type='constant')
    Frames = (f, f.shape[1], f.shape[0])
    return Frames

def MyFilterBank(NumFilters,fs,FminHz,FMaxHz,NFFT):
    NumFilters = NumFilters + 1
    ml_min = Hz2Mel(FminHz)
    ml_max = Hz2Mel(FMaxHz)
    CenterFreq = np.zeros(NumFilters)
    f = np.zeros(NumFilters+2)
    for m in range(1,NumFilters+1):
        CenterFreq[m-1] = Mel2Hz(ml_min + (m+1)*((ml_max - ml_min)/(NumFilters + 1)))
        f[m] = np.floor((NFFT/fs) * CenterFreq[m-1])
    
    f[0] = np.floor((FminHz/fs)*NFFT)+1
    f[-1] = np.ceil((FMaxHz/fs)*NFFT)-1
    H = np.zeros((NumFilters+1,int(NFFT/2+1)))
    for n in range(1,NumFilters+1):
        fnb = int(f[n-1]) # before
        fnc = int(f[n])   # current
        fna = int(f[n+1]) # after
        fko = fnc - fnb
        flo = fna - fnc
        for k in range(fnb,fnc+1):            
            if fko==0:
                fko = 1
            
            H[n-1,k-1] = (k - fnb)/fko
        for l in range(fnc,fna+1):            
            if flo==0:
                flo = 1
            
            if fna - fnc != 0:
                H[n-1,l-1] = (fna - l)/flo
    
    H = H[0:NumFilters-1,:]
    H = H.T
    return H

def hamming(win_len):
    w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(win_len)/(win_len - 1))
    return w
    
    
def computeFFTCepstrum(windowed_frames, mfcc_bank, MFCCParam):
    n_fft = 2 * mfcc_bank.shape[0]
    SmallNumber = 0.000000001
    ESpec = np.power(abs(np.fft.fft(windowed_frames, n=n_fft)),2).T
    
    ESpec = ESpec[0:int(n_fft/2), :]
    FBSpec = mfcc_bank.T @ ESpec
    LogSpec = np.log(FBSpec + SmallNumber);
    Cep = scipy.fftpack.dct(LogSpec.T,norm='ortho').T
    if Cep.shape[0]>2:
        Cep = Cep[0:MFCCParam['no']+1, :].T
    else:
        Cep = []
        
    return Cep

def delta_delta_mfcc_post_processing(features):
    filter_vector = np.array([[1],[0],[-1]])
    delta = scipy.signal.convolve2d(features, filter_vector, mode='same')
    delta_delta = scipy.signal.convolve2d(delta, filter_vector, mode='same')
    f_d_dd = np.concatenate((features, delta,delta_delta), axis=1)
    return f_d_dd


def calculate_num_vad_frames(signal, MFCCParam, fs):
    Segment_length = round(MFCCParam['FLT'] * fs)
    Segment_shift = round(MFCCParam['FST'] * fs)
    Frames = framing(signal, Segment_length, Segment_shift)[0]
    win = hamming(Segment_length)
    win_repeated = np.tile(win, (Frames.shape[0], 1))
    windowed_frames = np.multiply(Frames, win_repeated)
    ss = 20 * np.log10(np.std(windowed_frames,axis=1) + 0.0000000001)
    max1 = np.max(ss)
    vad_ind = np.all(((ss > max1 - 30),(ss > -55)),axis=0)
    return len(np.where(vad_ind)[0])
##########################################################
    
def main_mfcc_function(orig_signal,fs,MFCCParam):
    Segment_length=round(MFCCParam['FLT']*fs)
    Segment_shift=round(MFCCParam['FST']*fs)
    Frames = framing(orig_signal, Segment_length, Segment_shift)[0]
    win = hamming(Segment_length)
    win_repeated = np.tile(win,(Frames.shape[0],1))
    windowed_frames = np.multiply(Frames,win_repeated)
    mfcc_bank = MyFilterBank(MFCCParam['NumFilters'],fs,MFCCParam['FminHz'],MFCCParam['FMaxHz'],MFCCParam['NFFT'])
    mfcc_coefficients = computeFFTCepstrum(windowed_frames, mfcc_bank, MFCCParam)
    if MFCCParam['vad_flag']==1:
        ss = 20 * np.log10(np.std(windowed_frames, axis=1) + 0.0000000001)
        max1 = np.max(ss)
        vad_ind = np.all(((ss > max1 - 30), (ss > -55)), axis=0)
        mfcc_coefficients = mfcc_coefficients[vad_ind,:]
    else:
        vad_ind=np.ones((Frames.shape[0]))
    return mfcc_coefficients,vad_ind,Frames




    


