## Acoustic Features Calculation

import numpy as np
import librosa
import scipy
import parselmouth
from parselmouth.praat import call
import statistics

def array2vector(array):
    array = array.reshape((array.shape[0], 1))
    return array

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
    ss = 20 * np.log10(np.std(windowed_frames,axis=1,ddof=1) + 0.0000000001)
    max1 = np.max(ss)
    vad_ind = np.all(((ss > max1 - 30),(ss > -55)),axis=0)
    return len(np.where(vad_ind)[0])

def measurePitch(signal, f0min, f0max, unit, time_step):
    sound = parselmouth.Sound(signal)  # read the sound
    duration = call(sound, "Get total duration")  # duration
    pitch = call(sound, "To Pitch", time_step, f0min, f0max)  # create a praat pitch object
    pitch_values = array2vector(pitch.selected_array['frequency']).T
    meanF0 = call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)  # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", time_step, f0min, 0 , 1.0)
    harmonicity_values = harmonicity.values
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    min_dim = np.minimum(pitch_values.shape[1],harmonicity_values.shape[1])
    frame_level_features = np.concatenate((pitch_values[:,0:min_dim],harmonicity_values[:,0:min_dim]),axis=0)
    recording_level_features = np.array([duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer])
    return frame_level_features, recording_level_features

def measureFormants(signal, f0min, f0max):
    sound = parselmouth.Sound(signal)  # read the sound
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []

    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)

    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']

    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list)
    f2_mean = statistics.mean(f2_list)
    f3_mean = statistics.mean(f3_list)
    f4_mean = statistics.mean(f4_list)

    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list)
    f2_median = statistics.median(f2_list)
    f3_median = statistics.median(f3_list)
    f4_median = statistics.median(f4_list)
    all_formants = np.array([f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median])
    return all_formants
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
        ss = 20 * np.log10(np.std(windowed_frames, axis=1,ddof=1) + 0.0000000001)
        max1 = np.max(ss)
        vad_ind = np.all(((ss > max1 - 30), (ss > -55)), axis=0)
        mfcc_coefficients = mfcc_coefficients[vad_ind,:]
    else:
        vad_ind=np.ones((Frames.shape[0]))
    if 'CMVN' in MFCCParam:
        if MFCCParam['CMVN'] == 1:
            mfcc_coefficients = (mfcc_coefficients - np.tile(np.mean(mfcc_coefficients,axis=0), (mfcc_coefficients.shape[0],1))) / np.tile(np.std(mfcc_coefficients,axis=0,ddof=1),(mfcc_coefficients.shape[0], 1))
    return mfcc_coefficients,vad_ind,Frames




    


