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
    Frames = librosa.util.frame(sig, Segment_length, Segment_shift).T
    # f = scipy.signal.detrend(f, type='constant')
    # Frames = (f, f.shape[1], f.shape[0])
    return Frames

def Hz2Bark(f_Hz):
    f_bark = 6 * np.arcsinh(f_Hz / 600)
    return f_bark

def Bark2Hz(f_bark):
    f_Hz = 600 * np.sinh(f_bark / 6)
    return f_Hz

def FFT2Bark_matrix(n_fft,sr,n_filts,width,min_freq,max_freq):
    min_bark = Hz2Bark(min_freq)
    nyqbark  = Hz2Bark(max_freq) - min_bark
    if n_filts == 0:
        n_filts = np.ceil(nyqbark) + 1
    wts = np.zeros((n_filts,n_fft))
    step_barks = nyqbark / (n_filts - 1) # bark per filter
    x = np.arange(int(n_fft/2 + 1)) * sr / n_fft
    binbarks = Hz2Bark(x)
    for i in range(n_filts):
        f_bark_mid = min_bark + i * step_barks
        lof = array2vector(binbarks - f_bark_mid - 0.5).T
        hif = array2vector(binbarks - f_bark_mid + 0.5).T
        tmp = np.concatenate((hif,-2.5*lof),axis=0)
        wts[i, np.arange(int(n_fft / 2) + 1)] = 10 ** (np.min((np.zeros((1, tmp.shape[1])), array2vector(np.min(tmp, axis=0) / width).T), axis=0))
    return wts

def postaud(cbf,f_max,broaden):
    # loudness equalization and cube root compression
    # cbf = critical band filters (rows: critical bands, cols: frames)
    n_bands,n_frames = cbf.shape
    # Include frequency points at extremes, discard later
    nfpts = n_bands + 2 * broaden
    bandcfhz = Bark2Hz(np.linspace(0, Hz2Bark(f_max), nfpts))
    # Remove extremal bands (the ones that will be duplicated)
    bandcfhz = bandcfhz[(broaden):(nfpts-broaden)]
    # Hynek's magic equal-loudness-curve formula
    fsq = bandcfhz ** 2
    ftmp = fsq + 1.6e5
    eql = ((fsq/ftmp)**2) * ((fsq + 1.44e6)/(fsq + 9.61e6))
    # weight the critical bands and cube root compress
    z = (np.tile(array2vector(eql),(1,n_frames))*cbf) ** 0.33
    # replicate first and last band (because they are unreliable as calculated)
    if broaden:
        indx = np.append(np.append(0, np.arange(n_bands)), n_bands-1)
        y = z[indx,:]
    else:
        indx = np.append(np.append(1,np.arange(1,n_bands-1)),n_bands-2)
        y = z[indx,:]
    return y

def dolpc(x,model_order):
    n_bands,n_frames = x.shape
    new_array = np.concatenate((x,x[np.arange(17-2,0,-1),:]),axis=0)
    r = np.real(np.fft.ifft(new_array.T).T)  # autocorrelation
    r = r[0:n_bands,:] # First half only
    y = np.ones((n_frames, model_order + 1))
    e = np.zeros((n_frames, 1))
    for i in range(n_frames):
        y_tmp, e_tmp = lvdb(r[:, i], model_order)
        y[i, 1:model_order + 1] = y_tmp
        e[i, 0] = e_tmp
    y = np.divide(y.T, np.add(np.tile(e.T, (model_order + 1, 1)), 1e-8))
    return y

def lvdb(x,model_order):
    # Levinson-Durbin Recursion
    A = np.zeros(model_order)
    e  = x[0]
    T = x[1:]
    for i in range(0, model_order):
        b = T[i]
        if i == 0:
            tmp = -b / e
        else:
            for j in range(0, i):
                b = b + A[j] * T[i-j-1]
            tmp = -b / e
        e = e * (1 - tmp**2)
        A[i] = tmp
        if i == 0:
            continue
        khalf = (i+1)//2
        for j in range(0, khalf):
            ij = i-j-1
            b = A[j]
            A[j] = b + tmp * A[ij]
            if j != ij:
                A[ij] += tmp*b
    return A, e


def lpc2cep(lpc, n_cep):
    n_lpc, n_column = lpc.shape
    cep = np.zeros((n_cep, n_column))
    cep[0, :] = -np.log(lpc[0, :])
    norm_lpc = lpc / (np.tile(lpc[0, :], (n_lpc,1)))
    for n in range(1, n_cep):
        sum_vec = 0
        for m in range(1, n+1):
            sum_vec = sum_vec + (n - m) * norm_lpc[m, :] * cep[(n - m), :]
        cep[n, :] = -(norm_lpc[n, :] + (sum_vec / n))
    return cep


def lifter(cep, lift=0.6, invs=False):
    n_cep = cep.shape[0]
    liftwts = np.append(1, np.arange(1, n_cep) ** lift)
    y = np.diag(liftwts) @ cep
    return y

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

def delta_delta_feature_post_processing(features):
    filter_vector = np.array([[1],[0],[-1]])
    feature_delta = scipy.signal.convolve2d(features, filter_vector, mode='same')
    return feature_delta


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

def calculate_CI(mu,sigma,N,alpha):
    # This function calculates the estimate of the population mean and        %
    # the alpha % confidence interval of the population mean for the sampel   %
    # size more than 30, using the Z-distribution table.                      %
    # Inputs:                                                                 %
    #       mu: sample mean                                                   %
    #       sigma: sample standard deviation                                  %
    #       N: number of samples                                              %
    #       alpha: confidence level (either 95 or 99, default = 95)           %
    # Outputs:                                                                %
    #       ci: alpha % confidence interval of the population mean            %
    #       lb: lower bound                                                   %
    #       ub: upper bound                                                   %
    ###### Amir H. Poorjam ####################################################
    if alpha == 95:
        z_val = 1.96
    elif alpha == 99:
        z_val = 2.58
    else:
        print('alpha is neither 95 nor 99. It is set to 95.')
        z_val = 1.96
    SE = sigma / np.sqrt(N)
    ci = z_val * SE
    lb = mu - ci
    ub = mu + ci
    return ci, lb, ub


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

def main_rasta_plp_function(orig_signal,fs,PLP_Param):
    Segment_length = round(PLP_Param['FLT'] * fs)
    Segment_shift = round(PLP_Param['FST'] * fs)
    # Frames = framing(orig_signal, Segment_length, Segment_shift)[0]
    Frames = framing(orig_signal, Segment_length, Segment_shift)
    win = hamming(Segment_length)
    win_repeated = np.tile(win, (Frames.shape[0], 1))
    windowed_frames = np.multiply(Frames, win_repeated)
    NFFT = PLP_Param['NFFT'] # 512
    pspectrum = np.power(abs(np.fft.fft(windowed_frames, n=NFFT)),2).T
    pspectrum = pspectrum[0:int(NFFT/2) + 1,:]
    nfreqs = pspectrum.shape[0]
    n_fft = (nfreqs-1)*2
    n_filts = int(np.ceil(Hz2Bark(fs/2))+1)
    wts = FFT2Bark_matrix(n_fft, fs, n_filts, 1, 0, (fs/2))
    wts = wts[:, 0: nfreqs]
    aspectrum = wts @ pspectrum
    # final auditory compressions
    postspectrum = postaud(aspectrum, (fs/2) , 0)
    lpcas = dolpc(postspectrum, PLP_Param['lpc_order'])
    rasta_plp_features      = lpc2cep(lpcas, PLP_Param['lpc_order'] + 1)
    rasta_plp_features      = lifter(rasta_plp_features, 0.6)
    rasta_plp_features_D    = delta_delta_feature_post_processing(rasta_plp_features)
    rasta_plp_features_DD   = delta_delta_feature_post_processing(rasta_plp_features_D)
    rasta_plp_features_D_DD = np.concatenate((rasta_plp_features, rasta_plp_features_D, rasta_plp_features_DD), axis=0)
    if PLP_Param['vad_flag']==1:
        ss = 20 * np.log10(np.std(windowed_frames, axis=1,ddof=1) + 0.0000000001)
        max1 = np.max(ss)
        vad_ind = np.all(((ss > max1 - 30), (ss > -55)), axis=0)
        rasta_plp_features_D_DD = rasta_plp_features_D_DD[:,vad_ind]
    else:
        vad_ind=np.ones((Frames.shape[0]))
    return rasta_plp_features_D_DD.T, vad_ind, Frames





    


