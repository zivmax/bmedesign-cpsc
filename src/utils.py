import numpy as np
from scipy import signal as sig
from scipy.stats import entropy
from scipy.fft import fft, fftfreq

class SignalOps:
    @staticmethod
    def convolve(x, y):
        return sig.convolve(x, y, mode='same')

    def correlate(x, y):
        return sig.correlate(x, y, mode='same')
    
    @staticmethod
    def get_fft_components(signal, fs=400):
        n = len(signal)
        fft_result = fft(signal)
        freqs = fftfreq(n, 1/fs)
        

        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitudes = np.abs(fft_result[pos_mask]) * 2 / n
        phases = np.angle(fft_result[pos_mask])
        
        return freqs, magnitudes, phases
        
    @staticmethod
    def bandpass_filter(signal, fs=400, lowcut=1, highcut=50, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = sig.butter(order, [low, high], btype='band')
        return sig.filtfilt(b, a, signal)
    
    

class SignalFeatures:
    @staticmethod
    def ps_density(signal, fs=400, nperseg=256):
        f, Psd = sig.welch(signal, fs=fs, nperseg=nperseg)
        return f, Psd
    
    @staticmethod
    def entropy(signal, factor=1e-10):
        normalized = np.abs(signal) + factor
        normalized = normalized / np.sum(normalized)
        return entropy(normalized)
    
    @staticmethod
    def peaks(signal, threshold=0.5):
        peaks, _ = sig.find_peaks(signal, height=threshold)
        return peaks
    
    @staticmethod
    def moving_average(signal, window_length=5):
        window = np.ones(window_length) / window_length
        return np.convolve(signal, window, mode='same')