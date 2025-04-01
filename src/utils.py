import numpy as np
from scipy import signal as sig
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from typing import Tuple, List
import os

class SignalOps:
    @staticmethod
    def convolve(x, y):
        return sig.convolve(x, y, mode='same')

    def correlate(x, y):
        return sig.correlate(x, y, mode='same')
    
    @staticmethod
    def fft_components(signal, fs=400):
        n = len(signal)
        fft_result = fft(signal)
        freqs = fftfreq(n, 1/fs)
        

        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        magnitudes = np.abs(fft_result[pos_mask]) * 2 / n
        phases = np.angle(fft_result[pos_mask])
        
        return freqs, magnitudes, phases
    
    @staticmethod
    def moving_average(signal, window_length=100):
        window = np.ones(window_length) / window_length
        return np.convolve(signal, window, mode='same')
    
    @staticmethod
    def time_shift(signal, lag=400, diff=True):
        shifted_signal = np.zeros_like(signal)
        shifted_signal[lag:] = signal[:-lag]
        return shifted_signal if not diff else signal - shifted_signal
        
    @staticmethod
    def add_noise(signal, noise_factor=0.1):
        noise = np.random.normal(0, noise_factor, size=signal.shape)
        return signal + noise
    

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
    

    

class SignalPlot:
    @staticmethod
    def fft_plot(signal, fs=400, duration=10, path=None):
    
        n = len(signal)
        if duration is None:
            duration = n/fs
        t = np.linspace(0, duration, n)

        fft_result = np.fft.fft(signal)
        freq = np.fft.fftfreq(n, 1/fs)

        phase = np.angle(fft_result)

        plt.figure(figsize=(12, 9))

        plt.subplot(3, 1, 1)
        plt.plot(t, signal)
        plt.title('Original Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        freq_range = n // 2
        plt.plot(freq[:freq_range], np.abs(fft_result)[:freq_range]*2/n)
        plt.title('Frequency Spectrum (Magnitude)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(freq[:n//2], phase[:n//2])
        plt.title('Frequency Spectrum (Phase)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(path) if path else plt.show()
        
    @staticmethod
    def statistics_plot(df:DataFrame, cutedge=500, path=None):

        t = np.linspace(0, cutedge, cutedge)
        plt.figure(figsize=(16, 9))

        plt.subplot(1, 2, 1)
        disease_std = df[:cutedge].std(axis=1)  
        healthy_std = df[cutedge:].std(axis=1)

        sns.lineplot(x=t, y=disease_std, label='Has Disease')
        sns.lineplot(x=t, y=healthy_std, label='Healthy')

        plt.title('Standard Deviation of Signals')
        plt.xlabel('sample point')
        plt.ylabel('Standard Deviation')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(1, 2, 2)
        disease_mean = df[:cutedge].mean(axis=1)
        healthy_mean = df[cutedge:].mean(axis=1)

        disease_se = df[:cutedge].sem(axis=1)
        healthy_se = df[cutedge:].sem(axis=1)


        plt.plot(t, disease_mean, label='Has Disease', color='blue')
        plt.plot(t, healthy_mean, label='Healthy', color='orange')

        plt.fill_between(t, disease_mean-disease_se, disease_mean+disease_se, 
                         alpha=0.3, color='blue', label='_nolegend_')
        plt.fill_between(t, healthy_mean-healthy_se, healthy_mean+healthy_se, 
                         alpha=0.3, color='orange', label='_nolegend_')

        plt.title('Mean of Signals')
        plt.xlabel('sample point')
        plt.ylabel('Mean Magnitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.savefig(path) if path else plt.show()

    @staticmethod
    def binary_class_differential_plot(healthy:Tuple[DataFrame, DataFrame], 
                                       ill:Tuple[DataFrame, DataFrame], 
                                       duration=4000, 
                                       title='Differential Plot', 
                                       label1='origin', 
                                       label2='transformed',
                                       path=None):
        healthy_origin = healthy[0]
        healthy_transformed = healthy[1]
        t = np.linspace(0, duration, duration)
        plt.figure(figsize=(16, 12))

        plt.subplot(2, 2, 1)
        sns.lineplot(x=t, y=healthy_origin, label=label1)
        sns.lineplot(x=t, y=healthy_transformed, label=label2)
        plt.title('healthy ' + title)
        plt.xlabel('sample point')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plt.subplot(2, 2, 2)
        sns.lineplot(x=t, y=healthy_origin, label=label1)
        sns.lineplot(x=t, y=healthy_origin-healthy_transformed, label=label1 + '-' + label2)
        plt.title('healthy ' + title)
        plt.xlabel('sample point')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.subplots_adjust(hspace=0.3)
        ill_origin = ill[0]
        ill_transformed = ill[1]

        plt.subplot(2, 2, 3)
        sns.lineplot(x=t, y=ill_origin, label=label1)
        sns.lineplot(x=t, y=ill_transformed, label=label2)
        plt.title('ill ' + title)
        plt.xlabel('sample point')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.subplot(2, 2, 4)
        sns.lineplot(x=t, y=ill_origin, label=label1)
        sns.lineplot(x=t, y=ill_origin-ill_transformed, label=label1 + '-' + label2)
        plt.title('ill ' + title)
        plt.xlabel('sample point')
        plt.ylabel('Magnitude')
        plt.legend()

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(path) if path else plt.show()
    @staticmethod
    def signal_feature_plot(df:DataFrame, cutedge=500, path=None):
        if not os.path.exists(path):
            os.makedirs(path)
        t = np.linspace(0, cutedge, cutedge)
        ill_df = df[:cutedge]
        healty_df = df[cutedge:]
        feature_list = df.columns.tolist()
        for feature in feature_list:
            plt.figure(figsize=(12, 8))
            sns.lineplot(x=t, y=ill_df[feature], label='Has Disease')
            sns.lineplot(x=t, y=healty_df[feature], label='Healthy')
            plt.title(feature)
            plt.xlabel('sample point')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(path + feature + '.png') if path else plt.show()


    @staticmethod
    def time_shift_diff_plot(healthy:DataFrame, ill:DataFrame, lag=80, path=None):
        
        shifted_signal = np.zeros_like(healthy)
        shifted_signal[lag:] = healthy[:-lag]
        healthy_diff = healthy - shifted_signal
        shifted_signal[lag:] = ill[:-lag]
        ill_diff = ill - shifted_signal
        time_points = np.linspace(0, len(healthy_diff)-lag, len(healthy_diff)-lag)

        plt.figure(figsize=(14, 8))
        sns.lineplot(x=time_points, y=healthy_diff[lag:], linewidth=1, label='Healthy')
        sns.lineplot(x=time_points, y=ill_diff[lag:], linewidth=1, label='Has Disease')
        plt.title(f'Signal Difference Plot: signal[t] - signal[t-{lag}]')
        plt.xlabel('Time (t)')
        plt.ylabel(f'Difference: signal[t] - signal[t-{lag}]')
        plt.grid(True)

        # Add some statistics
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=np.mean(healthy_diff[lag:]), color='g', linestyle='--', alpha=0.7, 
                    label=f'Healthy Mean: {np.mean(healthy_diff[lag:]):.4f}')
        plt.axhline(y=np.mean(ill_diff[lag:]), color='r', linestyle='--', alpha=0.7, 
                    label=f'Ill Mean: {np.mean(ill_diff[lag:]):.4f}')

        plt.legend()
        plt.tight_layout()
        plt.savefig(path) if path else plt.show()


class ModelEval:
    @staticmethod
    def comparison_acc(results:List[np.array]):

        if not results or len(results) < 2:
            return {"error": "Need at least two prediction arrays to compare"}

        first_len = len(results[0])
        if not all(len(arr) == first_len for arr in results):
            return {"error": "All prediction arrays must have the same length"}

        reference = results[0]
        all_agree = np.ones(first_len, dtype=bool)

        for result in results[1:]:
            all_agree &= (result == reference)

        total_samples = first_len
        agreement_count = np.sum(all_agree)
        agreement_percentage = (agreement_count / total_samples) * 100

        return {
            "total_samples": total_samples,
            "agreement_count": int(agreement_count),
            "agreement_percentage": float(agreement_percentage)
        }
