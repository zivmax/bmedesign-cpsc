import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


class SignalOps:
    @staticmethod
    def fft_components(signal, fs=400):
        n = len(signal)
        fft_result = fft(signal)
        freqs = fftfreq(n, 1 / fs)

        # Keep only the positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        # Normalize magnitudes
        magnitudes = np.abs(fft_result[pos_mask]) * 2 / n
        phases = np.angle(
            fft_result[pos_mask]
        )  # Phases are not used in this script but kept for consistency

        return freqs, magnitudes, phases


def plot_average_fft_spectrum(csv_file_path):
    # Load the training data
    train_df = pd.read_csv(csv_file_path, header=None)  # Assuming no header in the CSV

    # Add a target column: 1 for diseased (0-499), 0 for healthy (500-999)
    # Unknown data is not explicitly handled here as we only focus on diseased and healthy
    train_df["target"] = -1  # Default to unknown
    train_df.iloc[0:500, train_df.columns.get_loc("target")] = 1  # Diseased
    train_df.iloc[500:1000, train_df.columns.get_loc("target")] = 0  # Healthy

    # Separate labeled data
    # We assume the signal data is in all columns except the last 'target' column
    signal_columns = train_df.columns[:-1]
    labeled_train_df = train_df[train_df["target"] != -1][signal_columns]
    target = train_df[train_df["target"] != -1]["target"]

    # Get diseased and healthy signals
    diseased_indices = target[target == 1].index
    healthy_indices = target[target == 0].index

    # Ensure indices are valid for .loc by using the original DataFrame's index
    # if labeled_train_df was created by filtering train_df
    diseased_signals_df = labeled_train_df.loc[diseased_indices]
    healthy_signals_df = labeled_train_df.loc[healthy_indices]

    # Calculate FFT magnitudes for all diseased signals
    all_diseased_fft_mags = []
    frequencies = None

    if not diseased_signals_df.empty:
        for index, signal_row in diseased_signals_df.iterrows():
            freq, mag, _ = SignalOps.fft_components(signal_row.values)
            all_diseased_fft_mags.append(mag)
            if frequencies is None:
                frequencies = freq

    if len(all_diseased_fft_mags) > 0:
        avg_diseased_fft_mag = np.mean(np.array(all_diseased_fft_mags), axis=0)
    else:
        avg_diseased_fft_mag = np.array([])
        print("Warning: No diseased signals found.")

    # Calculate FFT magnitudes for all healthy signals
    all_healthy_fft_mags = []
    if not healthy_signals_df.empty:
        for index, signal_row in healthy_signals_df.iterrows():
            # Frequencies should be the same if all signals have the same length and fs
            _, mag, _ = SignalOps.fft_components(signal_row.values)
            all_healthy_fft_mags.append(mag)
            if frequencies is None and diseased_signals_df.empty:
                frequencies = SignalOps.fft_components(signal_row.values)[0]

    if len(all_healthy_fft_mags) > 0:
        avg_healthy_fft_mag = np.mean(np.array(all_healthy_fft_mags), axis=0)
    else:
        avg_healthy_fft_mag = np.array([])
        print("Warning: No healthy signals found.")

    # Plotting
    plt.figure(figsize=(14, 7))
    plot_successful = False

    if frequencies is not None and avg_diseased_fft_mag.size > 0:
        n_plot = len(frequencies)  # Positive frequencies already selected
        plt.plot(
            frequencies[:n_plot],
            avg_diseased_fft_mag[:n_plot],
            label="Diseased (Avg FFT Magnitude)",
        )
        plot_successful = True

    if frequencies is not None and avg_healthy_fft_mag.size > 0:
        n_plot = len(frequencies)  # Positive frequencies already selected
        plt.plot(
            frequencies[:n_plot],
            avg_healthy_fft_mag[:n_plot],
            label="Healthy (Avg FFT Magnitude)",
        )
        plot_successful = True

    if plot_successful:
        plt.title("Average FFT Spectrum Comparison")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Average Magnitude (Log Scale)")
        plt.yscale('log')  # Use logarithmic scale for y-axis
        plt.legend()
        plt.grid(True)
    else:
        plt.title("Average FFT Spectrum Comparison (No data to plot)")
        print("No data available to plot.")

    plt.savefig("average_fft_spectrum.png")


if __name__ == "__main__":
    data_file_path = "data/traindata.csv"
    plot_average_fft_spectrum(data_file_path)
