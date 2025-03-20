% filepath: signal_analysis.m
% Load the .mat file
% Replace 'your_file.mat' with your actual .mat filename
data = traindata;

% Assuming your signal is stored in a variable within the .mat file
% Adjust the variable name according to your data structure
sig = data(1, :);
size(sig)
fs = 400;  % Sampling frequency
duration = 10;  % Duration in seconds

% Calculate time vector
n = length(sig);
t = linspace(0, duration, n);

% Perform FFT
freq_components = fft(sig);
freq = linspace(0, fs/2, n/2);

% Create figure with subplots
figure('Position', [100, 100, 800, 600]);

% Frequency spectrum
subplot(3,1,1);
plot(freq, 2*abs(freq_components(1:n/2))/n);
title('Frequency Spectrum (Input)');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
grid on;

% Time domain signal
subplot(3,1,2);
reconstructed_signal = ifft(freq_components);
plot(t, real(reconstructed_signal));
title('Reconstructed Signal (IFFT)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Phase spectrum
subplot(3,1,3);
plot(freq, angle(freq_components(1:n/2)));
title('Phase Spectrum');
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
grid on;

% Adjust subplot spacing
sgtitle('Signal Analysis');