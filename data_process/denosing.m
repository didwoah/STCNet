clear all
clc

folder_path = 'C:\Users\PC\Desktop\Ninapro4';
processed_folder_path = 'C:\Users\PC\Desktop\NinaPro4_processed';

list = dir(folder_path);
list = list(3:end);

for i = 1:numel(list)
    ls = dir(fullfile(folder_path, list(i).name));
    ls = ls(3:end);
    
    for j = 1:numel(ls)
        load(fullfile(folder_path, list(i).name, ls(j).name));
        emg = double(emg);
        
        % Call fn_prep_notch_lp function
        preprocessed_emg = fn_prep_notch_lp(emg);

        clear emg glove inclin acc rerepetition restimulus;

        % Save file
        save(fullfile(processed_folder_path, ls(j).name));
    end
end

function preprocessed_emg = fn_prep_notch_lp(trial_data)
    fs = 2000; % Sampling frequency
    f0 = 50; Q = 30;
    w0 = f0/(fs/2);
    bw = w0/Q;
    
    % 50Hz notch filtering
    [num, den] = iirnotch(w0, bw);
    emg_notch = filter(num, den, trial_data);
    
    % Butterworth low pass filtering
    fc = 500; % Cut-off frequency for low pass filter
    [b, a] = butter(1, fc / (fs/2)); % Butterworth low pass filter
    data_filtered = filtfilt(b, a, emg_notch);

    % % Wavelet denoising
    % L = floor(log2(m));
    % data_denoised = wdenoise(data_filtered, L, 'Wavelet', 'sym8', 'DenoisingMethod', 'UniversalThreshold'); 
    
    % Take absolute value
    preprocessed_emg = abs(data_filtered);
end
