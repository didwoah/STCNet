# augmentation
import torch
import numpy as np
from scipy.interpolate import CubicSpline
import pywt
import random

# : for GN, the SNR = 30, for MW,
# the σ = 0.1 and for WD, the ψ = sym4, l = 5, b = 3. In putEMG, the corresponding values were:
# SNR = 35, σ = 0.1 and ψ = db7, l = 5, b = 0.

def generate_random_boolean(p):
    return random.random() < p

class GaussianNoise(object):
    def __init__(self, mean = 0., SNR = 30, p = .5):
        self.mean = mean
        self.SNR = SNR
        self.p = p
        
    def __call__(self, x):
        if not generate_random_boolean(self.p):
            return x
        squared_sum = torch.sum(torch.pow(x, 2))
        num_elements = x.numel()
        squared_mean = squared_sum / num_elements
        std = torch.sqrt(squared_mean / self.SNR)
        return x + torch.randn(x.size()) * std + self.mean
    
class MagnitudeWarping(object):
    def __init__(self, mean = 1., std = 0.1, T = 6, p = .5, mode = 'channel'):
        self.mean = mean
        self.std = std
        self.T = T
        self.p = p
        self.mode = mode

    def __call__(self, x):
        if not generate_random_boolean(self.p):
            return x
        
        if self.mode == 'channel':
            selected_time_points = torch.linspace(0, x.size()[0] - 1, 6)
            selected_values = torch.normal(mean=self.mean, std=self.std, size=(6, x.size()[1]))
            tmp = np.arange(x.size()[0])
            interpolated_values = []
            for i in range(x.size()[1]):  # x.size()[1] features에 대해 각각 interpolate
                cs = CubicSpline(selected_time_points.numpy(), selected_values[:, i].numpy())
                interpolated_values.append(cs(tmp))

            interpolated_values_tensor = torch.tensor(np.array(interpolated_values)).T

            return x * interpolated_values_tensor
        elif self.mode == 'signal':
            selected_time_points = torch.linspace(0, x.size()[0] - 1, 6)
            selected_values = torch.normal(mean=self.mean, std=self.std, size=(6))
            tmp = np.arange(x.size()[0])
            cs = CubicSpline(selected_time_points.numpy(), selected_values[:, i].numpy())

            interpolated_value = np.array(cs(tmp))
            interpolated_values = np.repeat(interpolated_value[:, np.newaxis], x.size()[1], axis=1)

            interpolated_values_tensor = torch.tensor(np.array(interpolated_values)).T

            return x * interpolated_values_tensor
        else:
            raise ValueError(self.mode)


class WaveletDecomposition(object):
    def __init__(self, wavelet = 'sym4', level = 5, b = 3., p=.5):
        self.wavelet = wavelet
        self.level = level
        self.b = b
        self.p = p

    def __call__(self, x):
        if not generate_random_boolean(self.p):
            return x
        coeffs_rows = []
        for i in range(x.shape[1]):
            # Perform DWT for the current dimension
            coeffs = pywt.wavedec(x[:, i].numpy(), self.wavelet, level = self.level)
            # coeffs is list [cA, cD(level max), ..., cD(level 1)]

            # Store the detail and approximation coefficients
            coeffs_rows.append(coeffs)

        coeffs_rows = [[arr * self.b if i > 0 else arr for i, arr in enumerate(lst)] for lst in coeffs_rows]

        # Perform Inverse Discrete Wavelet Transform (IDWT)
        reconstructed_data = []  # Store the reconstructed data
        for i in range(len(coeffs_rows)):
            coeff = coeffs_rows[i]
            # Perform IDWT
            reconstructed_signal = pywt.waverec(coeff, self.wavelet)
            # Store the reconstructed signal
            reconstructed_data.append(reconstructed_signal)

        return torch.tensor(np.array(reconstructed_data)).T
    
class Permute(object):
    def __init__(self, data, model):
        if data in ['nina1']:
            self.channel = 10
        elif data in ['nina2', 'nina4']:
            self.channel = 12
        self.model = model
    def __call__(self, x):
        x = x.reshape((25, 20, self.channel))
        if self.model == 'EMGHandNet':
            x = np.transpose(x,(0,2,1))
        return x.float().clone().detach()
    

if __name__ == '__main__':
    # Example usage of WaveletDecomposition class
    wavelet_decomposition = WaveletDecomposition()
    reconstructed_data = wavelet_decomposition(torch.randn(500, 10))
    print("Reconstructed data shape:", reconstructed_data.shape)