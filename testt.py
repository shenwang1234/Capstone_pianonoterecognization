from scipy import fft, arange
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os


def frequency_spectrum(x, sf):
    """
    Derive frequency spectrum of a signal from time domain
    :param x: signal in the time domain
    :param sf: sampling frequency
    :returns frequencies and their content distribution
    """
    x = x - np.average(x)  # zero-centering

    n = len(x)
    k = np.arange(n)
    tarr = n / float(sf)
    frqarr = k / float(tarr)  # two sides frequency range

    frqarr = frqarr[range(n // 2)]  # one side frequency range

    x = fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]

    return frqarr, abs(x)


# # Sine sample with a frequency of 1hz and add some noise from STACKOVERFLOW
# sr = 32  # sampling rate
# y = np.linspace(0, 2*np.pi, sr) #return sr number of even spaced samples from 0-2pi
# y = np.tile(np.sin(y), 5) # create a array of np.sin(y) and repeat it 5 times. 5 times mean 5 cycles
# # plt.subplot(3, 1, 1)
# # plt.plot(y,'.-')
# rand=np.random.normal(0, 1, y.shape) #add in random noise
# yrand = y+ rand
# # print(y[0])
# # print(rand[0])
# # plt.subplot(3, 1, 2)
# # plt.plot(y,'.-')
# # plt.subplot(3, 1, 3)
# # plt.plot(rand,'.-')
# # plt.show()
# t = np.arange(len(y)) / float(sr) #cycle number
#
#
# plt.subplot(4, 1, 1)
# plt.plot(t, yrand, '.-') #time domain signal
# plt.xlabel('t')
# plt.ylabel('y')
# plt.subplot(4, 1, 2)
# plt.plot(t, y,'.-') #time domain signal
# plt.xlabel('t')
# plt.ylabel('y')
# frq, X = frequency_spectrum(yrand, sr)
#
# plt.subplot(4, 1, 3)
# plt.plot(frq, X, '.-')#frequency domain signal. how did we get frq scale? and magnitude scale?
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|X(freq)|')
# frq, X = frequency_spectrum(y, sr)
# plt.subplot(4, 1, 4)
# plt.plot(frq, X, '.-')#frequency domain signal. how did we get frq scale? and magnitude scale?
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|X(freq)|') # basically magnitude
# plt.tight_layout()
# plt.show()



##############this part is trying to test out how to change frequency of sine wave

# F = 10
# T = 10/F
# Fs = 5000
# Ts = 1./Fs
# N = int(T/Ts)
#
# t = np.linspace(0, T, N)
# signal = np.sin(2*np.pi*F*t)
#
# plt.plot(t, signal)
# plt.show()