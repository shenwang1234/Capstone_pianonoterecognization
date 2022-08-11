from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
from scipy import fft
import math


def path_to_numpy(path):

    song = AudioSegment.from_file(path)
    return song.get_array_of_samples(), song.frame_rate


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
    tarr = n / float(sf) #sf increase tarr decrease. tar is float
    frqarr = k / float(tarr)  # two sides frequency range
    frqarr = frqarr[range(n // 2)]  # one side frequency range
    x = np.fft.fft(x) / n  # fft computing and normalization
    x = x[range(n // 2)]
    return frqarr, abs(x)


def convert_to_decibel(arr):
    ref = 1
    if arr != 0:
        return 20 * np.log10(abs(arr) / ref)
    else:
        return -60


if __name__ == "__main__":
    path = "Capstone piano test 1.m4a"
    signal_numpy, frame_rate = path_to_numpy(path)
    total_time_of_signal = len(signal_numpy)/frame_rate
    numerator, denominator = signal.butter(5, [15, 8000], 'bandpass', fs=48000)
    one_three_sec = signal_numpy[math.floor(frame_rate): math.floor(3.5*frame_rate)]
    filtered = signal.lfilter(numerator, denominator, one_three_sec)
    # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    # plt.title("filtered")
    # plt.plot(filtered)
    # plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    # plt.title("un-filtered")
    # plt.plot(one_three_sec)
    # plt.show()




    # frq, Y = frequency_spectrum(numeric_array[48432:169512], 44100) #X is half len of input signal len
    # print(time.time()-start)
    # #1-3.5s is roughly index 48432-169512
    # data = [convert_to_decibel(i) for i in Y]
    # plt.plot(frq, data, '.-')
    # plt.xlabel('Freq (Hz)')
    # plt.ylabel('|X(freq)|')
    # plt.xscale('log')
    # print(time.time()-start)
    # peaks= find_peaks(data)
    # print(len(peaks[0]))# need to change parameter for this. right now its getting 20045 peaks
    # plt.show()
    # start = time.time()
    # song = AudioSegment.from_file("Capstone piano test 1.m4a") # this is a bytestring array with length 27243.
    # song = song.low_pass_filter(8000)
    #
    # song = song.high_pass_filter(10)
    # #each index correspond to a ms
    #
    # #this gets the dbfs of the signal and plot it against time(s)
    # # SEGMENT_MS = 50
    # # volume = [segment.dBFS for segment in song[::SEGMENT_MS]]
    # # x_axis = np.arange(len(volume)) * (SEGMENT_MS / 1000)
    # # plt.subplot(1, 2, 1) # row 1, col 2 index 1
    # # plt.plot(x_axis, volume)
    #
    #
    # #iphone voice memo sampling fequency 44.1kHz
    # bit_depth = song.sample_width * 8 #sample_width is the number of bytes in each sample (same as Quantisation? )
    # array_type = get_array_type(bit_depth)
    # numeric_array = array.array(array_type, song._data)
    # print(time.time()-start)
    #
    #
    # # Time = np.linspace(0, len(numeric_array) / 44100, num=len(numeric_array))
    # # x_coord = [i for i in range(len(numeric_array))]
    # # plt.plot(Time, numeric_array)
    # # plt.show()
