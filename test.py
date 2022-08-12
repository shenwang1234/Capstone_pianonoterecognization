import numpy.fft
import scipy.signal
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
from scipy import fft
import math


note_string = ['C', 'C#/Db' , 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B' ]


def build_note_dictionary(freq_of_first_octave, num_of_octave):
    dict = freq_of_first_octave
    for num in range(1,num_of_octave+1):
        for note in freq_of_first_octave.keys():
            for m in note:
                if m.isdigit():
                    digit = m
            note.replace(str(digit), str(num))
            dict[note] = note*2**num
    return dict


def frequency_to_index(peak_freq):
    log_peak_freq =math.log(peak_freq,2)
    index = (log_peak_freq-3.94802101634847)/0.0833334184464846
    return round(index),index


def path_to_numpy(path):

    song = AudioSegment.from_file(path)
    return song.get_array_of_samples(), song.frame_rate


def frequency_spectrum(x,sf):
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

    frqarr = frqarr[range(min(n // 2, 8000))]  # one side frequency range

    x = fft.fft(x) / n  # fft computing and normalization
    x = x[range(min(n // 2, 8000))]

    return frqarr, abs(x)


def convert_to_decibel(arr):
    ref = 1
    if arr != 0:
        return 20 * np.log10(abs(arr) / ref)
    else:
        return -60


def chop(raw_signal, start_end, frame_rate):
    try:
        chopped = raw_signal[math.floor(start_end[0]*frame_rate): math.floor(start_end[1] * frame_rate)]
    except IndexError:
        return "you fucked"
    return chopped


def find_max_peak(freq_domain_signal,freq_axis):
    peak_index = numpy.argmax(freq_domain_signal)
    return freq_axis[peak_index], freq_domain_signal[peak_index]



def map_freq_to_note(freq):
    pass

def todo_next():
    '''
    1. chop up audio into x ms sequences done
    2. run fft on each segment
    3. identify the highest peak in all those segments
    4. output a sequence of labeled notes
    '''
    pass


def bandpass_filter(raw_signal):
    numerator, denominator = signal.butter(5, [15, 8000], 'bandpass', fs=48000)
    filtered = signal.lfilter(numerator, denominator, raw_signal)
    return filtered


def index_to_name(index):
    #12 keys in one octave
    octave_num = index//12
    note = int(index - octave_num*12-1)
    return note_string[note]+" "+str(octave_num)




if __name__ == "__main__":
    path = "Capstone piano test 1.m4a"
    signal_numpy, frame_rate = path_to_numpy(path)
    for sec in range(0,27):
        slice = chop(signal_numpy, [sec, sec+1], frame_rate)
        filtered_signal = bandpass_filter(slice)
        freq, fourier = frequency_spectrum(filtered_signal,48000)
        peak_freq = find_max_peak(fourier, freq)
        loged_num = frequency_to_index(peak_freq[0])
        print(index_to_name(loged_num[0]))

    # plt.subplot(1, 2, 1)  # row 1, col 2 index 1
    # plt.plot(freq, fourier)
    # peak_index = find_max_peak(fourier, freq)
    # plt.plot(peak_index[0], peak_index[1], marker="o", markersize=20, markeredgecolor="red", markerfacecolor="green")
    # plt.subplot(1, 2, 2)  # row 1, col 2 index 1
    # fourier = fft.fft(signal_numpy)
    # plt.plot(signal_numpy)
    # print(frequency_to_index(peak_index[0]))
    # plt.show()

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
