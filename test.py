import numpy.fft
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import fft
import math


note_string = ['C', 'C#/Db' , 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


def path_to_numpy(file_path):
    song = AudioSegment.from_file(file_path)
    return song.get_array_of_samples(), song.frame_rate


def generate_freq_spectrum(x, sf):
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


def bandpass_filter(raw_signal):
    numerator, denominator = signal.butter(5, [15, 8000], 'bandpass', fs=48000)
    filtered = signal.lfilter(numerator, denominator, raw_signal)
    return filtered


def frequency_to_note(frequency):
    log_peak_freq = math.log(frequency, 2)
    index = round((log_peak_freq-3.94802101634847)/0.0833334184464846)
    octave_num, note = index // 12, int(index % 12 - 1)
    return note_string[note] + " " + str(octave_num)


if __name__ == "__main__":
    path = "Capstone piano test 1.m4a"
    signal_numpy, frame_rate = path_to_numpy(path)
    section_len = 500  # length of each section in s
    total_duration = len(signal_numpy)/frame_rate
    n_section = math.ceil(total_duration/(section_len/1000))
    sample_per_section = int((section_len/1000)*frame_rate)
    filtered_signal = bandpass_filter(signal_numpy)
    for i in range(0, n_section):
        section = signal_numpy[i * sample_per_section: min((i+1) * sample_per_section, len(signal_numpy))]  # chop into section
        freq, signal_amp = generate_freq_spectrum(section, frame_rate)  # fft
        peak_freq = numpy.argmax(signal_amp)    # get peak freq
        note_name = frequency_to_note(freq[peak_freq])  # convert freq to note
        plt.axvline(x=min((i + 1) * sample_per_section, len(signal_numpy)), color='r', linewidth=0.5, linestyle="-", zorder=10)  # lines for separating segments
        plt.text(i * sample_per_section, 0, note_name)  # plot note name
    #x_axis = np.linspace(0, audio_length/frame_rate, audio_length)
    plt.plot(signal_numpy, zorder=0)
    plt.show()
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
