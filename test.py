import os

import numpy.fft
from scipy.io import wavfile
import scipy.io
from pydub import AudioSegment
import matplotlib.pyplot as plt
import numpy as np
from os.path import join as pjoin
from scipy import signal
from scipy import fft
import math


note_string = ['C', 'C#/Db' , 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']


def path_to_numpy(file_path):
    wav_file_path = pjoin(os.getcwd(), file_path)
    sampling_rate, audio_signal = wavfile.read(wav_file_path)
    return sampling_rate, audio_signal


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
    numerator, denominator = signal.butter(5, [15, 8000], 'bandpass', fs=48000) #order 5 keep frequencies between 15hz and 8000hz. Diff not visiable
    filtered = signal.lfilter(numerator, denominator, raw_signal)
    return filtered


def frequency_to_note(frequency):
    log_peak_freq = math.log(frequency, 2)
    index = round((log_peak_freq-3.94802101634847)/0.0833334184464846)
    octave_num, note = index // 12, int(index % 12 - 1)
    return note_string[note] + str(octave_num)


def slice_and_find_note(audio_signal, frame_rate, section_length, threshold):
    total_duration = len(audio_signal) / frame_rate
    n_section = math.ceil(total_duration / (section_length / 1000))  # number of sections in the audio file. secionts split using section len
    sample_per_section = int((section_length / 1000) * frame_rate)
    for i in range(0, n_section):  # calculate each section and plot it
        section = audio_signal[i * sample_per_section: min((i + 1) * sample_per_section, len(filtered_signal))]  # chop into section
        freq, signal_amp = generate_freq_spectrum(section, frame_rate)  # fft
        peak_freq_index = numpy.argmax(signal_amp)  # get peak freq, return the index of the highest value
        note_name = frequency_to_note(freq[peak_freq_index])
        if signal_amp[peak_freq_index] > threshold:
            plt.text(i * sample_per_section, 0, note_name)  # plot note name
        plt.axvline(x=min((i + 1) * sample_per_section, len(signal_numpy)), color='r', linewidth=0.5, linestyle="-",zorder=10)  # lines for separating segments
        plt.text(i * sample_per_section, 2000, round(signal_amp[peak_freq_index]))  # plot the freq magnitude
    plt.plot(audio_signal, zorder=0)

def to_do():
    #clean up
    #interface mac mic
    # research terminalgy
    pass


if __name__ == "__main__":
    path = "Capstone piano test 1.wav"
    frame_rate, signal_numpy = path_to_numpy(path)
    filtered_signal = bandpass_filter(signal_numpy)
    section_len = 500  # length of each section in ms
    magnitude_threshold = 50
    slice_and_find_note(filtered_signal, frame_rate, section_len, magnitude_threshold)
    plt.show()

