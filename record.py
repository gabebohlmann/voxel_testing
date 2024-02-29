# .py file equivalent of record.ipynb notebook

### imports ###
import os
import time

import sounddevice
import sounddevice as sd
import soundfile as sf
from time import sleep
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm.notebook import tqdm
import glob
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from scipy import fft
from scipy.signal import chirp
import math
import subprocess
import librosa
import torchaudio
from torch.nn import functional as F
import torchaudio.functional as Fa
import torch
import torch.fft
from IPython import embed
import librosa
import subprocess


### functions ###
def play_android(wav_file_path, fs_rec):
    # print(wav_file_path)
    data, fs_org = sf.read(wav_file_path)
    duration = librosa.get_duration(filename=wav_file_path)
    print('File duration = ', duration)
    file_name = '/'.join(wav_file_path.split('/')[-3:])
    print(file_name)
    subprocess.run([
        "adb shell am start -a android.intent.action.VIEW -d file:///storage/emulated/0/{}  -t video/* -f 0x00008000".format(
            file_name)], shell=True)
    print('Start Recording at fs = {}:'.format(fs_rec))
    rec = sd.rec(int(duration * fs_rec), samplerate=fs_rec, channels=2, blocking=True)
    print("Recording Done!")
    rec_in = rec[:, 0]  # inside box
    rec_out = rec[:, 1]  # outside box
    return data, rec_in, rec_out, duration


def get_f0_Q_tube(L, r, correct=False):
    # open_ended tube:
    n = 1
    T = 24  ## Temp in Celsius
    v = 20.05 * math.sqrt(T + 273.15)  #343   # sound speed in air in m/s
    f0 = n * v / 2 / L
    ## Corrected frequency (wiki)
    if correct:
        print('Correction is Applied!')
        f0 = n * v / 2 / (L + 0.8 * 2 * r)
    print("resonant frequency = ", f0)
    # Quality factor method 1:
    A = math.pi * r ** 2  # radius in meters
    d_rad = 5.34 * 10 ** -5 * A * f0 ** 2
    d_wall = 5.71 * 10 ** -3 * (A * f0) ** -0.5
    Q = round(1 / (d_rad + d_wall))
    print("Quality Factor = ", Q)
    # # Quality factor method 2:
    # x = 3.8  # 3500                 # x = (r/d)**2
    # Q2 = round(math.sqrt(x)/1.5)
    # print("Quality Factor 2 = ", Q2)
    return f0, Q


def resonant_filter_torch(wav_path, f0, Q, num_harmonics, save_audio=True):
    waveform, fs = torchaudio.load(wav_path)
    # print('Original wav limits:', torch.max(waveform), torch.min(waveform))
    # print(waveform.shape)
    y_sum = torch.zeros(waveform.shape)
    for i in range(1, num_harmonics + 1):
        y_sum += Fa.band_biquad(waveform, fs, i * f0, Q, noise=False)
    y_sum = y_sum / torch.max(torch.abs(y_sum))

    z_sum = torch.zeros(waveform.shape)
    for i in range(1, num_harmonics + 1):
        z_sum += Fa.bandpass_biquad(waveform, fs, f0 * i, Q, const_skirt_gain=False)
    z_sum = z_sum / torch.max(torch.abs(z_sum))

    return waveform, y_sum, z_sum, fs


def get_transfer_function_scipy(sig_org, sig_out, sig_in, fs, fs_org, T, save_fig=False, title=''):
    """
    :param org: original signal
    :param sig_in: signal recorded Inside the box
    :param sig_out: signal recorded Outside the box
    :param fs: sampling rate
    :return: TF: transfer function = FFT(sig2)/FFT(sig1)
    """
    l1 = 15
    l2 = 10
    d1 = 1
    d2 = 2
    ratio = (d2 / d1) ** 2
    x = np.arange(1, 7500, 1)
    out = -1 / np.tan(2 * math.pi * x * l1 / 34350) - ratio * 1 / np.tan(2 * math.pi * x * l2 / 34350)
    zero_crossings = np.where(np.diff(np.sign(out)).astype(bool) & (out[:-1] < 0.5) & (out[:-1] > -0.5))[0]

    num_fft = len(sig_in)
    sig_org_fft = np.abs(fft.rfft(sig_org, norm="ortho", n=num_fft))
    sig_in_fft = np.abs(fft.rfft(sig_in, norm="ortho"))
    sig_out_fft = np.abs(fft.rfft(sig_out, norm="ortho"))
    freq_org = fft.rfftfreq(num_fft, d=1 / fs_org)
    freq = fft.rfftfreq(num_fft, d=1 / fs)

    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.setp(ax, xticks=np.arange(0, fs / 2, 250))
    plt.xticks(rotation=45)
    plt.xlim([-1, 4000])
    ax[0].plot(freq_org, sig_org_fft, color='black', alpha=0.8, label='Original')
    ax[0].legend(loc='upper right')
    ax[1].plot(freq, sig_out_fft, color='blue', alpha=0.8, label='out_box')
    ax[1].legend(loc='upper right')
    ax[2].plot(freq, sig_in_fft, color='red', alpha=0.8, label='In_box')
    ax[2].legend(loc='upper right')
    for j in zero_crossings:
        ax[2].axvline(x=j, c='b', ls="--", alpha=0.3)
    plt.tight_layout()
    if save_fig:
        fig_save_dir = title.replace("out", "figures").replace('.wav', '_FFT.png')
        pathlib.Path(fig_save_dir).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_save_dir)
    plt.show()


def load_play_save(dataset, save_dir, fs_rec=16000, save=True):
    if '16' in dataset:
        fs_org = 16000
    else:
        fs_org = 8000
    for i, wav_file_path in enumerate(glob.glob("../Dataset/VoxCeleb/*/*.wav")):
        sleep(3)
        try:
            print('Original Dataset fs =', fs_org)
            print('Now playing file {}: '.format(i), os.path.basename(wav_file_path))
            ## Play using android:
            org, rec_in, rec_out, T = play_android(wav_file_path.replace("VoxCeleb", dataset), fs_rec)
            ## Play using PC connected speakers:
            # org, rec_in, rec_out, fs = play_record_2mic(wav_file_path)
            outbox_path = wav_file_path.replace("VoxCeleb", save_dir + "/out")
            inbox_path = wav_file_path.replace("VoxCeleb", save_dir + "/in")
            inbox_path_normalized = wav_file_path.replace("VoxCeleb", save_dir + "/in_normalized")
            if save:
                pathlib.Path(outbox_path).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(inbox_path).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(inbox_path_normalized).parent.mkdir(parents=True, exist_ok=True)
                sf.write(inbox_path, rec_in, fs_rec)
                rec_in_norm = rec_in / max(abs(rec_in)) * max(abs(org))
                rec_out_norm = rec_out / max(abs(rec_out)) * max(abs(org))
                sf.write(inbox_path_normalized, rec_in_norm, fs_rec)
                sf.write(outbox_path, rec_out_norm, fs_rec)
            get_transfer_function_scipy(org, rec_out_norm, rec_in_norm, fs_rec, fs_org, T, save, title=outbox_path)

        except KeyboardInterrupt:
            print('\nPausing...  (Hit ENTER to continue, type quit to exit.)')
            try:
                response = input()
                if response == 'quit':
                    break
                print('Resuming...')
            except KeyboardInterrupt:
                print('Resuming...')
                continue


def load_play_save_dataset(dataset, save_dir, fs_rec=16000, save=True):
    if '16' in dataset:
        fs_org = 16000
    else:
        fs_org = 8000
    for i, wav_file_path in enumerate(glob.glob("../Dataset/{}/*/*.wav".format(dataset))):
        try:
            if os.path.exists(wav_file_path.replace(dataset, save_dir + "/in")):
                print(wav_file_path, 'exists already')
                continue
            print('Original Dataset fs =', fs_org)
            print('Now playing file {}: '.format(i), os.path.basename(wav_file_path))
            ## Play using android:
            org, rec_in, rec_out, T = play_android(wav_file_path, fs_rec)
            ## Play using PC connected speakers:
            # org, rec_in, rec_out, fs = play_record_2mic(wav_file_path)
            outbox_path = wav_file_path.replace(dataset, save_dir + "/out")
            inbox_path = wav_file_path.replace(dataset, save_dir + "/in")
            inbox_path_normalized = wav_file_path.replace(dataset, save_dir + "/in_normalized")
            if save:
                pathlib.Path(outbox_path).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(inbox_path).parent.mkdir(parents=True, exist_ok=True)
                pathlib.Path(inbox_path_normalized).parent.mkdir(parents=True, exist_ok=True)
                sf.write(inbox_path, rec_in, fs_rec)
                rec_in_norm = rec_in / max(abs(rec_in)) * max(abs(org))
                rec_out_norm = rec_out / max(abs(rec_out)) * max(abs(org))
                sf.write(inbox_path_normalized, rec_in_norm, fs_rec)
                sf.write(outbox_path, rec_out_norm, fs_rec)
                # sf.write(outbox_path, rec_out, fs_rec)

            get_transfer_function_scipy(org, rec_out_norm, rec_in_norm, fs_rec, fs_org, T, save, title=outbox_path)
        except KeyboardInterrupt:
            print('\nPausing...  (Hit ENTER to continue, type quit to exit.)')
            try:
                response = input()
                if response == 'quit':
                    break
                print('Resuming...')
            except KeyboardInterrupt:
                print('Resuming...')
                continue

def swept_sin(save, fs_org, fs_rec):
    T = 3
    # fs_org = 8000
    t = np.arange(0, int(T * fs_org)) / fs_org
    f0 = 7800
    f1 = 100

    w = chirp(t, f0=f0, f1=f1, t1=T, method='linear')
    # print(w)
    file_name = 'chirp_{}_{}_{}s_{}.wav'.format(f1, f0, T, fs_org)
    # sf.write('../Dataset/chirp/'+file_name, w, fs_org)
    # plt.plot(t,w)
    # plt.show()
    # w_fft = np.abs(fft.rfft(w, norm="ortho"))
    # freq = fft.rfftfreq(len(w), d=1/fs_org)
    # plt.plot(freq, w_fft, alpha=0.8)
    # plt.show()
    # rec = sd.playrec(w, samplerate=fs, channels=2, blocking=True)

    # file_name = os.path.basename(wav_file_path)
    # subprocess.run(["adb push ../Dataset/chirp/{} /sdcard/chirp/".format(file_name)], shell=True)
    subprocess.run([
                       "adb shell am start -a android.intent.action.VIEW -d file:///storage/emulated/0/chirp/{}  -t video/* -f 0x00008000".format(
                           file_name)], shell=True)
    # subprocess.run(["adb shell am start -a android.intent.action.VIEW -d file:///storage/emulated/0/{}  -t video/* -f 0x00008000".format("id00012_21Uxsk56VDQ_00006_00000.wav")], shell=True)

    # fs = 8*1000
    print('Start Recording at fs :')
    fs = fs_rec
    rec = sd.rec(int(T * fs), samplerate=fs, channels=2, blocking=True)
    print("Recording Done!")
    rec_in = rec[:, 0]  # inside box
    rec_out = rec[:, 1]  # outside box
    rec_in_norm = rec_in / max(abs(rec_in)) * max(abs(w))
    rec_out_norm = rec_out / max(abs(rec_out)) * max(abs(w))
    # sf.write('../Dataset/Box_phone_chirp/direct_far_out_{}.wav'.format(fs), rec_out, fs)
    if save:
        sf.write('../Dataset/Box_phone_chirp/{}_{}_play{}_org.wav'.format(L_in, r, fs_org), w, fs)
        sf.write('../Dataset/Box_phone_chirp/{}_{}_play{}_in.wav'.format(L_in, r, fs_org), rec_in, fs)
        sf.write('../Dataset/Box_phone_chirp/{}_{}_play{}_out.wav'.format(L_in, r, fs_org), rec_out, fs)
    # _, band_filter, bandpass_filter, _ = resonant_filter_torch('../Dataset/'+file_name, f0, Q, round(fs/2/f0), False)
    # band_filter = band_filter.cpu().detach().numpy().flatten()
    # bandpass_filter = bandpass_filter.cpu().detach().numpy().flatten()
    get_transfer_function_scipy(w, rec_out_norm, rec_in_norm, fs, fs_org, T, save_fig=save,
                                title='../Dataset/Box_phone_chirp/{}_{}_play{}_16k.wav'.format(L_in, d, fs_org))
    # get_transfer_function_scipy(rec_in, band_filter, bandpass_filter, fs, save_fig=False, title='sin_swept')


### function calls ###
fs_rec = 16000
wav_file_path = 'm_hamil.wav'
data, fs_org = sf.read(wav_file_path)
duration = librosa.get_duration(path=wav_file_path)
print('File duration = ', duration)
file_name = '/'.join(wav_file_path.split('/')[-3:])
print(file_name)
subprocess.run([
    "adb shell am start -a android.intent.action.VIEW -d file:///storage/emulated/0/{}  -t video/* -f 0x00008000".format(
        file_name)], shell=True)
print('Start Recording at fs = {}:'.format(fs_rec))
rec = sd.rec(int(duration * fs_rec), samplerate=fs_rec, channels=2, blocking=True, device=1)
print("Recording Done!")
rec_in = rec[:, 0]  # inside box
rec_out = rec[:, 1]  # outside box