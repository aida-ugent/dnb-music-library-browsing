'''Utility module with functions for spectrogram processing.'''

import librosa
import numpy as np

from . import odf_op


def stft(audio, n_fft = 2048, hop_length = 512, **kwargs):

    S = librosa.stft(audio, n_fft=n_fft, hop_length = hop_length, **kwargs)
    S = np.abs(S)
    S[:3] = 0  # Important: filter out bass frequencies that are way too low

    return S


def stft_beatwise_average(audio, tempo, **kwargs):

    S = stft(audio, **kwargs)
    return beatwise_averaged_spectrogram(S, tempo, **kwargs)


def cqt(audio, sr=44100, hop_length=512, bins_per_note=3, n_octaves=8, log_amplitude=False, filter_scale=0.2, **kwargs):
    S = librosa.cqt(audio, hop_length=hop_length, sr=sr,
                    bins_per_octave=12*bins_per_note, n_bins=12*bins_per_note*n_octaves,
                    filter_scale=filter_scale,
                    **kwargs)
    S = np.abs(S)

    if log_amplitude:
        S = librosa.amplitude_to_db(S, ref=np.max)

    return(S)


def istft(S):
    return librosa.istft(S, hop_length=512)


def icqt(S, hop_length=512, bins_per_note=3,):
    return librosa.icqt(S[:-12*bins_per_note], sr=44100, hop_length=hop_length,
                        bins_per_octave=12*bins_per_note, )


def beatwise_averaged_spectrogram(S, tempo, sr=44100, hop_length=512):

    L = 4 * 60 * sr / (tempo * hop_length)
    L_int = int(L)
    start_indices = [int(L * i) for i in range(4)]
    S_perc_fragments = [S[:, idx: idx + L_int] for idx in start_indices]
    return odf_op.geom_mean(*S_perc_fragments)


def remove_noise_percentile_filter(S, p=50):
    return (S - np.percentile(S, p, axis=1, keepdims=True)).clip(0)
