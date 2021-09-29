'''Utility module for functions handling file operations, loading and saving annotations, ...'''

import csv
import datetime
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import re

from . import spectrogram_op
from . import nmf


def read_csv_file(filename, header_column_names=None, skip_header=True):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file, quotechar='"')
        if skip_header:
            header = next(reader, None)
            if header_column_names is not None:
                if len(header) != len(header_column_names):
                    raise Exception(f'CSV file should have {len(header_column_names)} columns')
        it = (l for l in reader if len(l) > 0 and not l[0].startswith('#'))
        for l in it:
            yield l


def iterate_songs_from_tracklist_file(filename, header_column_names=None):
    for l in read_csv_file(filename):
        song, tempo, start, label = l
        tempo, start, label = float(tempo), float(start), int(label)
        yield song, tempo, start, label


def filename_to_compact_string(fname):
    fname, _ = os.path.splitext(os.path.basename(fname))
    pattern = re.compile('([^\w]|_)+')
    return pattern.sub('', fname)


def create_odf_filename(song_filename, suffix=''):
    song_basename = filename_to_compact_string(song_filename)
    return f'{song_basename}{"-" if suffix else ""}{suffix}.npz'


def create_audio_component_filename(song_filename):
    song_basename = filename_to_compact_string(song_filename)
    return f'{song_basename}' + '{}.wav'
nmf_method=nmf.apply_nmfd


def annotate_odf_using_nmf(y, nmf_method = nmf.apply_nmfd,
                          audio_file=None, sr=44100, tempo=None, hpss=False,):

    # Calculate the spectrogram using the chosen method
    S_orig = nmf_method.spectrogram_fn(y)

    # Perform HPSS
    if hpss:
        _, S_perc = librosa.decompose.hpss(S_orig,
                                           kernel_size=(31, S_orig.shape[1] // 32),
                                           power=2.0, mask=False, margin=2.0)
    else:
        S_perc = S_orig

    # Perform NMF
    n_components = 3
    nmfdA, nmfdW, nmfdH = nmf_method(S_perc, n_components, tempo=tempo)

    # Save the individual onset functions
    odfs = {}
    odfs['odf_kick'] = nmfdH[0]
    odfs['odf_snare'] = nmfdH[1]
    odfs['odf_hihat'] = nmfdH[2]

    # Save the audio corresponding to each instrument
    audios = {}
    for k, name in enumerate(['kick', 'snare', 'hihat']):
        Y = nmfdA[k] * np.exp(1j * np.angle(S_orig))
        y = nmf_method.inverse_spectrogram_fn(Y)
        audios[name] = y

    return odfs, audios


def load_odf_annotation_for_song(song_filename, annotation_dir):

    odf_file = os.path.join(annotation_dir, create_odf_filename(song_filename))

    # Load the ODFs
    odfs = np.load(odf_file)
    odf_kick, odf_snare, odf_hihat = odfs['odf_kick'], odfs['odf_snare'], odfs['odf_hihat']

    # Store the audio and ODF for returning
    return odf_kick, odf_snare, odf_hihat


def load_songs_annotations(songs, tempos, start_times,
                           annotation_dir='', force_odf_calculation=False,
                           annotation_method=None,  # Annotation method, taking as input the audio and the odf file
                           silent=False,
                           **annotation_method_kwargs,
                           ):

    odfs_for_song = {}
    audio_for_song = {}

    for song_filename, tempo, s_start in zip(songs, tempos, start_times):

        odf_file = os.path.join(annotation_dir, create_odf_filename(song_filename))
        audio_file = os.path.join(annotation_dir, create_audio_component_filename(song_filename))
        if not silent:
            print(odf_file)
            print(audio_file)

        # Load the audio
        L_segment = 16 * 60.0 / tempo
        y, sr = librosa.load(song_filename, 44100, offset=s_start, duration=L_segment)

        # Annotate the song if it doesn't have an annotation yet
        if not os.path.exists(odf_file) or force_odf_calculation:
            print(audio_file)
            odfs, audios = annotation_method(y, tempo=tempo, path_to_audio=song_filename, **annotation_method_kwargs,)
            np.savez(odf_file, **odfs)  # Note: the keys of the ODFs dict should be odf_kick, odf_snare, odf_hihat
            if audios is not None:
                for i, (k, y) in enumerate(audios.items()):
                    librosa.output.write_wav(audio_file.format(i), y, 44100)

        # Load the ODFs
        odfs = np.load(odf_file)
        odf_kick, odf_snare, odf_hihat = odfs['odf_kick'], odfs['odf_snare'], odfs['odf_hihat']

        # Store the audio and ODF for returning
        odfs_for_song[filename_to_compact_string(song_filename)] = (odf_kick, odf_snare, odf_hihat,)
        audio_for_song[filename_to_compact_string(song_filename)] = y

    return odfs_for_song, audio_for_song


def save_plot(prefix, path_to_music, n_comp, isRhythmicSoftConstr):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    plt.savefig(f'results_{prefix}/{timestamp}'
                f'_nmf_{filename_to_compact_string(path_to_music)}_{n_comp}'
                f'_{"rhythm" if isRhythmicSoftConstr else "norhythm"}_')
