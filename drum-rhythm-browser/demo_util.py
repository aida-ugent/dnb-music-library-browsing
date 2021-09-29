import librosa
import numpy as np
from util_nmf_experiment import evaluation
from util_nmf_experiment import odf_op
from util_nmf_experiment import spectrogram_op
from util_nmf_experiment import files

import spleeter as spltr
from spleeter.separator import Separator as Spleeter
import soundfile as sf

spltr = Spleeter('spleeter:4stems')

def get_cqt_from_audio(y):
    # Calculate spectrogram
    # NOTE: filter scale helps a lot in time resolution!
    return spectrogram_op.cqt(y, filter_scale=0.2)


def get_onsets_from_spectrogram(S):
    channels = [
        slice(0, 50, 1),  # Low, kick
        slice(80, 200, 1),  # Low + Mid, ,kick+snare
        # slice(240, 288, 1),  # High, hihat
    ]

    def smoothen(x):
        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        kernel = gaussian(np.linspace(-.25, .25, 19), 0, 1)
        x_ = np.convolve(x, kernel, mode='full')
        x_ = x_[9:-9]
        x_ /= np.max(x_)
        # make peaks "rounder" by replacing value in original ODF only if mean of surroundings
        # actually makes them higher
        return np.maximum(x, x_)

    odfs = librosa.onset.onset_strength_multi(S=S, channels=channels, center=False, lag=2)
    odfs /= np.max(odfs, axis=1)[:, np.newaxis]  # Normalize each ODF
    kick_smoothened = smoothen(odfs[0, :])
    odfs[1, :] *= (1 - kick_smoothened)

    return [odfs[i] for i in range(odfs.shape[0])]


def standardize_align_odfs(odfs):
    tempo = 175 / 4
    sr, hopsize = 44100, 512
    N = int(16 * (tempo / 175))
    L = int(60.0 / (tempo) * N * sr)
    dummy_onsets = [60.0 / (tempo) * i for i in range(N)]
    reference_odf = evaluation.onsets_to_dummy_odf(dummy_onsets, sr, hopsize, audio_length_samples=L,
                                                   kernel_length_ms=150)

    tempo = 175
    N = int(16 * (tempo / 175))
    L = int(60.0 / (tempo) * N * sr)
    dummy_onsets = [60.0 / (tempo) * i for i in range(N)]
    reference_odf_fine = evaluation.onsets_to_dummy_odf(dummy_onsets, sr, hopsize, audio_length_samples=L,
                                                        kernel_length_ms=150)

    L_odf = len(reference_odf)

    for i, odf in enumerate(odfs):
        odfs[i] = odf_op.set_array_length(odf, L_odf)

        # Determine coarse alignment based on kick drum
        if i == 0:
            n_shift_coarse = odf_op.find_best_overlap(odfs[0], reference_odf, L_odf // 16)
            # print(f'Shifting: {n_shift_coarse}, one measure is {L_odf // 4} long.')

        odfs[i], _ = odf_op.align_odfs(odfs[i], None, n_shift=n_shift_coarse, hold_y=True)

        # Perform a fine alignment to avoid any problems with kick, snare, hh not exactly aligning
        if i != 0:
            odfs[i], _ = odf_op.align_odfs(odfs[i], reference_odf_fine, max_shift=2, hold_y=True)

        odfs[i] /= np.max(odfs[i])
    return odfs


def annotate_song_odf(song, tempo, start, filename=None):
    if filename is None:
        raise Exception('Please provide an annotation directory.')

    # Load the audio
    L_s = (60.0 / tempo) * 4 * 4
    y, sr = librosa.load(song, sr=44100, offset=start, duration=L_s)

    # Spleeter the audio
    y = y.reshape(-1, 1)
    y_separated = spltr.separate(y)['drums']
    y_separated = librosa.to_mono(y_separated.T)

    # Calculate spectral representation
    S = get_cqt_from_audio(y_separated)

    # Calculate onset functions for kick and snare
    odfs = list(get_onsets_from_spectrogram(S))

    # Calculate onsets for hi-hat
    n_fft, hop_length = 2048, 512
    S_fft = librosa.amplitude_to_db(
        spectrogram_op.stft(y.flatten(), n_fft=n_fft, hop_length=hop_length)
    )
    f_hp = 7000  # high pass the stft spectrogram at this frequency to detect hihats
    f_max = (44100 / 2)  # Nyquist frequency
    high_pass_bin = int(f_hp / f_max * n_fft)
    S_fft[:high_pass_bin, :] = 0
    odf_hihat = librosa.onset.onset_strength(S=S_fft, center=False)
    odf_hihat = (odf_hihat - np.median(odf_hihat)).clip(0)
    odf_hihat /= np.max(odf_hihat)
    odfs.append(odf_hihat)

    # Coarsely align with the dummy ODF, based on the kick drum
    odfs = standardize_align_odfs(odfs)

    # Save as .npy file
    np.save(filename, odfs)

    # Save audio as .wav file
    sf.write(filename + '.wav', y_separated, 44100)

    return odfs


def odf_to_beat_grid(y, tempo=None, hop_size=None, sr=44100, grid_note=16, ):
    # Tempo is the tempo of quarter notes
    if tempo is None:
        raise Exception('Please provide the tempo.')
    if hop_size is None:
        raise Exception('Please provide the ODF hop size in samples.')
    if tempo is None:
        raise Exception('Please provide the tempo.')
    if int(np.log2(grid_note)) != np.log2(grid_note):
        raise Exception('Grid note size should be a power of two.')

    if np.ndim(y) == 1:  # Input is a single ODF
        y = np.expand_dims(y, axis=0)
    elif np.ndim(y) > 2:
        raise Exception('Input should be one-dimensional (single ODF) or a matrix (each row is an individual ODF).')

    seconds_per_grid_note = (60 / (tempo * (grid_note / 4)))
    hops_per_second = sr / hop_size
    hops_per_grid_note = hops_per_second * seconds_per_grid_note  # ODF hops per grid element
    n_total_odf_hops = y.shape[1]
    n_total_notes = int(np.ceil(n_total_odf_hops / hops_per_grid_note))

    # phase_curve = beattracker.BeatTracker.sum_curve_at_intervals(y[1],
    #                                                [hops_per_grid_note],
    #                                                valid_offsets = np.arange(0, np.ceil(hops_per_grid_note)))
    offset = 0  # Previous alignment with dummy ODF took care of this

    boundaries = [(offset + hops_per_grid_note / 2 + i * hops_per_grid_note) for i in range(n_total_notes)]
    boundaries = [0] + boundaries

    y = y / np.max(y, axis=1)[:, np.newaxis]  # Normalize

    grid = np.zeros((y.shape[0], n_total_notes), dtype=np.bool)

    def peaks_to_grid_pos(peaks):
        return np.array(
            [(int(n_total_notes + (p - offset + hops_per_grid_note / 2) / hops_per_grid_note)) % n_total_notes
             for p in peaks])

    for row_idx in range(y.shape[0]):
        if row_idx < 2:
            pre_max, post_max = 3, 3
            pre_avg, post_avg = 20, 20
            delta = 0.30
            wait = 7
        else:
            pre_max, post_max = 2, 2
            pre_avg, post_avg = 7, 7
            delta = 0.1
            wait = 2

        peaks = librosa.util.peak_pick(y[row_idx], pre_max, post_max, pre_avg, post_avg, delta, wait)
        grid[row_idx, peaks_to_grid_pos(peaks)] = 1

        # pre_avg, post_avg = hops_per_grid_note * (grid_note * 4), hops_per_grid_note * (grid_note * 4)
        # peaks2 = librosa.util.peak_pick(y[row_idx], pre_max, post_max, pre_avg, post_avg, delta, wait)

    return grid, boundaries


def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )