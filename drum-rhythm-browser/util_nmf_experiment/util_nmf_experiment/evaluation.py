import numpy as np
import os
from . import odf_op
from . import files


def get_onsets_for_instrument(directory, song, instrument_name):

    simple_name = files.filename_to_compact_string(song)
    ground_truth_file = os.path.join(directory, f'{simple_name}.{instrument_name}.csv')
    onsets = [float(l[0]) for l in files.read_csv_file(ground_truth_file, skip_header=False)]

    return onsets


def onsets_to_dummy_odf(onsets, sr, hopsize, odf_length_hops = None,
                        audio_length_samples=None, kernel_type='exponential', kernel_length_ms=150):
    ''' Convert a list of onset instants to a percussive ODF, by convolving the time instants with an onset kernel.

    :param onsets: sorted list of onset instants in seconds.
    :param sr: audio sample rate
    :param hopsize: number of audio samples between successive onset function values.
    :param kernel_type: onset kernel used to generate peaks at the provided onset instances.
        ('exponential' for an exponentially decaying onset with)

    :return: onset detection function, as if calculated using the specified hop size on audio with sample rate sr.
    '''

    if audio_length_samples is None and odf_length_hops is None:
        raise Exception('Please provide the length in no. samples of the audio that is being annotated.')

    kernel_length_frames = int(kernel_length_ms / (1000 * hopsize / sr))
    kernel_head_length_frames = kernel_length_frames // 5
    hops_per_second = sr / hopsize

    if kernel_type == 'exponential':
        kernel_tail = np.exp(-np.linspace(0,6,kernel_length_frames))
        kernel_head = np.linspace(0,1,kernel_head_length_frames, endpoint=False)
        kernel = np.concatenate((kernel_head, kernel_tail))
    else:
        raise Exception('Unknown dummy ODF kernel type.')

    if odf_length_hops is None:
        odf_length_hops = int(audio_length_samples / hopsize)
    odf = np.zeros(odf_length_hops+1)

    for t in onsets:
        n = int(t * hops_per_second)
        odf[n] = 1

    odf = np.convolve(odf, kernel, mode='full')

    odf = odf[kernel_head_length_frames:]
    if odf_length_hops is not None:
        if odf_length_hops > len(odf):
            odf = np.concatenate((odf, np.zeros(odf_length_hops - len(odf))))
        else:
            odf = odf[:odf_length_hops]

    return odf


def borda_ranking(*rankings):
    '''Calculate the Borda count ordering, given the rankings of each individual.

    All rankings should be arrays of indices in descending order of preference, and of equal length.
    '''
    N = len(rankings[0])

    points_template = np.array(range(N-1,-1,-1))
    points_accumulated = np.zeros(N)
    for r in rankings:
        points_accumulated[r] += points_template
    _, final_ranking = odf_op.select_top_k(points_accumulated, len(points_accumulated))
    return final_ranking


def rank_by_kick_and_snare(M_kick, M_snare, song_idx, L):

    _, ranking_kick = odf_op.select_top_k(M_kick[song_idx, :], L)
    _, ranking_snare = odf_op.select_top_k(M_snare[song_idx, :], L)

    ranking = borda_ranking(ranking_kick, ranking_snare)

    return ranking


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.plot(onsets_to_dummy_odf([0,1,2.5,3], 44100, 512, odf_length=2.75))
    plt.show()