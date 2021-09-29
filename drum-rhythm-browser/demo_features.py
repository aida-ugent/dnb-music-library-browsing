import numpy as np

def metric_strength_of_position(i, N):  # N = max length of a pattern
    K = int(np.ceil(np.log2(N)))
    for j in range(K + 1):
        if i % 2 ** j != 0:
            return (j - 1)
    return K


def calculate_sync_pattern(onsets, grid_note=16):
    if onsets.shape[0] % grid_note != 0:
        raise Exception('The amount of notes must together form a multiple number of bars.', onsets.shape)

    syncopation_values = np.zeros_like(onsets)

    # To calculate the syncopation of last note, use the first note as a reference
    metric_strengths = np.array([metric_strength_of_position(i, grid_note) for i in range(len(onsets))])

    for i in range(1, len(onsets)):  # Ignore first beat, this is never a syncopation
        # Next note w/ higher rhythmic position
        next_i_same_strength = (i + 2 ** metric_strengths[i]) % len(onsets)
        if onsets[i] and not onsets[next_i_same_strength]:
            syncopation_values[i] = metric_strengths[next_i_same_strength] - metric_strengths[i]

    per_bar_syncopation_sum = [sum(syncopation_values[i * grid_note:(i + 1) * grid_note])
                               for i in range(int(np.ceil(len(syncopation_values) / grid_note)))]
    return np.array(per_bar_syncopation_sum)


def low_sync_pattern(song):
    return calculate_sync_pattern(song.drum_grid[0, :])


def mid_sync_pattern(song):
    return calculate_sync_pattern(song.drum_grid[1, :])


def hi_sync_pattern(song):
    return calculate_sync_pattern(song.drum_grid[2, :])


def low_density(song):
    return np.sum(song.drum_grid[0, :]) / song.drum_grid.shape[1]


def mid_density(song):
    return np.sum(song.drum_grid[1, :]) / song.drum_grid.shape[1]


def hi_density(song):
    return np.sum(song.drum_grid[2, :]) / song.drum_grid.shape[1]


def average_onset_pattern(song):
    grid = song.drum_grid
    N = grid.shape[1] // 4
    return np.average([grid[i*N:(i+1)*N] for i in range(4)])


def average_onset_pattern_kick(song):
    return average_onset_pattern(song)[0].flatten()


def average_onset_pattern_snare(song):
    return average_onset_pattern(song)[1].flatten()


def average_onset_pattern_hihat(song):
    return average_onset_pattern(song)[2].flatten()