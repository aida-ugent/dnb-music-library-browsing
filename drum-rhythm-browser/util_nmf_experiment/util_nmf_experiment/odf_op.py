''' Utility module with functions that perform operations on (one or multiple) onset functions.'''

import numpy as np
from scipy.interpolate import interp1d


def select_top_k(arr, k):
    idx = np.argpartition(arr, -k)[-k:]
    idx = idx[np.argsort(arr[idx])][::-1]
    return arr[idx], idx


def hwr(x, N=16):

    def adaptive_mean(x, N):
        return np.convolve(x, [1.0] * int(N), mode='same') / N

    x_mean = adaptive_mean(x, N)
    x_hwr = (x - x_mean).clip(min=0)
    return x_hwr


def norm2(x):
    return np.sqrt(np.sum(x**2))


def cosine_similarity(x, y):
    return np.sum(x*y) / (norm2(x) * norm2(y))


def geom_mean(*args):
    N = len(args)
    return np.exp(np.sum(np.log(x) for x in args) / N)


def set_array_length(x, L):
    if len(x) == L:
        return x
    # Stretch x to match length of y
    f_interp_x = interp1d(range(len(x)), x)
    return f_interp_x(np.linspace(0, len(x)-1, num=L, endpoint=True))


def match_length(x, y):
    x, y = np.copy(x), np.copy(y)
    if len(x) > len(y):
        x = set_array_length(x, len(y))
    elif len(x) < len(y):
        y = set_array_length(y, len(x))
    return x, y


def find_best_overlap(x, y, max_shift):
    corr = np.correlate(x, y, mode='full')
    zero_idx = len(x)-1
    corr = corr[zero_idx - max_shift : zero_idx + max_shift]
    return np.argmax(corr) - max_shift


def align_odfs(x, y, max_shift=None, n_shift = None, hold_y = False):
    # If hold_y is True, then y won't be shifted, only x
    # x will be wrap-padded

    if n_shift is None:
        n_shift = find_best_overlap(x, y, max_shift)

    if not hold_y:
        if n_shift >= 0:  # xi is shifted to the left wrt yi, so shift it right to compensate for this
            x, y = x[n_shift:], y[:-n_shift+len(y)]
        elif n_shift < 0:  # xi shifted to the right = xj shifted to the left
            x, y = x[:n_shift+len(x)], y[- n_shift:]
    else:
        if n_shift >= 0:
            x = np.pad(x[n_shift:], ((0, n_shift),), 'wrap')
        else:
            x = np.pad(x[:n_shift+len(x)], ((-n_shift, 0),), 'wrap')

    return x, y


def odf_similarity_matrix(odfs, measure='cosine'):

    N = len(odfs)

    corr = np.zeros((N, N))

    # Make a deep copy
    for i, x in enumerate(odfs):
        odfs[i] = np.copy(x)  # Deep copy
        # Windowing of beginning and ending
        mask = np.linspace(0, 1, 20)
        odfs[i][:20] *= mask
        odfs[i][-20:] *= mask[::-1]

        odfs[i] = odfs[i] ** (1 / 2)  # Accentuate peaks
        odfs[i] = odfs[i] / odfs[i].max()  # Normalization

    for i, xi in enumerate(odfs):
        for j, xj in enumerate(odfs):

            # Make sure ODFs are equally long
            xi_, xj_ = match_length(xi, xj)
            L = len(xi_)

            # Find best alignment (assume beat detection/odf accuracy is only approximately correct)
            # TODO this aligment must be per song maybe, not per odf!
            xi_, xj_ = align_odfs(xi_, xj_, 15)

            # cosine distance between u[k] = sqrt(xi[k]*xj[k]) and xi[k]
            # This is a measure for how much the common patterns between x and y are a part of x.
            # A high correlation means that when x is high, then it is likely that the common part
            # between x and y is high, or that x is a part of y to some extend
            if measure=='cosine':
                corr[i, j] = cosine_similarity(xi_, xj_)
            elif measure=='cosine-geom-mean':
                corr[i, j] = cosine_similarity(geom_mean(xi_, xj_), xi_)
            else:
                raise Exception(f'Unknown distance measure: {measure}')

    return corr