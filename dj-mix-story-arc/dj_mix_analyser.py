import argparse
import itertools
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
import sklearn.decomposition


def slicewise_spectrogram_density(S, frame_length, hop_length):
    
    result = []
    
    # Note: this operation can be parallellized (but it is not in this implementation)
    for t in range(0, int(S.shape[1]), int(hop_length)):
        S_slice = S[:, t:t+frame_length]  
        result.append(np.mean(S_slice))
        
    return np.array(result)


def extract_features(path_to_mix):
    
    SR = 44100
    hop_length_stft = 256
    n_fft = 512  # Pretty low resolution to keep computation time in check

    # Load the mix
    print('  Loading the mix...')
    y, sr = librosa.load(path_to_mix, sr=SR)
    y = y[::2]  # make the spectrogram smaller
    S_ = librosa.stft(y, n_fft=n_fft, hop_length=hop_length_stft)
    S_ /= np.max(np.abs(S_))  # Normalize
    
    # Perform HPSS
    print('  Performing HPSS on the mix...')
    H, P = librosa.decompose.hpss(S_, kernel_size=15, power=2.0, mask=False, margin=1.0)

    # Now process the harmonic part of the mix
    mix_frame_length = 10 * SR / hop_length_stft  # Analyse 10 seconds of audio at a time
    mix_hop_length = mix_frame_length // 2  # Jump over mix with hop size half of frame length
    
    print('  Processing harmonic part of the mix...')
    H_slicewise_density = slicewise_spectrogram_density(np.abs(H), int(mix_frame_length), int(mix_hop_length)) 
    print('  Processing percussive part of the mix...')
    P_slicewise_density = slicewise_spectrogram_density(np.abs(P), int(mix_frame_length), int(mix_hop_length))
    
    return {
        'harmonic_density' : H_slicewise_density,
        'percussive_density' : P_slicewise_density,
    }
    
# Another feature to experiment with, perhaps? :)
# S = np.abs(S_)
# spectral_contrast = librosa.feature.spectral_contrast(S=S)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Process some DJ mixes.')
    parser.add_argument('-i', '--input_dirs', required=True, nargs='+',
                        help='Path to folder(s) with DJ mixes to process.')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Directory to store the processed features in as .npz archives.')
    args = parser.parse_args()
    
    dirs = args.input_dirs
    dj_mixes = []
    for d in dirs:
        dj_mixes.extend([(f, os.path.join(d,f)) for f in sorted(os.listdir(d))])
    
    print(f'Analysing {len(dj_mixes)} DJ mixes in {len(dirs)} directories...')

    # Extract features
    for i, (dj_mix_title, dj_mix_path) in enumerate(dj_mixes):
        print(f'({i+1:2}/{len(dj_mixes)}) Processing {dj_mix_title}...')
        path_to_out_file = f'{args.output_dir}/{dj_mix_title}.npz'
        if os.path.exists(path_to_out_file):
            print('  File already analyzed...')
        else:
            features  = extract_features(dj_mix_path)
            np.savez(path_to_out_file, features)

    print('Done')
