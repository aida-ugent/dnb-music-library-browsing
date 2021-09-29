from autodj.annotation.style.theme_descriptor import ThemeDescriptorEstimator
from autodj.dj.annotators import wrappers

import numpy as np


class NonstructuralThemeDescriptorWrapper(wrappers.BaseAnnotationWrapper):
    '''A simplified version of the Theme Descriptor feature, which doesn't require beat or downbeat tracking 
    or structural segmentation.'''

    def __init__(self):
        super(NonstructuralThemeDescriptorWrapper, self).__init__()
        self.theme_annotator = ThemeDescriptorEstimator()

    def process(self, song):

        L = len(song.audio)
        # Get a heuristical slice of the audio.
        # This assumes that the drop will be approximately between 20% and 50% of the song,
        # and that the audio content in that slice will be representative for the whole song.
        start_sample = int(L * 0.2)
        end_sample = int(L * 0.5)

        song_theme_descriptor = self.theme_annotator(song.audio, [(start_sample, end_sample)])
        return {'simple_song_theme_descriptor' : song_theme_descriptor.tolist()}

    def is_annotated_in(self, song):
        return hasattr(song, 'simple_song_theme_descriptor')
        # return song.hasAnnot(annot_util.ANNOT_THEME_DESCR_PREFIX)

    def calculate_supplimentary_features(self, song):
        # When loaded from JSON, it is a normal array, not a numpy array.
        return {'simple_song_theme_descriptor' : np.array(song.simple_song_theme_descriptor)}