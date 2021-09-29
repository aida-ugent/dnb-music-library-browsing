'''Utility module bundling different flavours of NMF methods.'''

import NMFtoolbox as toolbox
import NMFtoolbox.NMFD
import NMFtoolbox.initTemplates
import NMFtoolbox.initActivations
import NMFtoolbox.alphaWienerFilter
import NMFtoolbox.rhythmicSoftConstraintsNMF

import numpy as np

from . import spectrogram_op


def apply_nmfd_fix_template(*args, **kwargs):

    return apply_nmfd(*args,
                      fixW=True,
                      force_repetition_after_n_beats=4,
                      **kwargs)


apply_nmfd_fix_template.spectrogram_fn = spectrogram_op.stft
apply_nmfd_fix_template.inverse_spectrogram_fn = spectrogram_op.istft


def apply_nmfd_fix_template_edm_stft(*args, **kwargs):
    return apply_nmfd(
        *args, fixW=True,
        force_repetition_after_n_beats=4,
        drum_template_init_strategy='drums-custom',
        drum_template_folder='FILL-OUT-TEMPLATE-FOLDER-HERE/spectrum_templates/stft',
        **kwargs,
    )


apply_nmfd_fix_template_edm_stft.spectrogram_fn = spectrogram_op.stft
apply_nmfd_fix_template_edm_stft.inverse_spectrogram_fn = spectrogram_op.istft


def apply_nmfd_fix_template_edm_cqt_norepetition(*args, fixW=True, **kwargs):
    return apply_nmfd(
        *args, fixW=fixW,
        drum_template_init_strategy='drums-custom',
        drum_template_folder='FILL-OUT-TEMPLATE-FOLDER-HERE/spectrum_templates/cqt',
        **kwargs,
    )


apply_nmfd_fix_template_edm_cqt_norepetition.spectrogram_fn = spectrogram_op.cqt
apply_nmfd_fix_template_edm_cqt_norepetition.inverse_spectrogram_fn = spectrogram_op.icqt


def apply_nmfd_fix_template_edm_cqt(*args, **kwargs):
    return apply_nmfd(
        *args, fixW=True,
        force_repetition_after_n_beats=4,
        drum_template_init_strategy='drums-custom',
        drum_template_folder='FILL-OUT-TEMPLATE-FOLDER-HERE/spectrum_templates/cqt',
        **kwargs,
    )


apply_nmfd_fix_template_edm_cqt.spectrogram_fn = spectrogram_op.cqt
apply_nmfd_fix_template_edm_cqt.inverse_spectrogram_fn = spectrogram_op.icqt


def apply_nmfd_edm_cqt(*args, **kwargs):
    return apply_nmfd(
        *args, fixW=False,
        force_repetition_after_n_beats=4,
        drum_template_init_strategy='drums-custom',
        drum_template_folder='FILL-OUT-TEMPLATE-FOLDER-HERE/spectrum_templates/cqt',
        **kwargs,
    )


apply_nmfd_edm_cqt.spectrogram_fn = spectrogram_op.cqt
apply_nmfd_edm_cqt.inverse_spectrogram_fn = spectrogram_op.icqt


def apply_nmfd_fix_template_edm_cqt_cascaded(X, n_components, *args, **kwargs):

    # Calculate a first approximation of the NMF decomposition
    nmfdA, nmfdW, nmfdH, _ = apply_nmfd_fix_template_edm_cqt(X, n_components, *args, **kwargs)

    # Apply an NMF decomposition on each component spectrogram (kick, snare, HH).
    # Merge all components
    for k in range(n_components):

        nmfdA_k, nmfdW_k, nmfdH_k, _ = apply_nmfd_fix_template_edm_cqt(nmfdA[k], n_components, *args, **kwargs)

        for l in range(n_components):
            nmfdW[l] += nmfdW_k[l]

    # Renormalize templates
    for k in range(n_components):
        nmfdW[k] = nmfdW[k] / nmfdW[k].sum()

    # Calculate final ODFs
    return apply_nmfd(X, n_components, *args, fixW=True, initW = nmfdW, **kwargs,)


apply_nmfd_fix_template_edm_cqt_cascaded.spectrogram_fn = spectrogram_op.cqt
apply_nmfd_fix_template_edm_cqt_cascaded.inverse_spectrogram_fn = spectrogram_op.icqt


def apply_nmfd(X, n_components,
               sr=44100, n_fft=2048, hop_size=512,
               force_repetition_after_n_beats=None, tempo=None,
               n_template_frames=7, n_iter=75,
               update_rule_template_width=None,
               drum_template_init_strategy='drums', drum_template_folder=None,
               fixW=False, initW=None,
               fixH=False, initH=None,
               regularization_strength=None, regularization_alpha=None,
               paramNMFD = None,
               nmfd_implementation = toolbox.NMFD.NMFD,
               ):

    # get dimensions and time and freq resolutions
    n_bins, n_frames = X.shape
    deltaT = hop_size / sr
    deltaF = sr / n_fft

    # set common parameters
    n_components = n_components

    # generate initial guess for templates
    if initW is None:
        print(f'Initializing initW with strategy "{drum_template_init_strategy}".')
        paramTemplates = dict()
        paramTemplates['deltaF'] = deltaF
        paramTemplates['numComp'] = n_components
        paramTemplates['numBins'] = n_bins
        paramTemplates['numTemplateFrames'] = n_template_frames
        paramTemplates['drumsTemplateFolder'] = drum_template_folder
        initW = toolbox.initTemplates.initTemplates(paramTemplates, drum_template_init_strategy)

    # generate initial activations
    if initH is None:
        print('Initializing initH with strategy "uniform".')
        paramActivations = dict()
        paramActivations['numComp'] = n_components
        paramActivations['numFrames'] = n_frames
        initH = toolbox.initActivations.initActivations(paramActivations, 'uniform')

    # NMFD parameters
    paramNMFD = paramNMFD if paramNMFD is not None else dict()
    paramNMFD['numComp'] = n_components
    paramNMFD['numFrames'] = n_frames
    paramNMFD['numIter'] = n_iter
    paramNMFD['numTemplateFrames'] = n_template_frames
    paramNMFD['updateRuleTemplateFrames'] = update_rule_template_width
    paramNMFD['initW'] = initW
    paramNMFD['initH'] = initH
    paramNMFD['fixW'] = fixW
    paramNMFD['fixH'] = fixH

    paramNMFD['regularization_strength'] = regularization_strength
    paramNMFD['regularization_alpha'] = regularization_alpha

    paramConstr = None

    # NMFD core method
    nmfdW, nmfdH, nmfdV, divKL, _ = nmfd_implementation(X, paramNMFD, paramConstr)

    # alpha-Wiener filtering
    # nmfdA, _ = toolbox.alphaWienerFilter.alphaWienerFilter(X, nmfdV, 1.0)

    # return nmfdA, nmfdW, nmfdH, divKL
    return nmfdV, nmfdW, nmfdH, divKL


apply_nmfd.spectrogram_fn = spectrogram_op.stft
apply_nmfd.inverse_spectrogram_fn = spectrogram_op.istft