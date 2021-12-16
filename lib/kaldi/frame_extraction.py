#!/usr/bin/env python3

# Functions in this script define some utility dsp functions that mimic the
# behaviour in Kaldi. Primarily used in testing the DSP layers implemented
# in tensorflow.

import numpy as np


def MirrorPad(x: np.ndarray, pad: int) -> np.ndarray:
    """
    Pads the input array on either side of the last axis by the
    number of padding samples specified. Padding is done by mirroring
    the samples at the edge; the edge element is replicated in the
    padding.

    Parameters
    ----------
    x : np.ndarray
        1D array containing samples.
    pad : int
        Number of samples to pad with on either side.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    leftPadding = np.flip(x[..., :pad + 1], axis=-1)  # [p, p-1, ..., 2, 1 , 0]
    rightPadding = np.flip(x[..., -pad:], axis=-1)    # [N, N-1, ..., N-p-2, N-p-1 , N-p]
    return np.concatenate([leftPadding, x, rightPadding], axis=-1)


def ExtractFrames(samples: np.ndarray,
                  frameSizeMs: float,
                  frameShiftMs: float,
                  sampleFreq: float,
                  snipEdges: bool) -> np.ndarray:
    """
    This implements the way Kaldi does framing of audio samples. It is 
    assumed that for snipEdges=False, the input samples have already been
    padded if required using `MirrorPad()`.

    # Adapted from `_get_strided()` function defined here:
    # https://pytorch.org/audio/stable/_modules/torchaudio/compliance/kaldi.html

    Parameters
    ----------
    samples : np.ndarray
        1D array contanining samples to frame.
    frameSizeMs : float
        Frame length in milliseconds.
    frameShiftMs : float
        Frame shift in milliseconds.
    sampleFreq : float
        Sampling frequency in hertz.
    snipEdges : bool
        If true, will only output frames for center frames where the frame
        is completely within the sample array.

    Returns
    -------
    np.ndarray
        2D array containing frames of samples.
    """
    m = int(sampleFreq * frameSizeMs / 1000.0)  # Frame size as # of samples.
    k = int(sampleFreq * frameShiftMs / 1000.0)  # Frame shift as # of samples.
    N = samples.shape[-1]   # Total number of samples.

    M = (N + (k // 2)) // k  # Number of frames.
    if snipEdges:
        M = 1 + ((N - m) // k)
        N = (M - 1) * k + m

    x = samples[:N]

    shape = x.shape[:-1] + (N - m + 1, m)
    strides = x.strides + (x.strides[-1],)

    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)[::k]
