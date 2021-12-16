#!/usr/bin/env python3

# Functions in this script define some utility dsp functions that mimic the
# behavior in Kaldi. Primarily used in testing the DSP layers implemented
# in tensorflow.

from typing import Tuple

import numpy as np


def MirrorPad(x: np.ndarray, left_pad: int, right_pad: int) -> np.ndarray:
    """
    Pads the input array on either side of the last axis by the
    number of padding samples specified. Padding is done by mirroring
    the samples at the edge; the edge element is replicated in the
    padding.

    Parameters
    ----------
    x : np.ndarray
        1D array containing samples.
    left_pad : int
        Number of samples to mirror for padding on the left side.
    right_pad : int
        Number of samples to mirror for padding on the right side.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    leftPadding = np.flip(x[..., :left_pad], axis=-1)  # [p-1, ..., 2, 1 , 0]
    rightPadding = np.flip(x[..., -right_pad:], axis=-1)    # [N, N-1, ..., N-p-2, N-p-1 , N-p]
    return np.concatenate([leftPadding, x, rightPadding], axis=-1)


def PadWaveform(x: np.ndarray, frameSize: int, frameShift: int) -> np.ndarray:
    """
    Pads given waveform x by mirroring samples at the boundaries,
    such that frames of the given size and shift can be taken
    containing the boundary values.

    Parameters
    ----------
    x : np.ndarray
        Waveform array.
    frameSize : int
        Frame size as number of samples.
    frameShift : int
        Frame shift as number of samples.

    Returns
    -------
    np.ndarray
        Padded waveform.
    """
    # TODO (shahruk): simplify this

    # N = total number of samples.
    N = x.shape[-1]

    # M = number of frames; adding half the frameShift before dividing, to have
    # the effect of rounding towards the closest integer. 
    M = (N + (frameShift // 2)) // frameShift

    Nv = (M - 1) * frameShift + frameSize
    leftOver = abs(N - Nv)

    leftPad = (frameSize - frameShift) // 2
    rightPad = leftOver - leftPad

    return MirrorPad(x, leftPad, rightPad)


def ExtractFrames(samples: np.ndarray,
                  frameSizeMs: float,
                  frameShiftMs: float,
                  sampleFreq: float,
                  snipEdges: bool) -> np.ndarray:
    """
    This implements the way Kaldi does framing of audio samples. It is 
    assumed that for snipEdges=False, the input samples have already been
    padded if required using `MirrorPad()`.

    Adapted from `_get_strided()` function defined here:
    https://pytorch.org/audio/stable/_modules/torchaudio/compliance/kaldi.html

    Parameters
    ----------
    samples : np.ndarray
        1D array containing samples to frame.
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


def GetWindowFunction(window_type: str, window_size: int) -> np.ndarray:
    """
    Computes the window function of the given type for the
    given size.

    Parameters
    ----------
    window_type : str
        Type of window [ hamming | hanning | rectangular | blackmann | povey | sine ]
    window_size : int
        Size of the window as number of samples.

    Returns
    -------
    np.ndarray
        Window function of length window_size.

    Raises
    ------
    ValueError
        If window_size == 0.
        If window_type is invalid.
    """
    if window_size == 0:
        raise ValueError("window_size must be > 0")

    if window_type == "hanning":
        return np.hanning(window_size)

    if window_type == "hamming":
        return np.hamming(window_size)

    if window_type == "rectangular":
        return np.ones(window_size)

    if window_type == "blackman":
        return np.blackman(window_size)

    M = window_size
    n = np.arange(0, M)

    if window_type == "povey":
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1)))**0.85

    if window_type == "sine":
        return np.sin(np.pi * n / (M - 1))

    raise ValueError(f"invalid window type {window_type}")


def ProcessFrames(frames: np.ndarray,
                  dither: float = 0.0,
                  remove_dc_offset: bool = True,
                  preemphasis_coefficient: float = 0.97,
                  window_type: str = "povey",
                  raw_energy: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Processes extracted frames by applying configured window functions and other
    transformations such as pre-emphasis and removal of DC offset. Returns the
    processed windows, and log energy of those windows.

    Parameters
    ----------
    frames : np.ndarray
        Numpy array containing frames, shape = (..., frames, samples)
    dither : float, optional
        Dithering constant, by default 0.0 (disabled)
    remove_dc_offset : bool, optional
        Subtract mean from waveform on each frame, by default True
    preemphasis_coefficient : float, optional
        Coefficient for use in signal preemphasis, by default 0.97
    window_type : str, optional
        Type of window [ hamming | hanning | povey | rectangular | sine | blackmann ],
        by default "povey"
    raw_energy : bool, optional
        If true, compute energy before preemphasis and windowing, by default True

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Processed frames and log energy.

    Raises
    ------
    ValueError
        If preemphasis_coefficient not between 0 and 1.
        if window_type is invalid.
    """
    if preemphasis_coefficient < 0 or preemphasis_coefficient > 1:
        raise ValueError("preemphasis coefficient must be between 0 and 1")

    # Getting window function adding axes so that it may be broadcasted when
    # multiplying with the frames array.
    windowSize = frames.shape[-1]
    windowFunc = GetWindowFunction(window_type, windowSize)
    rank = len(frames.shape)
    windowFuncShape = [1] * (rank - 1) + [-1]
    windowFunc = windowFunc.reshape(windowFuncShape)

    # Small constant added to energies to prevent log(0).
    eps = np.finfo(frames.dtype).eps

    windows = frames.copy()

    if dither != 0.0:
        ditherAmount = np.random.normal(size=windows.shape) * dither
        windows += ditherAmount.astype(windows.dtype)

    if remove_dc_offset:
        means = np.mean(windows, axis=-1, keepdims=True)
        windows = windows - means

    if raw_energy:
        energy = np.sum(np.power(windows, 2), axis=-1, keepdims=True).clip(min=eps)

    if preemphasis_coefficient > 0.0:
        windows[..., 1:] -= preemphasis_coefficient * windows[..., :-1]
        windows[..., 0] -= preemphasis_coefficient * windows[..., 0]

    windows = windows * windowFunc

    if not raw_energy:
        energy = np.sum(np.power(windows, 2), axis=-1, keepdims=True).clip(min=eps)

    return windows, np.log(energy)
