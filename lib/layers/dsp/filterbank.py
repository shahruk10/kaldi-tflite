#!/usr/bin/env python3

from typing import Union, Iterable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class FilterBank(Layer):

    """
    This layer implements a Mel Frequency Filter Bank. The output of this layer
    is compliant with Kaldi, i.e. we should get the same output has Kaldi's
    `compute-fbank-feats` binary. However, this layer does not support VLTN
    warping yet, which isn't used in many recipes anyway.

    The layer expects a 3D tensor of shape (batch, frames, samples) and applies
    the filter bank on each of the frames.
    """

    def __init__(self,
                 num_bins: int = 23,
                 sample_frequency: float = 16000.0,
                 high_freq_cutoff: float = 0.0,
                 low_freq_cutoff: float = 20.0,
                 use_log_fbank: bool = True,
                 use_power: bool = True,
                 epsilon: float = 1e-7,
                 name: str = None,
                 **kwargs):
        """
        Initializes FilterBank layer with given configuration.

        Parameters
        ----------
        num_bins : int, optional
            Number of triangular mel-frequency bins, by default 23
        sample_frequency : float, optional
            Sampling frequency in hertz, by default 16000.0
        high_freq_cutoff : float, optional
             High cutoff frequency for mel bins (if <= 0, offset from Nyquist),
             by default 0.0
        low_freq_cutoff : float, optional
            Low cutoff frequency for mel bins, by default 20.0
        use_log_fbank : bool, optional
            If true, produce log-filterbank, else produce linear, by default True
        use_power : bool, optional
            If true, use power, else use magnitude, by default True
        epsilon : float, optional
            Small constant added to energies to prevent taking log of 0, by default 1e-7
        name : str, optional
            Name of the given layer. If auto set if set to None.
            By default None

        Raises
        ------
        ValueError
            If num_bins <= 2.
            If sample_frequency <= 0 or low_freq_cutoff > high_freq_cutoff.
        """
        super(FilterBank, self).__init__(trainable=False, name=name, **kwargs)

        self.numBins = num_bins
        if self.numBins <= 2:
            raise ValueError(f"num_bins must be >= 3, got {num_bins}")

        self.sampleFreq = sample_frequency
        self.nyquist = sample_frequency / 2.0
        if self.sampleFreq <= 0:
            raise ValueError(f"sample_frequency must be > 0, got {sample_frequency}")

        self.lowerCutoff = low_freq_cutoff
        if self.lowerCutoff > self.nyquist or self.lowerCutoff < 0:
            raise ValueError(f"low_freq_cutoff must be > 0 and < Nyquist Rate ({self.nyquist} Hz)")

        self.upperCutoff = high_freq_cutoff
        if self.upperCutoff <= 0:
            self.upperCutoff += self.nyquist

        if self.lowerCutoff >= self.upperCutoff:
            raise ValueError(f"lower_freq_cutoff must be < higher_freq_cutoff")

        self.useLogFBank = use_log_fbank
        self.usePower = use_power

        self.eps = tf.constant(epsilon, dtype=self.dtype)

        # The triangular filter bank in the mel scale and FFT length are
        # computed in build() after being given the length of input frames.
        self.melBank = None
        self.fftLength = None
        self.fftPadding = None

        # Inputs to this layers are expected to be in the shape
        # (batch, frames, samples)
        self.sampleAxis = -1

    def build(self, input_shape: Iterable[Union[int, None]]):
        """
        Precomputes Mel Filter Bank weight matrix based on the size of each
        frame obtained from the input_shape.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have three axes,
            (batch, frames, samples).
        """
        super(FilterBank, self).build(input_shape)

        inputRank = len(input_shape)
        windowSize = input_shape[self.sampleAxis]
        self.precomputeMelBank(windowSize, inputRank)

    def nextPowerOf2(self, n: int) -> int:
        if (n & (n - 1) == 0) and n != 0:
            return n
        return 2 ** (n - 1).bit_length()

    def getMelScale(self, freq: float) -> float:
        return 1127.0 * np.log(1.0 + freq / 700.0)

    def precomputeMelBank(self, windowSize: int, inputRank: int):
        """
        Precomputes the weight matrix for the triangular mel filter bank which
        will be multiplied with the spectogram frames computed from the inputs
        to this layer.

        Parameters
        ----------
        windowSize : int
            Length of each window / frame on which the filter bank will
            be applied.

        inputRank : int
            Rank of input tensor to this layer.
        """
        self.fftLength = self.nextPowerOf2(windowSize)

        # Computing amount of padding required to make window size equal to the
        # next power of 2 if it is not already a power of 2.
        fftPadding = self.fftLength - windowSize
        if fftPadding > 0:
            # Padding will only be done on the last axis.
            paddingAmount = [[0, 0] for i in range(inputRank)]
            paddingAmount[self.sampleAxis] = [0, fftPadding]
            self.fftPadding = tf.constant(paddingAmount, dtype=tf.int32)

        fftBins = self.fftLength // 2
        fftBinWidth = self.sampleFreq / self.fftLength

        melLow = self.getMelScale(self.lowerCutoff)
        melHigh = self.getMelScale(self.upperCutoff)
        melDelta = (melHigh - melLow) / (self.numBins + 1)

        # Setting up triangular filters.
        melBank = np.zeros([self.numBins, fftBins + 1], dtype=np.float32)
        for i in range(self.numBins):
            left = melLow + (i * melDelta)
            center = left + melDelta
            right = center + melDelta

            for j in range(fftBins):
                mel = self.getMelScale(fftBinWidth * j)
                if left < mel < right:
                    if mel <= center:
                        melBank[i, j] = (mel - left) / (center - left)
                    else:
                        melBank[i, j] = (right - mel) / (right - center)

        self.melBank = tf.constant(melBank.T, dtype=self.dtype)

    def compute_output_shape(self, input_shape: Iterable[Union[int, None]]) -> Tuple[Union[int, None]]:
        """
        Returns the shape of the filter bank output, given the shape of the input.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have three axes,
            (batch, frames, samples).

        Returns
        -------
        Tuple[Union[int, None]]
            Shape of the output of this layer.
        """
        outputShape = input_shape
        outputShape[self.sampleAxis] = self.numBins

        return outputShape

    def get_config(self) -> dict:
        config = super(FilterBank, self).get_config()
        config.update({
            "sample_frequency": self.sampleFreq,
            "num_bins": self.numBins,
            "lower_freq_cutoff": self.lowerCutoff,
            "upper_freq_cutoff": self.upperCutoff,
            "use_log_fbank": self.useLogFBank,
            "use_power": self.usePower,
            "epsilon": self.eps.numpy(),
        })

        return config

    def call(self, inputs):

        # If frame size is not a power of 2, we need to pad frames with zeros first.
        if self.fftPadding is not None:
            inputs = tf.pad(inputs, self.fftPadding, constant_values=0)

        # Computing spectrum using FFT; inputs.shape = (batch, frames, samples + padding).
        stfts = tf.signal.rfft(inputs)
        specs = tf.abs(stfts)
        if self.usePower:
            specs = tf.pow(specs, 2)

        # Computing Mel Filter Bank.
        feats = tf.matmul(specs, self.melBank)
        if self.useLogFBank:
            feats = tf.math.log(tf.nn.relu(feats) + self.eps)

        return feats
