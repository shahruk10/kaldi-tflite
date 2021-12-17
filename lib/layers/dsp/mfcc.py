#!/usr/bin/env python3

# Copyright (2021-) Shahruk Hossain <shahruk10@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import Union, Iterable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from lib.layers import Windowing, FilterBank, DCT


class MFCC(Layer):

    """
    This layer implements MFCC extraction. The output of this layer is compliant
    with Kaldi, producing the same output has Kaldi's `compute-mfcc-feats`
    binary. There might be some differences due to math libaries and dithering
    options, but they should pretty close. This layer does not support VLTN
    warping yet, but that isn't enabled in many Kaldi recipes anyway.

    The layer expects a 3D tensor of shape (batch, frames, samples) and applies
    the filter bank on each of the frames. It is expected that the frames have
    *not* been passed through a window function or the `Windowing` layer, since
    this layer does that internally. 
    """

    def __init__(self,
                 num_mfccs: int = 23,
                 num_mels: int = 23,
                 cepstral_lifter: int = 22,
                 use_energy: bool = True,
                 sample_frequency: float = 16000.0,
                 high_freq_cutoff: float = 0.0,
                 low_freq_cutoff: float = 20.0,
                 use_log_fbank: bool = True,
                 use_power: bool = True,
                 window_type: str = "povey",
                 dither: float = 0.0,
                 remove_dc_offset: bool = True,
                 preemphasis_coefficient: float = 0.97,
                 raw_energy: bool = True,
                 energy_floor: float = 0.0,
                 epsilon: float = 1e-7,
                 name: str = None,
                 **kwargs):
        """
        Initializes MFCC layer with given configuration.

        Parameters
        ----------
        num_mfccs : int, optional
            Number of cepstra in MFCC computation (including C0), by default 23
        num_mels : int, optional
            Number of triangular mel-frequency bins, by default 23
        cepstral_lifter : int, optional
            Constant that controls scaling of MFCCs, by default 22
        use_energy : bool, optional
            If true, use energy (not C0) in MFCC computation, by default True
        sample_frequency : float, optional
            Audio sampling frequency in hertz, by default 16000.0
        high_freq_cutoff : float, optional
            High cutoff frequency for mel bins (if <= 0, offset from Nyquist),
            by default 0.0
        low_freq_cutoff : float, optional
            Low cutoff frequency for mel bins, by default 20.0
        use_log_fbank : bool, optional
            If true, produce log-filterbank, else produce linear, by default True
        use_power : bool, optional
            If true, use power spectrum, else use magnitude spectrum, by default True
        window_type : str, optional
            Type of window to apply on input frames; should be one of:
            [ hamming | hanning | povey | rectangular | sine | blackmann ],
            by default "povey"
        dither : float, optional
            Dithering constant, by default 0.0 (disabled)
        remove_dc_offset : bool, optional
            If true, subtract mean from waveform on each frame, by default True
        preemphasis_coefficient : float, optional
            Coefficient for use in signal preemphasis, by default 0.97
        raw_energy : bool, optional
            If true, compute energy before preemphasis and windowing, by default True
        energy_floor : float, optional
            Floor on energy (absolute, not relative) in MFCC computation, by default 0.0
        epsilon : float, optional
            Small constant added to energies to prevent taking log of 0, by default 1e-7
        name : str, optional
            Name of the given layer. If auto set if set to None.
            By default None.

        Raises
        ------
        ValueError
            If num_mfccs > num_mels.
        """
        super(MFCC, self).__init__(trainable=False, name=name)

        self.numMffcs = num_mfccs
        self.melBins = num_mels
        self.cepstralLifter = cepstral_lifter
        self.useEnergy = use_energy

        if self.numMffcs > self.melBins:
            raise ValueError("num_mfccs must be <= num_mels")

        self.eps = tf.constant(epsilon, dtype=self.dtype)

        # Inputs to this layers are expected to be in the shape
        # (batch, frames, samples)
        self.sampleAxis = -1

        # Instantiating DSP layers.
        self.precomputeLifterCoeffs()

        self.windowing = Windowing(
            window_type=window_type, dither=dither, remove_dc_offset=remove_dc_offset,
            preemphasis_coefficient=preemphasis_coefficient, raw_energy=raw_energy,
            return_energy=use_energy, energy_floor=energy_floor, epsilon=epsilon,
        )

        self.filterbank = FilterBank(
            num_bins=num_mels, sample_frequency=sample_frequency,
            high_freq_cutoff=high_freq_cutoff, low_freq_cutoff=low_freq_cutoff,
            use_log_fbank=use_log_fbank, use_power=use_power, epsilon=epsilon,
        )

        self.dct = DCT(length=num_mfccs, dct_type=2, norm="ortho")

    def precomputeLifterCoeffs(self):
        """
        Precomputes the liftering coefficients.
        """
        M = self.numMffcs
        if M <= 1:
            return

        q = self.cepstralLifter
        n = np.arange(0, M)
        l = 1 + 0.5 * np.sin(np.pi * n / q) * q

        self.lifters = tf.constant(l, dtype=self.dtype)
        self.lifters = tf.reshape(self.lifters, [1, 1, M])

    def compute_output_shape(self, input_shape: Iterable[Union[int, None]]) -> Tuple[Union[int, None]]:
        """
        Returns the shape of the MFCC output, given the shape of the input.

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
        outputShape[self.sampleAxis] = self.numMffcs

        return outputShape

    def get_config(self) -> dict:
        config = super(MFCC, self).get_config()

        config.update(self.windowing.get_config())
        config.update(self.filterbank.get_config())

        config.update({
            "num_mfccs": self.numMffcs,
            "num_mels": self.melBins,
            "cepstral_lifter": self.cepstralLifter,
            "use_energy": self.useEnergy,
            "epsilon": self.eps.numpy(),
        })

        return config

    def call(self, inputs):

        if self.useEnergy:
            frames, energy = self.windowing(inputs)
        else:
            frames = self.windowing(inputs)

        fbank = self.filterbank(frames)
        mfcc = self.dct(fbank)

        if self.cepstralLifter > 1:
            mfcc = mfcc * self.lifters

        if self.useEnergy:
            # Replacing C0 with log of frame energies; Flattening out mfcc and
            # energy tensors and then replacing every (numMfccs)th value in the
            # mfcc tensor (position of C0) with the energy. Reshaping doesn't
            # reallocate anything, so this should be memory efficient.
            orgShape = tf.shape(mfcc)
            mfcc = tf.reshape(mfcc, [-1, 1])
            energy = tf.reshape(energy, [-1, 1])
            zeroIdx = tf.expand_dims(tf.range(0, tf.size(mfcc), self.numMffcs), 1)

            # Current shapes (N = numFrames, B = batch, M = numMfccs):
            #   mfcc = (N*B*M, 1).
            #   energy = (N, 1)
            #   zeroIdx = (N, 1)
            mfcc = tf.tensor_scatter_nd_update(mfcc, zeroIdx, energy)

            # Reshaping back to orginal shape (B, N, M).
            mfcc = tf.reshape(mfcc, orgShape)

        return mfcc
