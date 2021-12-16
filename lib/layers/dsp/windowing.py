#!/usr/bin/env python3

from typing import Union, Iterable

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class Windowing(Layer):

    """
    This layer implements a windowing function that takes a 3D tensor (batch,
    frames, samples) containing frames of something like audio samples, and
    applies the configured window function, and other optional transformations,
    such as pre-emphasis, dithering and removal of DC offset. 
    """

    def __init__(self,
                 window_type: str = "povey",
                 blackman_coeff: float = 0.42,
                 dither: float = 0.0,
                 remove_dc_offset: bool = True,
                 preemphasis_coefficient: float = 0.97,
                 return_energy: bool = True,
                 raw_energy: bool = True,
                 energy_floor: float = 0.0,
                 epsilon: float = 1e-7,
                 name: str = None,
                 **kwargs):
        """
        Initializes Windowing layer with given configuration.

        Parameters
        ----------
        window_type : str, optional
            Type of window [ hamming | hanning | povey | rectangular | sine | blackmann ],
            by default "povey"
        blackman_coeff : float, optional
            Constant coefficient for generalized Blackman window, by default 0.42
        dither : float, optional
            Dithering constant, by default 0.0 (disabled)
        remove_dc_offset : bool, optional
            Subtract mean from waveform on each frame, by default True
        preemphasis_coefficient : float, optional
            Coefficient for use in signal preemphasis, by default 0.97
        return_energy : bool, optional
            If true, return the log of energy of each window along with the windows,
            by default True
        raw_energy : bool, optional
            If true, compute energy before preemphasis and windowing, by default True
        energy_floor : float, optional,
            Floor on computed log of energy. Recommended to be 0.1 or 1.0 if dithering
            is disabled, by default 0.0 
        epsilon : float, optional
            Small constant added to energies to prevent taking log of 0, by default 1e-7
        name : str, optional
            Name of the given layer. If auto set if set to None.
            By default None.

        Raises
        ------
        ValueError
            If preemphasis_coefficient is not within 0.0 to 1.0.
            If window type is invalid.
        """
        super(Windowing, self).__init__(trainable=False, name=name, **kwargs)

        self.preemphasisCoeff = preemphasis_coefficient
        if self.preemphasisCoeff < 0 or self.preemphasisCoeff > 1.0:
            raise ValueError("preemphasis_coefficient should be between 0.0 and 1.0")

        self.windowType = window_type.lower()
        if self.windowType not in ["hamming", "hanning", "povey", "rectangular", "sine", "blackman"]:
            raise ValueError(f"window_type '{window_type}' is not recognized")

        self.blackmanCoeff = blackman_coeff

        self.dither = dither
        self.removeDCOffset = remove_dc_offset
        self.returnEnergy = return_energy
        self.rawEnergy = raw_energy
        self.energyFloor = energy_floor
        self.eps = tf.constant(epsilon, dtype=self.dtype)

        # The the window function is computed in build().
        self.windowFunc = None

        # Inputs to this layers are expected to be in the shape
        # (batch, frames, samples)
        self.sampleAxis = -1

    def build(self, input_shape: Iterable[Union[int, None]]):
        """
        Precomputes the window function that will be applied to input frames.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have three axes,
            (batch, frames, samples or feats).

        Raises
        ------
        ValueError
            If input shape implies a window size of 0.
            If configured window type is not recognized.
        """
        super(Windowing, self).build(input_shape)

        M = input_shape[self.sampleAxis]  # M = window size
        if M == 0:
            raise ValueError(
                f"window size (input shape axis = {self.sampleAxis}) needs to be > 0"
            )

        n = np.arange(0, M)

        if M == 1:
            w = np.ones(1, float)
        elif self.windowType == "hamming":
            w = np.hamming(M)
        elif self.windowType == "hanning":
            w = np.hanning(M)
        elif self.windowType == "povey":
            w = (0.5 - 0.5 * np.cos(2.0 * np.pi * n / (M - 1)))**0.85
        elif self.windowType == "rectangular":
            w = np.ones((M,))
        elif self.windowType == "sine":
            w = np.sin(np.pi * n / (M - 1))
        elif self.windowType == "blackman":
            w = np.blackman(M)
            if self.blackmanCoeff != 0.42:  # numpy default
                w = w - 0.42 + self.blackmanCoeff
        else:
            raise ValueError(f"window_type '{self.windowType}' is not recognized")

        self.windowFunc = tf.constant(w, dtype=self.dtype)
        self.windowFunc = tf.reshape(self.windowFunc, [1, 1, M])

    def get_config(self) -> dict:
        config = super(Windowing, self).get_config()
        config.update({
            "window_type": self.windowType,
            "blackman_coeff": self.blackmanCoeff,
            "dither": self.dither,
            "remove_dc_offset": self.removeDCOffset,
            "preemphasis_coefficient": self.preemphasisCoeff,
            "return_energy": self.returnEnergy,
            "raw_energy": self.rawEnergy,
            "energy_floor": self.energyFloor,
            "epsilon": self.eps.numpy(),
        })

        return config

    def computeLogEnergy(self, inputs):
        energy = tf.reduce_sum(tf.pow(inputs, 2), axis=self.sampleAxis, keepdims=True)
        energy = tf.math.log(tf.nn.relu(energy) + self.eps)
        energy = tf.clip_by_value(energy, self.energyFloor, energy.dtype.max)
        return energy

    def call(self, inputs):

        if self.dither != 0.0:
            inputs = inputs + tf.random.normal(shape=tf.shape(inputs)) * self.dither

        # Subtract mean of each frame from the frame samples.
        if self.removeDCOffset:
            # means.shape = (batch, numFrames, 1).
            means = tf.reduce_mean(inputs, axis=self.sampleAxis, keepdims=True)
            inputs = inputs - means

        # Raw energy => computed before applying pre-emphasis and window function.
        if self.returnEnergy and self.rawEnergy:
            energy = self.computeLogEnergy(inputs)

        if self.preemphasisCoeff > 0:
            # TODO: make this more efficient; use `tf.scatter_nd_update`
            inputs0, inputsRest = tf.split(inputs, [1, -1], self.sampleAxis)
            inputsRest = inputsRest - self.preemphasisCoeff * inputs[..., :-1]
            inputs0 = inputs0 - self.preemphasisCoeff * inputs0
            inputs = tf.concat([inputs0, inputsRest], self.sampleAxis)

        inputs = inputs * self.windowFunc

        if self.returnEnergy:
            if not self.rawEnergy:
                energy = self.computeLogEnergy(inputs)
            return inputs, energy

        return inputs
