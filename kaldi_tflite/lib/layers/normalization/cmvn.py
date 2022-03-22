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

import tensorflow as tf
from tensorflow.keras.layers import Layer


class CMVN(Layer):

    """
    This layer implements sliding-window cepstral mean (and optionally variance)
    normalization on input frames. The output of this layer is compliant with
    Kaldi, producing the same output has Kaldi's `apply-cmvn-sliding` binary.

    The layer expects a 3D tensor of shape (batch, frames, samples). If this
    layer is expected to process chunks of frames at a time instead of all the
    frame in a given utterance at once, the input to this layer should be padded
    with frames from the previous input and `padding` should be set to "VALID".
    """

    def __init__(self,
                 center: bool = True,
                 norm_vars: bool = False,
                 window: int = 600,
                 min_window: int = 100,
                 padding: str = "SAME",
                 name: str = None,
                 **kwargs):
        """
        Initializes CMVN layer with given configuration.

        Parameters
        ----------
        center : bool, optional
            If true, use a window centered on the current frame (to the extent possible,
            modulo end effects). If false, window is to the left. By default True.__float__
        norm_vars : bool, optional
            If true, normalize to 1.0, by default False.
        window : int, optional
            Number of frames in window for running average CMN computation, by default 600.
        min_window : int, optional
            Minimum CMN window used at start of decoding (adds latency only at start). Only
            applicable if center == false, ignored if center==true. By default 100.
        padding : str, optional
            One of ["SAME", "VALID"]. If padding == "SAME", the output array will have the same
            shape as `input` except that it will have one less frame (padding frame discarded).
            If padding == "VALID", only frames about which the centered window fits completely
            within the `frames` array is evaluated. The output number of frames in this case will
            be `numFrames - (2N-1) // 2` where `numFrames = input_shape[-2]` and `N = window size`.
            By default "SAME".
        name : str, optional
            Name of the given layer. Is auto set if set to None.
            By default None.

        Raises
        ------
        ValueError
            If padding is not "SAME" or "LOWER".
            If window or min_window <= 0.
        NotImplementedError
            if center = False.
        """
        super(CMVN, self).__init__(trainable=False, name=name)

        self.center = center
        self.normVar = norm_vars
        self.N = window
        self.minN = min_window

        if not self.center:
            raise NotImplementedError("CMVN with center=False not supported yet")

        if self.N <= 0 or self.minN <= 0:
            raise ValueError("`window` and `min_window` must be > 0")

        self.padding = padding.upper()
        if self.padding not in ["SAME", "VALID"]:
            raise ValueError(f"`padding` should be either 'SAME' or 'VALID', got '{padding}'")

        # Inputs to this layers are expected to be in the shape
        # (batch, frames, feats)
        self.batchAxis = 0
        self.frameAxis = -2
        self.featAxis = -1

    def compute_output_shape(self, input_shape: Iterable[Union[int, None]]) -> Tuple[Union[int, None]]:
        """
        Returns the shape of the output of this layer, given the shape of the input.

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
        if self.padding == "SAME":
            return input_shape

        outputShape = input_shape
        numFrames = input_shape[self.frameAxis]

        if numFrames is None:
            outputShape[self.frameAxis] = None
        elif numFrames <= self.N:
            outputShape[self.frameAxis] = numFrames
        else:
            outputShape[self.frameAxis] = numFrames - (2 * self.N - 1) // 2

        return outputShape

    def get_config(self) -> dict:
        config = super(CMVN, self).get_config()

        config.update({
            "center": self.center,
            "norm_vars": self.normVar,
            "window": self.N,
            "min_window": self.minN,
            "padding": self.padding,
        })

        return config

    def getWindowedSums(self, frames: tf.Tensor) -> tf.Tensor:
        """
        Returns a tensor containing the sum of elements inside a window of size
        self.N, centered at each frame in the input `frames` array. The `frames`
        array is expected to be left padded with one zero-ed frame along the
        frame axis (-2) to facililate taking the difference of cumulative sums.

        If padding == "SAME", the output array will have the same shape as
        `frames` except that it will have one less frame (padding frame
        discarded). If padding == "VALID", only frames about which the centered
        window fits completely within the `frames` array is evaluated. The
        output number of frames in this case will be `numFrames - (2N-1) // 2 - 1`
        where `numFrames = frames.shape[-2]`. By default "SAME".

        Parameters
        ----------
        frames : tf.Tensor
            Tensor containing frames, shape = (..., frames, samples).

        Returns
        -------
        tf.Tensor
            Tensor containing sum of elements within the window centered at each frame of `frames`.
        """
        # Taking the difference between cs and cs shifted by the length of the
        # window gives the sum of elements within each window.
        cs = tf.math.cumsum(frames, axis=self.frameAxis)
        sumWindows = cs[..., self.N:, :] - cs[..., :-self.N, :]

        # Repeat sum at edges.
        if self.padding == "SAME":
            # TODO (shahruk): figure out how to do this more efficiently.
            sumWindows = tf.concat([
                tf.tile(sumWindows[..., :1, :], [1, self.N // 2, 1]),
                sumWindows,
                tf.tile(sumWindows[..., -1:, :], [1, (self.N - 1) // 2, 1]),
            ], self.frameAxis)

        return sumWindows

    def call(self, inputs):

        inputShape = tf.shape(inputs)
        batchSize = inputShape[self.batchAxis]
        numFrames = inputShape[self.frameAxis]
        featDim = inputShape[self.featAxis]

        # Branch to use when numFrames > window size. Will compute a mean
        # and std for each window.
        def statOverWindows():
            # First padding frames on the left by 1 frame to offset frames by 1 ...
            offsetPad = tf.constant([[0, 0], [1, 0], [0, 0]], dtype=tf.int32)
            offsetInputs = tf.pad(inputs, offsetPad)

            # ... and taking the difference between cumsum and cumsum shifted by
            # the length of the window gives the sum of elements within each
            # window
            xsum = self.getWindowedSums(offsetInputs)
            mean = tf.divide(xsum, tf.cast(self.N, xsum.dtype))
            if self.normVar:
                x2sum = self.getWindowedSums(tf.pow(offsetInputs, 2))
                std = tf.sqrt(tf.divide(x2sum, tf.cast(self.N, x2sum.dtype)) - tf.pow(mean, 2))
                return mean, std
            else:
                return mean, None

        # Branch to use when numFrames <= window size. Will compute a single
        # mean and std using all frames.
        def statOverAllFrames():
            xsum = tf.reduce_sum(inputs, axis=self.frameAxis, keepdims=True)
            mean = tf.divide(xsum, tf.cast(numFrames, xsum.dtype))
            if self.normVar:
                x2sum = tf.reduce_sum(tf.pow(inputs, 2), axis=self.frameAxis, keepdims=True)
                std = tf.sqrt(tf.divide(x2sum, tf.cast(numFrames, x2sum.dtype)) - tf.pow(mean, 2))
                return mean, std
            else:
                return mean, None

        mean, std = tf.cond(
            tf.less_equal(numFrames, self.N),
            true_fn=statOverAllFrames,
            false_fn=statOverWindows,
        )

        if self.padding == "VALID":
            # TODO (shahruk): if self.N > numFrames, this will cause the layer
            # to return an empty tensor since no valid windows are possible.
            # Need somehow to report this back the user, perhaps in the build()
            # method?
            a = self.N // 2
            b = numFrames - (self.N - 1) // 2
            inputs = inputs[..., a:b, :]

        if self.normVar:
            x = tf.divide(inputs - mean, std)
        else:
            x = inputs - mean

        # Reshaping to expected shape (batch size, num frames, num feats). This
        # is done to be stop-gap solution to a problem encountered in tensorflow
        # v2.8.0 during conversion to TFLite, where the feature dimension size
        # is somehow lost. Any layers following the MFCC layer can't compute the
        # input / output shapes and the converter complains. See the issue:
        # https://github.com/shahruk10/kaldi-tflite/issues/13
        return tf.reshape(x, [batchSize, -1, featDim])
