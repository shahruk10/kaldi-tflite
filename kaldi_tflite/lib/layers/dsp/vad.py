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


import tensorflow as tf
from tensorflow.keras.layers import Layer


class VAD(Layer):

    """
    This layer mimics kaldi's `compute-vad` and `select-voiced-frames` binaries
    by applying a simple energy based voice activiy detection filter on input
    feature frames. 

    The input to this layer is expected to be a 3D tensor (batch, frames, feat)
    containing the output of the `MFCC` layer. The first coefficient along the
    feat axis should contain the log-energy of each frame (`use_energy=True` in
    the MFCC layer).

    The layer can be configured to return either;

      - Binary masks containing 1s and 0s corresponding to frames with voice
        activity and those without respectively (return_indexes=False). This is
        the default.

      - Frame indexes corresponding to frames with voice acitivty which can be
        used with `tf.gather_nd` to gather the active frames into a single
        tensor (return_indexes=False).
    """

    def __init__(self,
                 energy_mean_scale: float = 0.5,
                 energy_threshold: float = 5,
                 frames_context: int = 0,
                 proportion_threshold: float = 0.6,
                 return_indexes=False,
                 energy_coeff: int = 0,
                 name: str = None,
                 **kwargs):
        """
        Initializes VAD layer with given configuration.

        Parameters
        ----------
        energy_mean_scale : float, optional
            If this is set to `s`, to get the actual threshold we let `m` be
            the mean log-energy of the file, and use `s*m + vad-energy-threshold`.
            By default, 0.5.
        energy_threshold : float, optional
            Constant term in energy threshold for VAD (also see energy_mean_scale.
            By default 5.
        frames_context : int, optional
            Number of frames of context on each side of central frame, in window for
            which energy is monitored. By default 0.
        proportion_threshold : float, optional
            Parameter controlling the proportion of frames within the window that need
            to have more energy than the threshold. By default 0.6.
        return_indexes : bool, optional
            If true, the output of this layer will be a tensor containing indexes of
            active frames for each frame sequence in the batch. Otherwise, the layer
            outputs binary masks containing 1s and 0s corresponding to active and
            inactive frames respectively.
        energy_coeff : int, optional
            The coefficient in each frame that should be used as the value on which
            the threshold will be applied. By default 0 (the first coefficient).
        name : str, optional
            Name of the given layer. If auto set if set to None.
            By default None.

        Raises
        ------
        ValueError
            If energy_mean_scale < 0.
            If frames_context < 0.
            If proportion_threshold not between 0 and 1 (exclusive).
        """
        super(VAD, self).__init__(trainable=False, name=name, **kwargs)

        if energy_mean_scale < 0:
            raise ValueError("`energy_mean_scale` must be >= 0")
        if frames_context < 0:
            raise ValueError("`frames_context` must be >= 0")
        if proportion_threshold <= 0 or proportion_threshold >= 1:
            raise ValueError("`proportion_threshold` must be between 0 and 1 (exlcusive)")

        self.energyThreshold = tf.constant(energy_threshold, dtype=self.dtype)
        self.energyMeanScale = tf.constant(energy_mean_scale, dtype=self.dtype)
        self.propThreshold = tf.constant(proportion_threshold, dtype=self.dtype)
        self.returnIndexes = return_indexes
        self.useEnergyMean = energy_mean_scale > 0

        self.framesContext = frames_context
        self.windowSize = self.framesContext * 2 + 1

        # Kernel will be convolved with thresholded log energies to give the
        # count of frames within the kernel window having energy greater than
        # the threshold.
        self.kernel = None
        if self.windowSize > 1:
            # kernel shape = (window width, input dim, output dim)
            self.kernel = tf.reshape(
                tf.constant([1.0 for i in range(self.windowSize)], dtype=self.dtype),
                [self.windowSize, 1, 1],
            )

            # When computing frame prorportions, Kaldi divides by "valid" number
            # of frames, i.e. that fit completely within the kernel window at
            # the edges. To acheive the same effect, we precompute these updates
            # to the `windowSizes` tensor here and apply them later in `call()`.
            N = self.windowSize
            self.windowSizesAtEdges = tf.constant(
                list(range(N // 2 + 1, N, 1)) +  # left edge = 3, 4 for N = 5
                list(range(N - 1, N // 2, -1)),  # right edge = 4, 3 for N = 5
            )

            # Indexes will be added to number of frames and then modulo-ed to
            # get all positive indexes.
            self.windowEdgeIndexes = tf.expand_dims(tf.constant(
                list(range(0, N // 2)) +  # left edge indexes = 0, 1 for N = 5
                list(range(-N // 2 + 1, 0)),  # right edge indexes = -1, -2 for N = 5
                dtype=tf.int32), 1)

        # Frames are expected to be the second to last axis of the input; The
        # 0th coefficient in each frame is expected to contain the log energy of
        # the frame by default.
        self.energyCoef = energy_coeff
        self.frameAxis = -2

    def get_config(self) -> dict:
        config = super(VAD, self).get_config()
        config.update({
            "energy_mean_scale": self.energyMeanScale.numpy(),
            "energy_threshold": self.energyThreshold.numpy(),
            "frames_context": self.framesContext,
            "proportion_threshold": self.propThreshold.numpy(),
            "return_indexes": self.returnIndexes,
            "energy_coeff": self.energyCoef,
        })

        return config

    def call(self, inputs):

        logEnergies = inputs[..., self.energyCoef:self.energyCoef + 1]
        numFrames = tf.shape(logEnergies)[self.frameAxis]

        # Computing energy threshold and frame level decisions.
        if self.useEnergyMean:
            meanEnergy = tf.reduce_mean(logEnergies, axis=self.frameAxis, keepdims=True)
            energyThreshold = self.energyThreshold + self.energyMeanScale * meanEnergy
        else:
            energyThreshold = self.energyThreshold

        frameDecisions = tf.greater(logEnergies, energyThreshold)

        # If window size is 1, we return frame level decisions.
        if self.windowSize == 1:
            if self.returnIndexes:
                return tf.where(tf.squeeze(frameDecisions, axis=-1))
            else:
                return tf.cast(frameDecisions, inputs.dtype)

        # Getting count of frames within each context window having energies
        # greater than the threshold.
        frameDecisions = tf.cast(frameDecisions, self.kernel.dtype)
        counts = tf.nn.conv1d(
            frameDecisions, self.kernel, stride=1, padding="SAME", data_format="NWC",
        )

        # Dividing counts by "valid" window size at each frame to get
        # proportions of frames exceeding the proporition threshold for the
        # context windows.
        windowSizes = self.windowSize * tf.ones((numFrames,), dtype=self.dtype)

        # Updating window sizes at the edges to be equal to actual number of
        # contributing frames, i.e. sans padding.
        edgeSizes = tf.cast(self.windowSizesAtEdges, windowSizes.dtype)
        edgeIndexes = tf.math.floormod(self.windowEdgeIndexes + numFrames, numFrames)
        windowSizes = tf.tensor_scatter_nd_update(windowSizes, edgeIndexes, edgeSizes)

        # Adding batch and feat dimension so that it will be broadcasted when dividing.
        windowSizes = tf.reshape(windowSizes, (1, numFrames, 1))
        proportions = tf.divide(counts, windowSizes)

        frameDecisions = tf.greater_equal(proportions, self.propThreshold)
        if self.returnIndexes:
            return tf.where(tf.squeeze(frameDecisions, axis=-1))
        else:
            return tf.cast(frameDecisions, inputs.dtype)
