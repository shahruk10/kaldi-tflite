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



from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras.layers import Layer


class StatsPooling(Layer):

    """
    This layer implements a kaldi's statistics extraction and pooling
    components, which computes the mean and standard deviation of inputs
    at configured intervals.
    """

    def __init__(self,
                 left_context: int,
                 right_context: int,
                 input_period: int = 1,
                 output_period: int = 1,
                 include_std: bool = True,
                 padding: str = "SAME",
                 epsilon: float = 1e-10,
                 reduce_time_axis: bool = False,
                 name: str = None,
                 **kwargs):
        """
        Instantiates a StatsPooling layer with the given configuration.

        Parameters
        ----------
        left_context : int
            Number of past timesteps (to the left of current timestep)
            to include in the window when computing stats.
        right_context : int
            Number of future timesteps (to the right of current timestep)
            to include in the window when computing stats.
        input_period : int, optional
            Subsampling of timesteps in the window determined by left and right
            contexts. If set to 1, there's no subsampling and all timesteps in the
            window are used to compute stats. If set to N, then only every Nth timestep
            is used in computing the stats. By default 1.
        output_period : int, optional
            Stride of the window over which stats are computed. If set to 1, stats are
            computed after each timestep. If set to N, stats are computed by centering
            the window at every Nth timestep. The output_period should be a multiple of
            the input_period.
        include_std : bool, optional
            If true, the output of this layer will contain the standard deviations, along
            with the means. The standard deviations will comprise the second half of the
            feature dimension. By default True.
        padding : str, optional
            Padding option can be either "SAME" or "VALID". If "SAME", the output
            will be computed such that it has the same number of timesteps as the
            input to this layer. If "VALID", the stat window will be evaluated only
            at timesteps where it is completely within the input. By default "SAME", 
            (same as Kaldi). This argument is called 'padding' to follow the tensorflow
            convention, but actually no padding with zeros etc. is done in this layer.
            Instead, the layer employs a mask to compute stats using only the elements
            that fall within the stat window.
        epsilon : float, optional
            Small float added to variance to avoid dividing by zero, by default 1e-10
        reduce_time_axis : bool, optional
            If true, will compute stats using all input timesteps and reduce the length of
            the time  axis to 1. This will make the layer ignore the output_period and 
            padding arguments, since it will reduce all the timesteps in the input to 1.
            By default, False.
        name : str, optional
            Name of the given layer. Auto set if set to None.
            By default None.

        Raises
        ------
        ValueError
            If left_context > 0 or right_context < 0.

            If input_period or output_period <=0 or output_period is not
            a multiple of input_period if not reducing along the time axis.

            If 'padding' is not set to either "SAME" or "VALID".
        """
        super(StatsPooling, self).__init__(trainable=False, name=name)

        self.leftContext = left_context
        self.rightContext = right_context
        self.inputPeriod = input_period
        self.outputPeriod = output_period
        self.includeStd = include_std
        self.reduce = reduce_time_axis

        if self.leftContext > 0 or self.rightContext < 0:
            raise ValueError("'left_context' must be <= 0 and 'right_context' must be >= 0")

        if self.inputPeriod <= 0 or self.outputPeriod <= 0:
            raise ValueError("'input_period' and 'output_period' must be > 0")

        if self.outputPeriod % self.inputPeriod != 0 and not self.reduce:
            raise ValueError("'output_period' must be a multiple of 'input_period'")

        self.padding = padding.upper()
        if self.padding not in ["VALID", "SAME"]:
            raise ValueError("padding should be either 'VALID' or 'SAME'")

        self.epsilon = tf.constant(epsilon, dtype=self.dtype)
        self.maxWindowWidth = tf.constant(right_context - left_context + 1, dtype=tf.int32)

        self.batchAxis = 0
        self.timeAxis = 1
        self.featAxis = -1

    def compute_output_shape(self, input_shape) -> tuple:

        batchSize = input_shape[self.batchAxis]
        inputTimesteps = input_shape[self.timeAxis]

        outputDim = input_shape[self.featAxis]
        if self.includeStd:
            outputDim = outputDim * 2

        if self.padding == "SAME":
            return (batchSize, inputTimesteps, outputDim)

        indices, _ = self.getIndicesToEval(inputTimesteps)
        outputTimesteps = tf.size(indices)

        return (batchSize, outputTimesteps, outputDim)

    def get_config(self) -> dict:

        config = super(StatsPooling, self).get_config()
        config.update({
            "left_context": self.leftContext,
            "right_context": self.rightContext,
            "input_period": self.inputPeriod,
            "output_period": self.outputPeriod,
            "include_std": self.includeStd,
            "padding": self.padding,
            "epsilon": self.epsilon.numpy(),
        })

        return config

    def getStartEndSteps(self, inputTimesteps: Union[int, tf.Tensor]) -> Tuple[int, int]:

        start, end = 0, inputTimesteps
        if self.padding == "SAME":
            return start, end

        if self.leftContext < 0:
            start = -1 * self.leftContext

        if self.rightContext > 0:
            end = tf.cond(
                tf.less(self.maxWindowWidth, inputTimesteps),
                true_fn=lambda: inputTimesteps - self.rightContext,
                false_fn=lambda: end,
            )

        return start, end

    def getIndicesToEval(self, inputTimesteps: Union[int, tf.Tensor]) -> tf.Tensor:

        start, end = self.getStartEndSteps(inputTimesteps)
        indices = tf.range(start=start, limit=end, delta=self.outputPeriod)

        rightContext = self.rightContext + 1
        rightContext = tf.cond(
            tf.greater(rightContext, inputTimesteps),
            true_fn=lambda: inputTimesteps,
            false_fn=lambda: rightContext,
        )

        windowOffset = tf.expand_dims(
            tf.range(self.leftContext, rightContext, self.inputPeriod), 0
        )

        context = tf.tile(input=windowOffset, multiples=[tf.size(indices), 1])
        indices = tf.expand_dims(indices, axis=1)
        indices = context + indices

        # We will mask any indices outside the bounds when computing statistics.
        # mask has shape = (numEval, windowWidth). Adding dimensions along the
        # batch and feature axis to facililate multiplying with 'gathered'
        # inputs. Final shape of mask = (1, numEval, windowWidth, 1).
        mask = tf.logical_and(tf.greater_equal(indices, 0), tf.less(indices, inputTimesteps))
        mask = tf.expand_dims(tf.expand_dims(mask, -1), 0)

        # For using gather, we limit the indices to be within bounds.
        indices = tf.clip_by_value(indices, 0, inputTimesteps - 1)

        return indices, mask

    def computeStatsAcrossAll(self, inputs) -> tf.Tensor:
        """
        Computes stats over all input timesteps.

        Parameters
        ----------
        inputs :
            Input tensor, or dict/list/tuple of input tensors to use
            for computing stats. Expected shape = (batch, timesteps, featsDim).

        Returns
        -------
        tf.Tensor
            Tensor containing stats computed across all input timesteps.
            Shape = (batch, 1, statsDim), where statsDim = featsDim if only
            computing means, otherwise stats = featDim * 2.
        """
        if self.inputPeriod > 1:
            inputs = inputs[:, ::self.inputPeriod, :]

        mean = tf.reduce_mean(inputs, axis=self.timeAxis, keepdims=True)

        if not self.includeStd:
            return mean

        x2mean = tf.reduce_mean(tf.pow(inputs, 2), axis=self.timeAxis, keepdims=True)
        var = x2mean - tf.pow(mean, 2)
        std = tf.sqrt(tf.nn.relu(var) + self.epsilon)

        return tf.concat([mean, std], self.featAxis)

    def computeStatsAcrossWindows(self, inputs, inputTimesteps: Union[int, tf.Tensor]) -> tf.Tensor:
        """
        Computes stats over windows taken across input timesteps.

        Parameters
        ----------
        inputs :
            Input tensor, or dict/list/tuple of input tensors to use
            for computing stats. Expected shape = (batch, timesteps, featsDim).

        inputTimesteps : Union[int, tf.Tensor]
            Number of timesteps in inputs.

        Returns
        -------
        tf.Tensor
            Tensor containing stats computed across all input timesteps.
            Shape = (batch, numWindows, statsDim), where statsDim = featsDim
            if only computing means, otherwise stats = featDim * 2. numWindows
            is determined by the number of input timesteps, and configured
            padding options.
        """
        # We will need to get input indices, along the the time axis, about
        # which the stats pooling window should be evaluated.
        indicesToEval, mask = self.getIndicesToEval(inputTimesteps)

        # mask shape = (1, numEval, windowWidth, 1)
        # n shape = (1, numEval, 1)
        mask = tf.cast(mask, inputs.dtype)
        n = tf.reduce_sum(mask, axis=2)

        # Computing mean

        # x shape = (batch, numEval, windowWidth, inputFeatDim)
        # xsum and mean shape = (batch, numEval, inputFeatDim)
        x = tf.gather(params=inputs, indices=indicesToEval, axis=self.timeAxis)

        xsum = tf.reduce_sum(tf.multiply(x, mask), axis=2)
        mean = tf.divide(xsum, n)

        if not self.includeStd:
            return mean

        # Computing standard deviation

        # x2 has shape = (batch, numEval, windowWidth, inputFeatDim)
        # x2sum, var and std shape = (batch, numEval, inputFeatDim)
        inputs2 = tf.pow(inputs, 2)
        x2 = tf.gather(params=inputs2, indices=indicesToEval, axis=self.timeAxis)
        x2sum = tf.reduce_sum(tf.multiply(x2, mask), axis=2)
        var = tf.divide(x2sum, n) - tf.pow(mean, 2)
        std = tf.sqrt(tf.nn.relu(var) + self.epsilon)

        return tf.concat([mean, std], self.featAxis)

    def call(self, inputs):

        if self.reduce:
            return self.computeStatsAcrossAll(inputs)

        inputShape = tf.shape(inputs)
        inputTimesteps = inputShape[self.timeAxis]

        if self.padding == "SAME":
            stats = self.computeStatsAcrossWindows(inputs, inputTimesteps)
            if self.outputPeriod > 1:
                return tf.repeat(stats, self.outputPeriod, axis=self.timeAxis)
            else:
                return stats

        return tf.cond(
            tf.greater(inputTimesteps, self.maxWindowWidth),
            true_fn=lambda: self.computeStatsAcrossWindows(inputs, inputTimesteps),
            false_fn=lambda: self.computeStatsAcrossAll(inputs),
        )
