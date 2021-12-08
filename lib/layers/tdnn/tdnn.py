#!/usr/bin/env python3

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Initializer, GlorotUniform


class TDNN(Layer):

    def __init__(self,
                 units: int,
                 context: list = [0],
                 subsampling_factor: int = 1,
                 padding: str = "SAME",
                 use_bias: bool = True,
                 kernel_intializer: Initializer = GlorotUniform(),
                 bias_initializer: Initializer = GlorotUniform(),
                 activation: str = None,
                 name: str = None,
                 **kwargs):
        """
        This layer implements a kaldi styled time delayed neural network layer.
        It's implemented to produce the same output as a TDNN layer implemented
        in Kaldi's Nnet3 framework. 

        Asymmetrical left / right context is allowed just like Kaldi's splicing
        specification (e.g. context = [-3, -1, 0, 1]).

        This layer's weights can be intialized using the `<LinearParams>` and
        `<BiasParams>` of tdnn.affine components with the same number of units
        and context configuration as this layer. 

        Parameters
        ----------
        units : int
            Dimension of layer output.
        context: list, optional,
            List of timesteps to use in the convolution where 0 is the current
            timestep and -N would be the previous Nth timestep and +N would be
            the future Nth timestep. By default [0], no temporal context.
        subsampling_factor: int, optional
            If set to N, will evaluate output for kernel centered at every
            Nth timestep in the input. By default, 1 (no subsampling).
        padding: str, optional
            Padding option can be either "SAME" or "VALID". If "SAME", the input
            will be padded so that the output has the same number of timesteps as
            the input when subsampling_factor = 1. If "VALID", no padding will be
            done, and the kernel will be evaluated only at timestamps where it is
            completely within the input. By default "SAME", (same as Kaldi).
        use_bias: bool, optional
            If true, bias vector added to layer output, by default True.
        kernel_initializer: tf.keras.initializers.Initializer, optional
            Initializer to use when randomly initializing TDNN kernel weights, by
            default GlorotUniform (also called Xavier uniform initializer).
        bias_initializer: tf.keras.initializers.Initializer, optional
            Initializer to use when randomly initializing bias vector, by
            default GlorotUniform (also called Xavier uniform initializer).
        name : str, optional
            Name of the given layer. If auto set if set to None.
            By default None.
        """
        super(TDNN, self).__init__(trainable=True, name=name, **kwargs)

        self.units = units
        self.useBias = use_bias

        self.subsamplingFactor = subsampling_factor
        if self.subsamplingFactor <= 0:
            raise ValueError("subsampling_factor should be > 0")

        self.padding = padding.upper()
        if self.padding not in ["VALID", "SAME"]:
            raise ValueError("padding should be eiter 'VALID' or 'SAME'")

        if context is None:
            self.context = [0]
        elif isinstance(context, int):
            self.context = [context]
        elif isinstance(context, list):
            self.context = context if len(context) > 0 else [0]
        else:
            raise ValueError("context should be None, a list or an integer")

        self.context.sort()
        self.contextOffset = tf.constant([context], dtype=tf.int32)

        self.kernelWidth = len(context)
        self.kernelInitializer = kernel_intializer
        self.biasInitializer = bias_initializer

        self.activation = activation
        if self.activation is not None:
            self.activationFunc = tf.keras.activations.get(activation)

        # Inputs to this layers are expected to be in the shape
        # (batch, timesteps, featdim)
        self.batchAxis = 0
        self.timeAxis = 1
        self.featAxis = -1

    def build(self, input_shape: tuple):

        super(TDNN, self).build(input_shape)

        inputFeatDim = input_shape[self.featAxis]

        # Convolutional kernel weights; 2D kernel with length = 1 and width =
        # length of specified context timesteps. We use a 2D convolution kernel
        # here because it becomes simpler to apply on how the inputs are shaped
        # after applying tf.gather on them; see call()
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1, self.kernelWidth, inputFeatDim, self.units),
            initializer=self.kernelInitializer,
        )

        # Bias vector.
        self.bias = None
        if self.useBias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=self.biasInitializer,
            )

    def compute_output_shape(self, input_shape) -> tuple:

        batchSize = input_shape[self.batchAxis]
        inputTimesteps = input_shape[self.timeAxis]

        start, end = self.getStartEndSteps(inputTimesteps)
        outputTimesteps = (end - start) / self.subsamplingFactor

        outputShape = (batchSize, outputTimesteps, self.units)

        return outputShape

    def get_config(self) -> dict:

        config = super(TDNN, self).get_config()
        config.update({
            "units": self.units,
            "context": self.context,
            "subsampling_factor": self.subsamplingFactor,
            "padding": self.padding,
            "use_bias": self.useBias,
            "kernel_intializer": self.kernelInitializer,
            "bias_initializer": self.biasInitializer,
            "activation": self.activation,
        })

        return config

    def getStartEndSteps(self, inputTimesteps: int) -> Tuple[int, int]:

        start = 0
        end = inputTimesteps
        if self.padding == "VALID":
            if self.context[0] < 0:
                start = -1 * self.context[0]
            if self.context[-1] > 0:
                end = inputTimesteps - self.context[-1]

        return start, end

    def getIndicesToEval(self, inputTimesteps: int) -> tf.Tensor:

        start, end = self.getStartEndSteps(inputTimesteps)
        indices = tf.range(start=start, limit=end, delta=self.subsamplingFactor)
        context = tf.tile(input=self.contextOffset, multiples=[tf.size(indices), 1])
        indices = tf.expand_dims(indices, axis=1)
        indices = context + indices

        # Limiting indices to be within bounds. This is equivalent to padding
        # the input by repeating the values at the boundaries.
        if self.padding == "SAME":
            indices = tf.clip_by_value(indices, 0, inputTimesteps - 1)

        return indices

    @tf.function
    def call(self, inputs):

        inputShape = tf.shape(inputs)
        inputTimesteps = inputShape[self.timeAxis]

        # inputToEval has shape = (batch, numEval, kernelWidth, inputFeatDim)
        indicesToEval = self.getIndicesToEval(inputTimesteps)
        inputToEval = tf.gather(params=inputs, indices=indicesToEval, axis=self.timeAxis)

        # Using 2D convolution with a kernel length of 1, effectively 1D
        # convolution along kernel width. It works out easier this way when
        # working with tf.gather.
        #
        # Furthermore, tf.nn.conv1d reshapes the inputs and invokes tf.nn.conv2d
        # anyway (https://www.tensorflow.org/api_docs/python/tf/nn/conv1d)
        output = tf.nn.conv2d(
            inputToEval, self.kernel, strides=(1, 1), padding="VALID", data_format="NHWC",
        )

        # Removing the dimension along kernelWidth since it has become 1 after
        # applying the convolution above.
        output = tf.squeeze(output, axis=-2)

        if self.useBias:
            output = output + self.bias

        if self.activation is not None:
            output = self.activationFunc(output)

        return output
