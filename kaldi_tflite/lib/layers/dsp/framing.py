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


class Framing(Layer):

    """
    This layer implements a framer that takes a tensor containing something like
    audio samples, and creates 'frames' from them of the given size and shift.
    The last axis is considered to be containing the 'samples' that need to be
    framed.

    This layer can be used to convert a mono audio waveform into overlapping
    audio frames, which can then be processed by other Layers that expect frames
    such as `Windowing`, `Spectrogram`, `FilterBank`, `MFCC` etc.

    The framing of audio samples is compliant with `FrameExtracionOptions` in
    Kaldi. However, unlike Kaldi, this layer does not do any padding of the
    input when creating frames. It is expected that user will do this as
    necessary when providing the first / last buffer of input samples to this
    layer. 

    Post-processing of frames such as applying pre-emphasis and windowing is
    delegated to a separate `Windowing` layer. This is because one may *not*
    want to include this `Framing` layer in the model, and choose to buffer
    samples and generate frames outside of the model.
    """

    def __init__(self,
                 frame_length_ms: float = 25.0,
                 frame_shift_ms: float = 10.0,
                 sample_frequency: float = 16000.0,
                 name: str = None,
                 dynamic_input_shape: bool = False,
                 **kwargs):
        """
        Initializes Framing layer with the given configuration.

        Parameters
        ----------
        frame_length_ms : float, optional
            Frame length in milliseconds, by default 25.0
        frame_shift_ms : float, optional
            Frame shift in milliseconds, by default 10.0
        sample_frequency : float, optional
            Sampling frequency in hertz, by default 16000.0
        dynamic_input_shape : bool, optional
            If true, the input to this layer can have different or unknown
            shape. At inference time, the indexes along the last axis that
            will be gathered into frames wll be recomputed each time. If this
            is false, then the indexes are never recomputed, and the layer
            will always expect the same input shape.
        name : str, optional
            Name of the given layer. If auto set if set to None.
            By default None.

        Raises
        ------
        ValueError
            If frame_length, frame_shift or sample_frequency <= 0.
        """
        super(Framing, self).__init__(trainable=False, name=name, **kwargs)

        self.sampleFreq = sample_frequency
        self.frameSizeMs = frame_length_ms
        self.frameShiftMs = frame_shift_ms
        self.dynamicInputShape = dynamic_input_shape

        self.batchAxis = 0
        self.sampleAxis = -1

        if self.frameSizeMs <= 0 or self.frameShiftMs <= 0 or self.sampleFreq <= 0:
            raise ValueError("frame_length, frame_shift and sample_frequency should be > 0")

        # Frame size and shift in number of samples.
        self.frameSize = int(sample_frequency * frame_length_ms / 1000.0)
        self.frameShift = int(sample_frequency * frame_shift_ms / 1000.0)

        if self.frameSize <= 0:
            raise ValueError("frame_length should be high enough to contain at least 1 sample")
        if self.frameShift <= 0:
            raise ValueError("frame_shift should be high enough to shift by at least 1 sample")

        # frameOffsetIndexes contains the indexes of samples in the frame,
        # with 0 being at the center of the frame; e.g. for frameSize = 8, it
        # would be [ [ -4, -3 ,-2 ,-1, 0, 1, 2, 3 ] ].
        self.halfFrameSize = self.frameSize // 2
        self.frameOffsetIndexes = tf.range(-self.halfFrameSize, self.halfFrameSize)
        self.frameOffsetIndexes = tf.expand_dims(self.frameOffsetIndexes, 0)

        # The indexes of the input that will be put into frames will be
        # computed in build() after getting input_shape.
        self.frameIndexes = None
        self.numInputSamples = None

    def build(self, input_shape: Iterable[Union[int, None]]):
        """
        Precomputes the sample indexes which will be gathered into
        each frame. The number of samples (last axis) must not be
        unknown.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer.

        Raises
        ------
        ValueError
            If input_shape is unknown and dynamic_input_shape = False.
            If length of sample axis < frame size.
        """
        numSamples = input_shape[self.sampleAxis]

        if numSamples is None and not self.dynamicInputShape:
            raise ValueError(
                "input_shape must not be unknown if dynamic_input_shape set to False",
            )

        if numSamples is None:
            # Will have to dynamically compute indexes are inference time.
            return

        if numSamples < self.frameSize:
            raise ValueError(
                f"input sample size (axis={self.sampleAxis}) must be "
                f">= frame size ({self.frameSize})",
            )

        self.numInputSamples = numSamples
        self.frameIndexes = self.computeFrameIndexes(numSamples)

    def compute_output_shape(self, input_shape: Iterable[Union[int, None]]) -> Tuple[Union[int, None]]:
        """
        Returns the shape of the output tensor containing frames,
        given the shape of the input.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have two axes,
            (batch, samples or feats).

        Returns
        -------
        Tuple[Union[int, None]]
            Shape of the output of this layer.

        Raises
        ------
        ValueError
            If input_shape is unknown and dynamic_input_shape = False.
        """
        numSamples = input_shape[self.sampleAxis]

        if numSamples is None and not self.dynamicInputShape:
            raise ValueError(
                "input_shape must not be unknown if dynamic_input_shape set to False",
            )

        if numSamples is None:
            numFrames, frameSize = None, self.frameSize
        else:
            frameIndexes = self.computeFrameIndexes(numSamples)
            numFrames, frameSize = tf.shape(frameIndexes)
            assert self.frameSize == frameSize

        sampleAxis = self.sampleAxis
        if sampleAxis < 0:
            sampleAxis += len(input_shape)

        outputShape = []
        for axis, dim in enumerate(input_shape):
            if axis != sampleAxis:
                outputShape.append(dim)
            else:
                outputShape.extend([numFrames, frameSize])

        return outputShape

    def get_config(self) -> dict:
        config = super(Framing, self).get_config()
        config.update({
            "frame_length": self.frameSizeMs,
            "frame_shift": self.frameShiftMs,
            "sample_frequency": self.sampleFreq,
            "dynamic_input_shape": self.dynamicInputShape,
        })

        return config

    def computeFrameIndexes(self, numSamples: int) -> tf.Tensor:
        """
        Computes the indexes in the input to this layer that will
        be gathered in each output frame.

        Parameters
        ----------
        numSamples : int
            Total number of samples given to this layer at a time.

        Returns
        -------
        tf.Tensor
            Indexes of samples gathered into each frame, shape = (numFrames, frameSize)
        """
        # Sample indexes on which the frames will be centered. We assume the
        # input to this layer has been appropriately padded to facilitate
        # streaming input - i.e. subsequent input should have samples from the
        # previous input at its leading edge if the frames are overlapping.
        centerIndexes = tf.range(
            start=self.halfFrameSize,
            limit=numSamples - self.halfFrameSize + 1,
            delta=self.frameShift,
        )

        offset = tf.tile(input=self.frameOffsetIndexes, multiples=[tf.size(centerIndexes), 1])
        centerIndexes = tf.expand_dims(centerIndexes, axis=1)
        indexes = offset + centerIndexes

        return indexes

    def call(self, inputs):

        # Gathering frames using precomputed indexes.
        if not self.dynamicInputShape:
            # frames.shape = (batch, numFrames, frameSize)
            frames = tf.gather(params=inputs, indices=self.frameIndexes, axis=self.sampleAxis)
            return frames

        inputShape = tf.shape(inputs)
        batchSize = inputShape[self.batchAxis]
        numSamples = inputShape[self.sampleAxis]

        # Recomputing indexes and gathering frames.
        frameIndexes = self.computeFrameIndexes(numSamples)
        frames = tf.gather(params=inputs, indices=frameIndexes, axis=self.sampleAxis)

        # Reshaping so that the output shape of this layer contains the frame
        # size for downstream layers. Reshaping just changes metadata of the
        # tensor instead of actually reallocating anything, so this should be OK.
        # TODO (shahruk): verify above / find better solution.
        frames = tf.reshape(frames, (batchSize, -1, self.frameSize))

        return frames
