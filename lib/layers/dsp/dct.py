#!/usr/bin/env python3

from typing import Union, Iterable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class DCT(Layer):

    """
    This layer implements the Discrete Cosine Transform. The transform is
    compliant with Kaldi in that we should expect to get the same output as
    Kaldi for the same input.

    Only supports DCT-II at the moment because that's what we need to calculate
    MFCCs.

    This is adapted from PyTorch's Kaldi compliance module:
    https://pytorch.org/audio/stable/_modules/torchaudio/compliance/kaldi.html#mfcc
    """

    def __init__(self,
                 length: int,
                 dct_type: int = 2,
                 norm: str = "ortho",
                 name: str = None,
                 **kwargs):
        """
        Instantiates a DCT layer with the given configuration.
        """
        super(DCT, self).__init__(trainable=False, name=name, **kwargs)

        self.length = length
        if self.length <= 0:
            raise ValueError(f"DCT length must be > 0, got {length}")

        self.dctType = dct_type
        if self.dctType not in [2]:
            raise NotImplementedError(f"DCT-{dct_type} is not supported yet")

        self.norm = norm.lower()
        if self.norm not in ["ortho"]:
            raise NotImplementedError(f"{norm} normalization is not supported yet")

        # The DCT is acheived by multipying the input with the DCT matrix and
        # then carrying out any normalization. The matrix is computed after
        # receiving the shape of the input to this layer in build().
        self.dct = None

        # Inputs to this layers are expected to be in the shape
        # (batch, timesteps, featdim)
        self.featAxis = -1

    def build(self, input_shape: Iterable[Union[int, None]]):
        """
        Precomputes transfrom matrix that needs to be applied on the input
        to achieve the desired DCT Transform.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have three axes,
            (batch, time, feats).

        Raises
        ------
        ValueError
            If input feature length (featDim) < DCT length..
        """
        super(DCT, self).build(input_shape)

        featDim = input_shape[self.featAxis]
        if featDim < self.length:
            raise ValueError("input feature length must be >= DCT length")

        if self.dctType == 2:
            self.computeType2Matrix(featDim)

    def computeType2Matrix(self, inputLength: int):
        """
        Create a DCT-II transformation matrix with shape (inputLength, self.length),
        normalized depending on norm. The matrix when applied to the input tensor,
        acheives the DCT-II transform. See the following link for the equation and
        meaning of terms:

        https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II

        This has been adapted from the `create_dct()` method defined in 
        `torchaudio.functional.functional`:

        https://pytorch.org/audio/stable/_modules/torchaudio/functional/functional.html

        Parameters
        ----------
        inputLength : int
            Number of samples in the input.
        """
        # 0 to N.
        N = inputLength
        n = np.arange(N)
        N = float(N)

        # 0 to K; adding a dimension for matrix multiplication.
        K = self.length
        k = np.expand_dims(np.arange(K, dtype=np.float64), 1)

        # dct.shape = (K, N) = (num_mfcc, num_mels)
        dct = np.cos((np.pi / N) * (n + 0.5) * k)

        if self.norm is None:
            dct *= 2.0
        elif self.norm == "ortho":
            dct[0] *= (1.0 / np.sqrt(2.0))
            dct *= np.sqrt(2.0 / N)
            dct = dct.T  # shape = (N, K) = (num_mels, num_mfcc)

        # Kaldi expects the first cepstral to be the weighted sum of factor
        # sqrt(1/num_mels); Since we multiply the input on the right by the DCT
        # Matrix in this layer (i.e. input @ dct), this would be the first
        # column in the dct. Note that Kaldi uses a left multiply which would be
        # the first column of the kaldi's DCT Matrix.
        dct[:, 0] = np.sqrt(1.0 / N)

        self.dct = tf.constant(dct, dtype=self.dtype)

    def compute_output_shape(self, input_shape: Iterable[Union[int, None]]) -> Tuple[Union[int, None]]:
        """
        Returns the shape of the DCT output, given the shape of the input.

        Parameters
        ----------
        input_shape : Iterable[Union[int, None]]
            Shape of the input to this layer. Expected to have three axes,
            (batch, time, feats).

        Returns
        -------
        Tuple[Union[int, None]]
            Shape of the output of this layer.
        """
        batch, time, feat = input_shape
        outputShape = (batch, time, self.length)

        return outputShape

    def get_config(self) -> dict:
        config = super(DCT, self).get_config()
        config.update({
            "length": self.length,
            "dct_type": self.dctType,
            "norm": self.norm,
        })

        return config

    def call(self, inputs):
        return tf.matmul(inputs, self.dct)
