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


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class PLDA(Layer):

    """
    This layer implements Probabilistic Linear Discriminant Analysis: see
    "Probabilistic Linear Discriminant Analysis" by Sergey Ioffe, ECCV 2006.
    This layer can be initialized with a PLDA model trained using Kaldi for
    transforming and scoring i-vectors / x-vectors.

    The layer expects a either a 2D tensor of shape (batch, vec_dim) or a 3D
    tensor of shape (batch, 1, vec_dim) where the vectors to be transformed and
    scored are stacked along the batch axis. The vec_dim must be the same as the
    expected input dimension of this layer (`dim` argument).

    The layer outputs a 2D tensor of shape (batch, batch) containing the
    pairwise scores between the possible pairs of vectors in the input. 
    """

    def __init__(self,
                 dim: int,
                 plda_mean: np.ndarray,
                 plda_transform: np.ndarray,
                 plda_psi: np.ndarray,
                 normalize_length: bool = True,
                 simple_length_norm: bool = False,
                 dtype: tf.dtypes.DType = tf.float64,
                 return_transformed: bool = True,
                 name: str = None):
        """
        Initializes PLDA layer with given configuration.

        Parameters
        ----------
        dim : int
            Dimension of input iVector / xVector on which PLDA matrix
            was trained on.
        plda_mean: np.ndarray
            Mean vector of samples in original space. Shape = (dim, ).
        plda_transform: np.darray
            Transform matrix that makes within-class covar unit diagonalizes the
            between-class covar. Shape = (dim, dim).
        plda_psi: np.ndarray
            The between-class (diagonal) covariance elements, in decreasing order.
            Shape = (dim, 1).
        normalize_length : bool, optional
            If true, do length normalization as part of PLDA.
            This does not set the length unit; by default it instead
            ensures that the inner product with the PLDA model's inverse
            variance (which is a function of how many utterances the iVector
            was averaged over) has the expected value, equal to the iVector
            dimension. By default True.
        simple_length_norm : bool, optional
            If true, replace the default length normalization by an
            alternative that normalizes the length of the iVectors to
            be equal to the square root of the iVector dimension.
            By default False.
        dtype : tf.dtype, optional
            Data type of model parameters and the inputs that will be
            provided to this model. By default tf.float64 (same as Kaldi),
            but conversion to TFLite without the need of FlexOps require
            this to be set to tf.float32 or lower.
        return_transformed : bool, optional
            If true, will return the inputs transformed by the PLDA transform
            matrix along with the PLDA scores. 
        name : str, optional
            Name of the given layer. Is auto set if set to None.
            By default None.
        """
        super().__init__(trainable=False, name=name)

        self.dim = dim
        self.normalizeLength = normalize_length
        self.simpleLengthNorm = simple_length_norm
        self.paramDtype = dtype
        self.returnTransformed = return_transformed

        # This gets updated during `build()`. By default we expect input tensors
        # to be (batch, 1, dim).
        self.inputRank = 3

        # Converting parameters into tensors.
        self.mean = tf.constant(plda_mean, dtype=self.paramDtype)
        self.transformMat = tf.constant(plda_transform, dtype=self.paramDtype)
        self.psi = tf.constant(plda_psi, dtype=self.paramDtype)

        # Checking shapes of parameters are consistent.
        self.assertParamShapes()

        # Adding an extra dimensions to vectors for matrix multiplication.
        self.mean = tf.reshape(self.mean, (self.dim, 1))
        self.psi = tf.reshape(self.psi, (self.dim, 1))

        # Derived variable: `-1.0 * self.transformMat @ self.mean`
        self.offset = -1.0 * tf.matmul(self.transformMat, self.mean)

        # Storing dim as float64 and square root of dimension for use in computation.
        self.dim = tf.constant(self.dim, dtype=self.paramDtype)
        self.sqrtDim = tf.math.sqrt(self.dim)

        # Natural Log of 2 * pi. Used in computing log likelihood.
        self.log2pi = tf.constant(1.8378770664093454835606594728112, dtype=self.paramDtype)

    def build(self, input_shape: tuple):
        super(PLDA, self).build(input_shape)

        vecDim = input_shape[-1]
        if vecDim != self.dim:
            raise ValueError(f"expected input vector dimension to be {self.dim}, got {vecDim}")

        self.inputRank = len(input_shape)
        if self.inputRank not in [2, 3]:
            raise ValueError(f"expected input tensor rank to be 2 or 3, got {len(input_shape)}")

    def assertParamShapes(self):
        meanDims = len(self.mean.shape)
        assert meanDims == 1, \
            f"plda_mean must be a vector, got dimension={meanDims}"

        psiDims = len(self.psi.shape)
        assert psiDims == 1, \
            f"plda_psi must be a vector, got dimension={psiDims}"

        transformMatDims = len(self.transformMat.shape)
        assert transformMatDims == 2, \
            f"plda_transform_mat must be a matrix, got dimension={transformMatDims}"

        meanDim = self.mean.shape[0]
        psiDim = self.psi.shape[0]
        matRow = self.transformMat.shape[0]
        matCol = self.transformMat.shape[1]

        assert meanDim == self.dim, \
            f"plda_mean dimension size ({meanDim}) != input dim ({self.dim})"
        assert self.psi.shape[0] == self.dim, \
            f"plda_psi dimension size ({psiDim}) != input dim ({self.dim})"
        assert self.transformMat.shape[0] == self.dim, \
            f"plda_transform_mat dimension size ({matRow}) != input dim ({self.dim})"
        assert self.transformMat.shape[0] == self.transformMat.shape[1], \
            f"plda_transform_mat ({matRow} x {matCol}) is not a square matrix"

    def getNormFactor(self, transformed, num_examples=1):
        assert num_examples > 0, "num_examples must be greater than 1"

        # Work out the normalization factor.  The covariance for an average over
        # "num_examples" training iVectors equals self.psi + I/num_examples.
        transformedSq = tf.math.pow(transformed, 2.0)

        # invCovar will equal 1.0 / (self.psi + I/num_examples).
        invCovar = self.psi
        invCovar += (1.0 / num_examples)
        invCovar = tf.math.reciprocal(invCovar)

        # transformed vector should have covariance (self.psi + I/num_examples),
        # i.e. within-class/num_examples plus between-class covariance.  So
        # `transformedSq . (I/num_examples + self.psi)^{-1}` should be equal to
        # the dimension.
        dotProd = tf.reduce_sum(tf.multiply(invCovar, transformedSq), axis=1, keepdims=True)
        normFactor = tf.math.sqrt(tf.divide(self.dim, dotProd))

        return normFactor

    def transformVector(self, inputs, num_examples=1.0):
        transformed = self.offset
        transformed += tf.matmul(self.transformMat, inputs)

        if self.normalizeLength:
            if self.simpleLengthNorm:
                normFactor = tf.divide(self.sqrtDim, tf.norm(transformed, ord=2, axis=1, keepdims=True))
            else:
                normFactor = self.getNormFactor(transformed, num_examples)

            transformed = tf.multiply(transformed, normFactor)

        return transformed

    def logLikelihood(self, inputs, mean, var):
        # Taking the sum of logs of var
        logDet = tf.math.reduce_sum(tf.math.log(var))  # shape = ()

        # Computing the square difference between all pairs of mean and input.
        # Here, the mean (which has been transposed above) is broadcasted along
        # the the first axis for each entry in the batch.
        sqDiff = tf.math.pow(inputs - mean, 2.0)    # shape = (batch, 512, batch)
        var = tf.math.reciprocal(var)               # shape = (512, 1)

        # Dot product between sqDif and inverse var.
        dotProd = tf.reduce_sum(tf.multiply(sqDiff, var), axis=1)  # shape = (batch, batch)

        loglike = -0.5 * (logDet + (self.log2pi * self.dim) + dotProd)

        return loglike

    def logLikelihoodRatio(self, inputs, num_examples=1.0):

        n = tf.constant(num_examples, dtype=self.paramDtype)
        I = tf.constant(1.0, dtype=self.paramDtype)

        # First computing `loglikeGivenClass`
        #
        # "mean" will be the mean of the distribution if it comes from the
        # training example. The mean is:
        #  `(n/self.psi) / (n/(self.psi+ I)) * vec`
        #
        # "variance" will be the variance of that distribution, equal to:
        #   `I + (self.psi / (n * self.psi + I))`
        mean = n * tf.multiply(self.psi, inputs)
        mean = tf.divide(mean, (n * self.psi) + I)         # shape = (batch, 512, 1)
        mean = tf.transpose(mean, [2, 1, 0])               # shape = (1, 512, batch)
        var = I + tf.divide(self.psi, (n * self.psi) + I)  # shape = (512, 1)

        loglikeGivenClass = self.logLikelihood(inputs, mean, var)

        # Now computing `loglikeWithoutClass`
        #
        # Here the mean is zero and the variance is `I + self.psi`.
        mean = tf.zeros_like(mean)
        var = I + self.psi
        loglikeWithoutClass = self.logLikelihood(inputs, mean, var)

        # Computing the ratio
        loglikeRatio = loglikeGivenClass - loglikeWithoutClass

        return loglikeRatio

    def call(self, inputs):

        if self.inputRank == 2:
            inputs = tf.expand_dims(inputs, -1)
        elif self.inputRank == 3:
            inputs = tf.transpose(inputs, [0, 2, 1])
        else:
            raise ValueError(f"expected input tensor rank to be 2 or 3, got {self.inputRank}")

        inputs = tf.cast(inputs, self.paramDtype)
        transformed = self.transformVector(inputs)
        scores = self.logLikelihoodRatio(transformed)

        if self.returnTransformed:
            return scores, transformed
        else:
            return scores
