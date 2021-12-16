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



from typing import Iterable

import numpy as np
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import Initializer, Constant


class BatchNorm(BatchNormalization):

    """
    This layer implements a kaldi styled batch normalization layer. The
    actual implementation is the same as tensorflow's `BatchNormalization`
    layer which this layer inherits. The instantiation arguments are modified
    to match those used in Kaldi, and the `set_weights()` method is overloaded
    to be able to interpret the parameters of Kaldi's <BatchNormComponent>, 
    loaded from an nnet3 model.

    Kaldi only applies scaling to the normalized output unlike tensorflow which
    also has the option to apply a learnable offset. Kaldi also has the option to
    divide the normalization procedure into sub-blocks instead of using the whole
    input (when block_dim != dim). This is generally not used and so not implemented
    here.
    """

    def __init__(self,
                 axis: int = -1,
                 momentum: float = 0.99,
                 target_rms: float = 1.0,
                 epsilon: float = 0.001,
                 mean_initializer: Initializer = Constant(0),
                 variance_initializer: Initializer = Constant(1),
                 name: str = None,
                 **kwargs):
        """
        Instantiates a BatchNorm layer with the given configuration.

        Parameters
        ----------
        axis : int, optional
            The axis that should be normalized (typically the features axis), by default -1
        momentum : float, optional
            Momentum for the moving average, by default 0.99
        target_rms : float, optional
            This is the equivalent of gamma in the tensorflow. It is the
            scaling factor applied to the normalized data. It is a way to
            control how fast the following layer learns (smaller -> slower).
            A differece here is that Kaldi uses a constant scalar, while tensorflow
            has this as a lernable tensor. By default 1.0.
        epsilon : float, optional
            Small float added to variance to avoid dividing by zero, by default 0.001
        mean_initializer : Initializer, optional
            Initializer for the moving mean. , by default Zeros
        variance_initializer : Initializer, optional
            Initializer for the moving mean, by default Ones
        name : str, optional
            Name of the given layer. Auto set if set to None.
            By default None.
        """
        self.targetRMS = target_rms
        gammaInitializer = Constant(target_rms)

        super(BatchNorm, self).__init__(
            axis=axis, momentum=momentum, epsilon=epsilon,
            center=False, scale=True,
            moving_mean_initializer=mean_initializer,
            moving_variance_initializer=variance_initializer,
            gamma_initializer=gammaInitializer,
            name=name, **kwargs,
        )

    def get_config(self) -> dict:
        """
        Returns the configuration of the
        """
        config = super(BatchNorm, self).get_config()
        config.update({
            "target_rms": self.targetRMS,
        })

        return config

    def set_weights(self, weights: Iterable[np.ndarray], fmt: str = "kaldi"):
        """
        Sets the weights of the layer, from numpy arrays.

        Parameters
        ----------
        weights : Iterable[np.ndarray]
            Weights as a list of numpy arrays containing [gamma, mean, var].
            If the format is "kaldi" gamma is a scalar (equal to target-rms),
            while for tensorflow, it's a 1D array.
        fmt : str, optional
            The format of the supplied list of weights - either "kaldi" or 
            "tensorflow", by default "kaldi".

        Raises
        ------
        ValueError
            If the "order" is not "kaldi" or "tensorflow".
            if the number of weights in the weight list is unexpected.
            If the shape of the weights do not match expected shapes.
        """
        if fmt not in ["kaldi", "tensorflow"]:
            raise ValueError(f"expected 'fmt' to be either 'kaldi' or 'tensorflow', got {fmt}")

        if fmt == "tensorflow":
            return super(BatchNorm, self).set_weights(weights)

        if len(weights) != 3:
            raise ValueError(f"expected a weight list of length 3, got {len(weights)}")

        targetRMS, mean, var = weights
        gamma = targetRMS * np.ones_like(mean)

        return super(BatchNorm, self).set_weights([gamma, mean, var])
