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
import numpy.typing as npt

from kaldi_tflite.lib.io import KaldiObjReader


class KaldiIvecExtractorReader(KaldiObjReader):
    """
    Reader for kaldi ivector extractor model files. It parses the model file,
    loads and derives the parameters.
    """

    def __init__(self, ivec_path: str, binary: bool = True):
        """
        Initializes the KaldiIvectorExtractorReader by reading in the data from
        the model file at the path specified.

        Parameters
        ----------
        ivec_path : str
            Path to ivector extractor model file.
        binary : bool
            If true, will read the file as a binary file. Otherwise will
            parse the file as a text file (not implemented at the moment).
        """
        super(KaldiIvecExtractorReader, self).__init__(ivec_path, binary=binary)

        # Number of gaussians in UBM the gaussian mixture model.
        self.numGauss: int = None

        # Dimeinsions in the input features.
        self.featDim: int = None

        # Dimensions in the i-vector.
        self.ivecDim: int = None

        # The comments on each parameter are taken from the kaldi repo
        # (src/ivector/ivector-extractor.h). The notation used for dimensions
        # are as follows: I = num_gaussians, D = input_feat_dim, S = ivector_dim

        # Weight projection vectors, if used.  Dimension is [I][S]
        self.w: npt.NDArray[np.float64] = None

        # If we are not using weight-projection vectors, stores the Gaussian
        # mixture weights from the UBM.  This does not affect the iVector; it is
        # only useful as a way of making sure the log-probs are comparable
        # between systems with and without weight projection matrices.
        self.wVec: npt.NDArray[np.float64] = None

        # Ivector-subspace projection matrices, dimension is [I][D][S]. The I'th
        # matrix projects from ivector-space to Gaussian mean. There is no mean
        # offset to add -- we deal with it by having a prior with a nonzero
        # mean.
        self.M: Iterable[npt.NDArray[np.float64]] = None

        # Inverse variances of speaker-adapted model, dimension [I][D][D].
        self.sigmaInv: Iterable[np.NDArray[np.float64]] = None

        # 1st dim of the prior over the ivector has an offset, so it is not
        # zero. This is used to handle the global offset of the speaker-adapted
        # means in a simple way.
        self.priorOffset: float = None

        # Below are *derived variables* that can be computed from the variables
        # above.

        # U_i = M_i^T \Sigma_i^{-1} M_i is a quantity that comes up in ivector
        # estimation.  This is conceptually a symmetric matrix, but here we store
        # only the lower triangular part as a flat array.
        self.U: npt.NDArray[np.float64] = None

        # The product of Sigma_inv_[i] with M_[i].
        self.sigmaInvM: npt.NDArray[np.float64] = None

        self.read()
        self.deriveVars()

    def read(self):
        """
        Reads the loaded data and parses the parameters of the
        the ivector extractor model.

        Raises
        ------
        ValueError
            If file is not in the expected format.
        """
        self.expectToken("<IvectorExtractor>")

        self.expectToken("<w>")
        self.w = self.readMat()

        self.expectToken("<w_vec>")
        self.wVec = self.readVec()

        self.expectToken("<M>")
        self.numGauss = self.readInt()
        self.M = [self.readMat() for _ in range(self.numGauss)]

        self.expectToken("<SigmaInv>")
        self.sigmaInv = [self.readPackedMat() for _ in range(self.numGauss)]

        self.expectToken("<IvectorOffset>")
        self.priorOffset = self.readDouble()

        self.expectToken("</IvectorExtractor>")

    def deriveVars(self):
        """
        Derives further parameters from the ones loaded.

        Raises
        ------
        ValueError
            If parameters are missing or were not parsed
            correctly from the model file. 
        """
        if len(self.M) == 0:
            raise ValueError("expected at least 1 projection matrix (M_), got 0")

        self.featDim = self.M[0].shape[0]
        self.ivecDim = self.M[0].shape[-1]

        self.sigmaInvM = np.matmul(self.sigmaInv, self.M)

        self.U = np.zeros(
            [self.numGauss, int(self.ivecDim / 2) * (self.ivecDim + 1)], dtype=np.float64,
        )

        for i in range(self.numGauss):
            tmp = np.matmul(self.M[i].T, self.sigmaInvM[i])
            k = 0
            for m in range(self.ivecDim):
                for n in range(m + 1):
                    self.U[i, k] = tmp[m, n]
                    k = k + 1
