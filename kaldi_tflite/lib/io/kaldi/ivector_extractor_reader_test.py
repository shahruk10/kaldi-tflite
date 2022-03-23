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


import unittest
import numpy as np

from kaldi_tflite.lib.io import KaldiIvecExtractorReader
from kaldi_tflite.lib.testdata import RefIvectorExtractor


class TestKaldiIvecExtractorReader(unittest.TestCase):

    def compare_parameters(self, want, got):
        self.assertEqual(want["numGauss"], got.numGauss)
        self.assertEqual(want["featDim"], got.featDim)
        self.assertEqual(want["ivecDim"], got.ivecDim)
        self.assertEqual(want["priorOffset"], got.priorOffset)

        n = want["numGauss"]
        self.assertTrue(n > 0, f"expected num_gaussians > 0, got 0")

        self.assertTrue(
            len(got.M) == n,
            f"expected at least {n} `M` matrices, got {len(got.M)}",
        )
        self.assertTrue(
            np.array_equal(want["M"], got.M[0]),
            "did not get expected `M` matrix",
        )

        self.assertTrue(
            len(got.sigmaInv) == n,
            f"expected at least {n} `sigmaInv` matrices, got {len(got.M)}",
        )
        self.assertTrue(
            np.array_equal(want["sigmaInv"], got.sigmaInv[0]),
            "did not get expected `sigmaInv` matrix",
        )

        wantSigmaInvM = np.matmul(want["sigmaInv"], want["M"])
        self.assertTrue(
            np.array_equal(wantSigmaInvM, got.sigmaInvM[0]),
            "did not get expected `sigmaInvM` matrix",
        )

        # U contains the element from the lower triangular part.
        wantU = np.matmul(want["M"].T, wantSigmaInvM)
        idx = np.tril_indices(wantU.shape[0])
        wantU = wantU[idx]
        self.assertTrue(
            np.array_equal(wantU, got.U[0]),
            "did not get expected `U` matrix",
        )

    def test_KaldiIvecExtractorReader(self):
        testNames = [f"dummy_{i:03d}" for i in range(1, 16)]
        for name in testNames:
            with self.subTest(name=name):
                mdlFile = RefIvectorExtractor.getModel(name)
                want = RefIvectorExtractor.getParams(name)
                got = KaldiIvecExtractorReader(mdlFile, binary=True)
                self.compare_parameters(want, got)


if __name__ == "__main__":
    unittest.main()
