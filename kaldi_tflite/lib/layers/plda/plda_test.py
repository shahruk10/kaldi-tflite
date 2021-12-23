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

from kaldi_tflite.lib.layers import PLDA
from kaldi_tflite.lib.testdata import RefPldaModel, RefXVectors, RefPldaScores

# RMSE tolerance. In Kaldi, we PLDA weights are stored as double (float64)
# values. But if we want to us tflite, we need to use float32 values. This
# increases the RMSE between kaldi and tflite a little bit but is still within
# this limit which seems reasonable enough.
tolerance = 2e-4


class TestPLDALayer(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def getPldaLayer(self):
        return PLDA(
            RefPldaModel.dim, RefPldaModel.mean, RefPldaModel.transformMat, RefPldaModel.psi,
            return_transformed=True, normalize_length=True, simple_length_norm=False,
            dtype=np.float32,
        )

    def test_WithoutPCA(self):
        plda = self.getPldaLayer()
        inputs = RefXVectors.pldaInput()
        scores, transformed = plda(inputs)

        refTransformed = RefXVectors.pldaTransformed(withoutPCA=True)
        refScores = RefPldaScores.scores(withoutPCA=True)

        # Expecting shapes of reference and layer output to be the same.
        self.assertEqual(refTransformed.shape, transformed.shape)
        self.assertEqual(refScores.shape, scores.shape)

        # Expecting difference between reference and layer output to be within reference.
        err = self.rmse(refTransformed, transformed)
        self.assertTrue(err <= tolerance, f"rmse={err}")

        err = self.rmse(refScores, scores)
        self.assertTrue( err <= tolerance, f"rmse={err}")


if __name__ == "__main__":
    unittest.main()
