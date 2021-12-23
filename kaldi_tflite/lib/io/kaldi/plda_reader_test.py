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

from kaldi_tflite.lib.io import KaldiPldaReader
from kaldi_tflite.lib.testdata import RefPldaModel


tolerance = 1e-9


class TestKaldiPldaReader(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def test_Read(self):
        # Loading PLDA model.
        plda = KaldiPldaReader("kaldi_tflite/lib/testdata/plda/plda", True)

        # Expecting shapes of parsed data to be the same as reference.
        self.assertEqual(RefPldaModel.mean.shape, plda.mean.shape)
        self.assertEqual(RefPldaModel.psi.shape, plda.psi.shape)
        self.assertEqual(RefPldaModel.transformMat.shape, plda.transformMat.shape)

        # Expecting difference between parsed and reference values to be within reference.
        self.assertTrue(self.rmse(RefPldaModel.mean, plda.mean) <= tolerance)
        self.assertTrue(self.rmse(RefPldaModel.psi, plda.psi) <= tolerance)
        self.assertTrue(self.rmse(RefPldaModel.transformMat, plda.transformMat) <= tolerance)


if __name__ == "__main__":
    unittest.main()
