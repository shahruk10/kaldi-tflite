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

from kaldi_tflite.lib.io import ReadKaldiArray

tolerance = 5e-8


class TestKaldiArrayReader(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def test_ReadKaldiArray(self):
        testCases = {
            "vector": {
                "bin_path": "kaldi_tflite/lib/testdata/plda/xvectors_train_combined_200k/mean.vec",
                "txt_path": "kaldi_tflite/lib/testdata/plda/xvectors_train_combined_200k/mean.vec.txt",
            },
            "matrix": {
                "bin_path": "kaldi_tflite/lib/testdata/plda/xvectors_train_combined_200k/transform.mat",
                "txt_path": "kaldi_tflite/lib/testdata/plda/xvectors_train_combined_200k/transform.mat.txt",
            },
        }

        for name, data in testCases.items():
            with self.subTest(name=name, data=data):
                arrayBin = ReadKaldiArray(data["bin_path"], binary=True)
                arrayTxt = ReadKaldiArray(data["txt_path"], binary=False, dtype=np.float32)

                # Expecting both to be non empty.
                self.assertTrue(arrayBin.size > 0 and arrayTxt.size > 0)

                # Expecting shapes of parsed data to be the same.
                self.assertEqual(arrayBin.shape, arrayTxt.shape)

                # Expecting difference between parsed data to be within tolerance.
                rmse = self.rmse(arrayBin, arrayTxt)
                self.assertTrue(
                    rmse <= tolerance,
                    f"parsed binary and text data does not match, rmse={rmse}",
                )


if __name__ == "__main__":
    unittest.main()
