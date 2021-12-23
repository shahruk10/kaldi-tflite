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

from kaldi_tflite.lib.io import KaldiNnet3Reader
from kaldi_tflite.lib.testdata import RefTdnnNarrow

tolerance = 1e-7


class TestKaldiNnet3Reader(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def test_ReadNnet3Model(self):
        # Loading nnet3 model.
        nnet3 = KaldiNnet3Reader("kaldi_tflite/lib/testdata/tdnn/src/tdnn_narrow/final.raw", True)

        # Expecting config lines of parsed model to be the same as reference.
        self.assertEqual(len(RefTdnnNarrow.config), len(nnet3.config))
        for wantLine, gotLine in zip(RefTdnnNarrow.config, nnet3.config):
            self.assertEqual(wantLine, gotLine)

        # Expecting to get the same components as parsed parameters as the reference.
        self.assertEqual(len(RefTdnnNarrow.components), len(nnet3.components))
        for want, got in zip(RefTdnnNarrow.components, nnet3.components):
            self.compareComponent(want, got)

    def compareComponent(self, want, got):
        # Checking all expected keys were parsed.
        self.assertEqual(
            len(want.keys()), len(got.keys()), f"component={want['name']}",
        )

        # Checking the values were parsed correctly.
        for key in want.keys():
            self.assertTrue(
                key in got.keys(), f"component={want['name']}, key={key}",
            )

            if isinstance(want[key], np.ndarray):
                if len(want[key]) > 0 or len(got[key]) > 0:
                    err = self.rmse(want[key], got[key])
                    self.assertTrue(
                        err <= tolerance, f"component={want['name']}, key={key}, rmse={err}, got={got[key]}",
                    )
            elif isinstance(want[key], float):
                err = self.rmse(want[key], got[key])
                self.assertTrue(
                    err <= tolerance, f"component={want['name']}, key={key}, rmse={err}, got={got[key]}"
                )
            else:
                self.assertEqual(
                    want[key], got[key], f"component={want['name']}, key={key}",
                )


if __name__ == "__main__":
    unittest.main()
