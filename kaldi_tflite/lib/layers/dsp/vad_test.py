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
from tempfile import NamedTemporaryFile, TemporaryDirectory

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

from kaldi_tflite.lib.models import SavedModel2TFLite
from kaldi_tflite.lib.layers import VAD
from kaldi_tflite.lib.testdata import RefVAD

tolerance = 1e-5


class TestVADLayer(unittest.TestCase):

    def compareMasks(self, want: np.ndarray, got: np.ndarray):

        self.assertTrue(
            want.size > 0 and got.size > 0,
            f"received empty arrays to compare, got.shape={got.shape}, want.shape={want.shape}",
        )

        self.assertTrue(
            want.shape == got.shape,
            f"reference vad output shape ({want.shape}) does not match shape got ({got.shape})",
        )

        wantActive = np.sum(want)
        gotActive = np.sum(got)
        self.assertTrue(
            np.array_equal(want, got),
            f"vad output ({gotActive} active) does not match reference ({wantActive} active)",
        )

    def defaultCfg(self):
        return {
            "input_shape": (1, 298, 23),
            "vad": {
                "energy_mean_scale": 0.5,
                "energy_threshold": 5,
                "frames_context": 0,
                "proportion_threshold": 0.6,
                "return_indexes": False,
                "energy_coeff": 0,
            },
        }

    def checkTFLiteInference(
        self, interpreter: tf.lite.Interpreter, x: np.ndarray, ref: np.ndarray, expectIndexes: bool,
    ):
        inputDetails = interpreter.get_input_details()[0]
        inputLayerIdx = inputDetails['index']
        outputLayerIdx = interpreter.get_output_details()[0]['index']

        if not np.array_equal(inputDetails["shape"], x.shape):
            interpreter.resize_tensor_input(inputLayerIdx, x.shape)

        interpreter.allocate_tensors()
        interpreter.set_tensor(inputLayerIdx, x)
        interpreter.invoke()
        y = interpreter.get_tensor(outputLayerIdx)

        if not expectIndexes:
            self.compareMasks(ref, y)
        else:
            yMask = np.zeros(x.shape[:-1] + (1,))
            yMask[y[:, 0], y[:, 1], :] = 1
            self.compareMasks(ref, yMask)

    def test_ConvertTFLite(self):

        # Arbitariliy chosen reference file; this one has a good mix of active /
        # inactive frames in the reference output.
        refName = "16000_001_025"

        tests = {
            "default": {
                "input_shape": [None, 23],
                "vad": {},
            },
            "return_indexes": {
                "input_shape": [None, 23],
                "vad": {"return_indexes": True},
            },
        }

        for name, overrides in tests.items():
            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultCfg()
                cfg["input_shape"] = overrides["input_shape"]
                cfg["vad"].update(RefVAD.getConfig(refName)["vad"])
                cfg["vad"].update(overrides["vad"])

                # Creating VAD model.
                mdl = Sequential([
                    Input(cfg["input_shape"]),
                    VAD(**cfg["vad"]),
                ])

                # Saving model and converting to TF Lite.
                with TemporaryDirectory() as mdlPath, \
                        NamedTemporaryFile(suffix='.tflite') as tflitePath:
                    mdl.save(mdlPath)
                    SavedModel2TFLite(mdlPath, tflitePath.name)

                    # Testing if inference works.
                    interpreter = tf.lite.Interpreter(model_path=tflitePath.name)
                    x = RefVAD.getInputs(refName)
                    y = RefVAD.getOutputs(refName)
                    self.checkTFLiteInference(interpreter, x, y, cfg["vad"]["return_indexes"])

    def test_VAD(self):

        # Default config overrides
        testNames = [f"16000_001_{i:03d}" for i in range(1, 47)]

        for name in testNames:

            overrides = RefVAD.getConfig(name)
            feat = RefVAD.getInputs(name)
            want = RefVAD.getOutputs(name)
            numFrames = want.shape[-2]

            with self.subTest(name=name, overrides=overrides, return_indexes=False):
                cfg = self.defaultCfg()
                cfg["vad"] = overrides["vad"]

                # Creating VAD layer and evaluating output.
                vad = VAD(**cfg["vad"])
                got = vad(feat).numpy()

                self.compareMasks(want, got)


if __name__ == "__main__":
    unittest.main()
