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
from kaldi_tflite.lib.layers import Framing, CMVN
from kaldi_tflite.lib.testdata import RefCMVN

tolerance = 1e-5


class TestCMVNLayer(unittest.TestCase):

    def compareFeats(self, want, got, expectEmpty=False, toleranceAdj=0):

        if expectEmpty:
            self.assertTrue(
                want.size == 0 and got.size == 0,
                f"expected empty arrays, got.shape={got.shape}, want.shape={want.shape}",
            )
            return

        self.assertTrue(
            want.size > 0 and got.size > 0,
            f"received empty arrays to compare, got.shape={got.shape}, want.shape={want.shape}",
        )

        self.assertTrue(
            want.shape == got.shape,
            f"reference feature shape ({want.shape}) does not match shape got ({got.shape})",
        )

        rmse = np.sqrt(np.mean(np.power(want - got, 2)))
        self.assertTrue(
            rmse < tolerance + toleranceAdj,
            f"features does not match reference, rmse={rmse}",
        )

    def defaultCfg(self):
        return {
            "input_shape": (1, 16000 * 3, ),
            "framing": {
                "frame_length_ms": 25.0,
                "frame_shift_ms": 10.0,
                "sample_frequency": 16000.0,
                "dynamic_input_shape": False,
            },
            "cmvn": {
                "center": True,
                "norm_vars": False,
                "window": 600,
                "min_window": 100,
                "padding": "SAME",
            }
        }

    def checkTFLiteInference(
        self, interpreter: tf.lite.Interpreter, x: np.ndarray, wantFrames: int,
    ):
        inputLayerIdx = interpreter.get_input_details()[0]['index']
        outputLayerIdx = interpreter.get_output_details()[0]['index']

        interpreter.allocate_tensors()
        interpreter.set_tensor(inputLayerIdx, x)
        interpreter.invoke()
        y = interpreter.get_tensor(outputLayerIdx)
        gotFrames = y.shape[-2]

        self.assertTrue(
            gotFrames == wantFrames,
            f"output number of frames ({gotFrames}) does not match expected ({wantFrames})",
        )

    def test_ConvertTFLite(self):

        tests = {
            "default": {
                "want_frames": 298,
                "cmvn": {}
            },
            "without_norm_var": {
                "want_frames": 298,
                "cmvn": {"norm_vars": False},
            },
            "norm_var": {
                "want_frames": 298,
                "cmvn": {"norm_vars": True},
            },
            "even_window": {
                "want_frames": 298,
                "cmvn": {"norm_vars": True, "window": 200},
            },
            "odd_window": {
                "want_frames": 298,
                "cmvn": {"norm_vars": True, "window": 201},
            },
            "valid": {
                "want_frames": 99,
                "cmvn": {"padding": "VALID", "window": 200},
            },
            "valid_norm_var": {
                "want_frames": 99,
                "cmvn": {"padding": "VALID", "window": 200, "norm_vars": True},
            },
        }

        for name, overrides in tests.items():
            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultCfg()
                cfg["cmvn"].update(overrides["cmvn"])

                # Creating Filter Bank extraction model.
                mdl = Sequential([
                    Input(batch_shape=cfg["input_shape"]),
                    Framing(**cfg["framing"]),
                    CMVN(**cfg["cmvn"]),
                ])

                # Saving model and converting to TF Lite.
                with TemporaryDirectory() as mdlPath, \
                        NamedTemporaryFile(suffix='.tflite') as tflitePath:
                    mdl.save(mdlPath)
                    SavedModel2TFLite(mdlPath, tflitePath.name)

                    # Testing if inference works.
                    interpreter = tf.lite.Interpreter(model_path=tflitePath.name)
                    x = np.random.random(cfg["input_shape"]).astype(np.float32)
                    self.checkTFLiteInference(interpreter, x, overrides["want_frames"])

    # @unittest.SkipTest
    def test_CMVN(self):

        # Default config overrides
        testNames = [f"16000_001_{i:03d}" for i in range(1, 9)]

        for name in testNames:

            overrides = RefCMVN.getConfig(name)
            feat = RefCMVN.getInputs(name)
            want = RefCMVN.getOutputs(name)
            numFrames = want.shape[-2]

            with self.subTest(name=name, overrides=overrides, padding="SAME"):
                cfg = self.defaultCfg()
                cfg["cmvn"] = overrides["cmvn"]
                cfg["cmvn"]["padding"] = "SAME"

                # Creating CMVN layer and evaluating output.
                cmvn = CMVN(**cfg["cmvn"])
                got = cmvn(feat).numpy()

                self.compareFeats(want, got)

            with self.subTest(name=name, overrides=overrides, padding="VALID"):
                cfg = self.defaultCfg()
                cfg["cmvn"] = overrides["cmvn"]
                cfg["cmvn"]["padding"] = "VALID"

                # Creating CMVN layer and evaluating output.
                cmvn = CMVN(**cfg["cmvn"])
                got = cmvn(feat).numpy()

                # Trimming reference output to expected frames for "VALID" padding setting.
                N = cfg["cmvn"]["window"]
                a = N // 2
                b = numFrames - (N - 1) // 2
                wantValid = want[..., a:b, :]

                expectEmpty = False
                if N > numFrames:
                    expectEmpty = True

                self.compareFeats(wantValid, got, expectEmpty=expectEmpty)


if __name__ == "__main__":
    unittest.main()
