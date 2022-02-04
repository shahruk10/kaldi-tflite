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
from kaldi_tflite.lib.layers import Framing, MFCC
from kaldi_tflite.lib.kaldi_numpy import PadWaveform
from kaldi_tflite.lib.testdata import RefMFCC

tolerance = 2.25e-4


class TestMFCCLayer(unittest.TestCase):

    def compareFeats(self, want, got, toleranceAdj=0):

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
            "snip_edges": False,
            "framing": {
                "frame_length_ms": 25.0,
                "frame_shift_ms": 10.0,
                "sample_frequency": 16000.0,
                "dynamic_input_shape": False,
            },
            "mfcc": {
                "num_mfccs": 30,
                "num_mels": 30,
                "cepstral_lifter": 22,
                "use_energy": True,
                "sample_frequency": 16000.0,
                "high_freq_cutoff": 7600.0,
                "low_freq_cutoff": 20.0,
                "use_log_fbank": True,
                "use_power": True,
                "window_type": "povey",
                "dither": 0.0,
                "remove_dc_offset": True,
                "preemphasis_coefficient": 0.97,
                "raw_energy": True,
                "energy_floor": 0.0,
                "epsilon": np.finfo(np.float32).eps,
            }
        }

    def checkTFLiteInference(
        self, interpreter: tf.lite.Interpreter, x: np.ndarray,
            wantFrames: int, wantDim: int, resize: bool,
    ):
        inputLayerIdx = interpreter.get_input_details()[0]['index']
        outputLayerIdx = interpreter.get_output_details()[0]['index']

        # Setting input size.
        if resize:
            interpreter.resize_tensor_input(inputLayerIdx, x.shape)

        interpreter.allocate_tensors()
        interpreter.set_tensor(inputLayerIdx, x)
        interpreter.invoke()
        y = interpreter.get_tensor(outputLayerIdx)

        gotFrames = y.shape[1]
        self.assertTrue(
            gotFrames == wantFrames,
            f"output number of frames ({gotFrames}) does not match expected ({wantFrames})",
        )

        gotDim = y.shape[-1]
        self.assertTrue(
            gotDim == wantDim,
            f"output feature dimension ({gotDim}) does not match expected ({wantDim})",
        )

    def test_ConvertTFLite(self):

        tests = {
            "fixed_input": {
                "input_shape": 16000 * 3,
                "inputs": [(16000 * 3, 298)],  # (numSamples, wantFrames)
                "framing": {"dynamic_input_shape": False},
            },
            "dynamic_input": {
                "input_shape": None,
                "inputs": [(16000 * 1, 98), (16000 * 3, 298), (16000 * 2, 198)],
                "framing": {"dynamic_input_shape": True},
            },
            "with_dithering_fixed_input": {
                "input_shape": 16000 * 3,
                "inputs": [(16000 * 3, 298)],  # (numSamples, wantFrames)
                "framing": {"dynamic_input_shape": False},
                "mfcc": {"dither": 1.0},
            },
            "with_dithering_dynamic_input": {
                "input_shape": None,
                "inputs": [(16000 * 1, 98), (16000 * 3, 298), (16000 * 2, 198)],
                "framing": {"dynamic_input_shape": True},
                "mfcc": {"dither": 1.0},
            },
        }

        for name, overrides in tests.items():
            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultCfg()
                cfg["framing"].update(overrides.get("framing", {}))
                cfg["mfcc"].update(overrides.get("mfcc", {}))

                # Creating Filter Bank extraction model.
                mdl = Sequential([
                    Input((overrides["input_shape"], )),
                    Framing(**cfg["framing"]),
                    MFCC(**cfg["mfcc"]),
                ])

                wantDim = cfg["mfcc"]["num_mfccs"]
                resize = cfg["framing"]["dynamic_input_shape"]

                # Saving model and converting to TF Lite.
                with TemporaryDirectory() as mdlPath, \
                        NamedTemporaryFile(suffix='.tflite') as tflitePath:
                    mdl.save(mdlPath)
                    SavedModel2TFLite(mdlPath, tflitePath.name)

                    # Testing if inference works.
                    interpreter = tf.lite.Interpreter(model_path=tflitePath.name)
                    for numSamples, wantFrames in overrides["inputs"]:
                        with self.subTest(name=name, num_samples=numSamples):
                            x = np.random.random((1, numSamples)).astype(np.float32)
                            self.checkTFLiteInference(interpreter, x, wantFrames, wantDim, resize)

    def test_MFCC(self):

        # Default config overrides
        testNames = [f"16000_{i:03d}" for i in range(1, 49)]

        for name in testNames:

            overrides = RefMFCC.getConfig(name)
            samples = RefMFCC.getInputs(name)
            want = RefMFCC.getOutputs(name)

            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultCfg()
                cfg["snip_edges"] = overrides["snip_edges"]
                cfg["framing"].update(overrides["framing"])
                cfg["mfcc"].update(overrides["mfcc"])

                frameSize = cfg["framing"]["frame_length_ms"]
                frameShift = cfg["framing"]["frame_shift_ms"]
                sampleFreq = cfg["framing"]["sample_frequency"]

                # Mirror padding input samples.
                if not cfg["snip_edges"]:
                    m = int(frameSize / 1000.0 * sampleFreq)
                    k = int(frameShift / 1000.0 * sampleFreq)
                    samples = PadWaveform(samples, m, k)

                # Creating MFCC extraction model.
                mdl = Sequential([
                    Input((samples.shape[-1], )),
                    Framing(**cfg["framing"]),
                    MFCC(**cfg["mfcc"]),
                ])

                got = mdl(samples).numpy()
                self.compareFeats(want, got)


if __name__ == "__main__":
    unittest.main()
