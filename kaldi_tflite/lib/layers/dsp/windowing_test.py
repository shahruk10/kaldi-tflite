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
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tempfile import NamedTemporaryFile, TemporaryDirectory

from kaldi_tflite.lib.layers import Windowing
from kaldi_tflite.lib.models import SavedModel2TFLite
from kaldi_tflite.lib.kaldi_numpy import ProcessFrames

np.random.seed(12345)


tolerance = 2e-7


class TestWindowingLayer(unittest.TestCase):

    def compareFrames(self, want, got, name, toleranceAdj=0):

        self.assertTrue(
            want.size > 0 and got.size > 0,
            f"received empty arrays to compare, got.shape={got.shape}, want.shape={want.shape}",
        )

        self.assertTrue(
            want.shape == got.shape,
            f"{name} reference shape ({want.shape}) does not match shape got ({got.shape})",
        )

        rmse = np.sqrt(np.mean(np.power(want - got, 2)))
        self.assertTrue(
            rmse < tolerance + toleranceAdj,
            f"{name} does not match reference, rmse={rmse}",
        )

    def defaultLayerCfg(self):
        return {
            "window_type": "povey",
            "blackman_coeff": 0.42,
            "dither": 0.0,
            "remove_dc_offset": True,
            "preemphasis_coefficient": 0.97,
            "raw_energy": True,
            "return_energy": True,
            "energy_floor": 0.0,
            "epsilon": np.finfo(np.float32).eps,
        }

    def checkConvertTFLite(self, layer: Windowing):

        # TODO (shahruk): With `dithering` enabled, the conversion fails because
        # the tf.RandomStandardNormal op isn't natively supported in Tensorflow
        # Lite (requires Flex Ops). Need to find a workaround that does not
        # involve needing to link to Flex lib for inference. Pre-computing
        # random dither values and storing it in the model could be one
        # solution.
        if layer.dither > 0:
            print("TF Lite conversion not supported with dithering enabled, skipping check")
            return

        i = Input((None, 256))
        mdl = Model(
            inputs=i,
            outputs=layer(i),
        )

        # Saving model and converting to TF Lite.
        with TemporaryDirectory() as mdlPath, \
                NamedTemporaryFile(suffix='.tflite') as tflitePath:
            mdl.save(mdlPath)
            SavedModel2TFLite(mdlPath, tflitePath.name)

    def test_Windowing(self):

        # Default config overrides
        configs = [
            {},
            {"window_type": "hanning"},
            {"window_type": "hamming"},
            {"window_type": "rectangular"},
            {"window_type": "sine"},
            {"window_type": "blackman"},
            {"remove_dc_offset": False},
            {"preemphasis_coefficient": 0.0},
            {"preemphasis_coefficient": 0.90},
            {"raw_energy": False},
            {"dither": 0.1},
        ]

        frames = np.random.random((1, 1000, 256))

        for overrides in configs:
            cfg = self.defaultLayerCfg()
            cfg.update(overrides)

            # Getting reference output using numpy impl.
            wantWindows, wantEnergy = ProcessFrames(
                frames,
                dither=cfg["dither"],
                remove_dc_offset=cfg["remove_dc_offset"],
                preemphasis_coefficient=cfg["preemphasis_coefficient"],
                window_type=cfg["window_type"],
                raw_energy=cfg["raw_energy"],
            )

            with self.subTest(overrides=overrides):
                windowing = Windowing(**cfg)
                gotWindows, gotEnergy = windowing(frames)

                self.compareFrames(wantWindows, gotWindows.numpy(), "processed windows", 2 * cfg["dither"])
                self.compareFrames(wantEnergy, gotEnergy.numpy(), "window energies", 2 * cfg["dither"])
                self.checkConvertTFLite(windowing)

            # Testing layer when return_energy is set to False. The processed
            # windows should be the same the as in the test above.
            overrides["return_energy"] = False
            cfg.update(overrides)
            with self.subTest(overrides=overrides):
                windowing = Windowing(**cfg)
                gotWindows = windowing(frames)

                self.compareFrames(wantWindows, gotWindows.numpy(), "processed windows", 2 * cfg["dither"])
                self.checkConvertTFLite(windowing)


if __name__ == "__main__":
    unittest.main()
