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


import os
import unittest
import numpy as np
from tempfile import NamedTemporaryFile, TemporaryDirectory

import tensorflow as tf
from kaldi_tflite.lib.models import SavedModel2TFLite
from kaldi_tflite.lib.models import XvectorExtractorFromConfig
from kaldi_tflite.lib.testdata import RefXVectorsE2E

# Absolute difference in Cosine Similarity score.
tolerance = 0.075


class TestKaldiXvectorExtractor(unittest.TestCase):

    def calcErr(self, want, got):
        """
        Calculates (1.0 - cosine similarity)
        """
        dot = np.sum(want.flatten() * got.flatten())
        norm = np.linalg.norm(want) * np.linalg.norm(got)
        cos = np.divide(dot, norm)
        return 1.0 - cos

    def checkInference(
        self,
        mdl: tf.keras.Model,
        tfliteMdlPath: str,
        inputs: np.ndarray,
        wantOutputs: np.ndarray,
    ):
        # Checking inference using regular Tensorflow model.
        gotOutputs = mdl(inputs).numpy()
        err = self.calcErr(wantOutputs, gotOutputs)
        self.assertTrue(err <= tolerance, f"TF Model (1-cosine_similarity)={err:.6f}, tolerance={tolerance:.6f}")

        # Checking inference using TFLite model.
        ip = tf.lite.Interpreter(model_path=tfliteMdlPath)
        inputIdx = ip.get_input_details()[0]['index']
        outputIdx = ip.get_output_details()[0]['index']

        ip.resize_tensor_input(inputIdx, inputs.shape)
        ip.allocate_tensors()
        ip.set_tensor(inputIdx, inputs)
        ip.invoke()
        gotOutputs = ip.get_tensor(outputIdx)

        err = self.calcErr(wantOutputs, gotOutputs)
        self.assertTrue(err <= tolerance, f"TFLite Model (1-cosine_similarity)={err:.6f}, tolerance={tolerance:.6f}")

    def test_ConvertTFLiteAndInference(self):

        kaldiMdlDir = "data/kaldi_models"
        cfgDir = "data/tflite_models"

        # Names of kaldi models to test.
        modelNames = ["0008_sitw_v2_1a"]

        for name in modelNames:
            cfgPath = os.path.join(cfgDir, f"{name}.yml")

            with self.subTest(model=cfgPath):
                # Building model.
                mdl = XvectorExtractorFromConfig(cfgPath)

                # Saving model and converting to TF Lite.
                with TemporaryDirectory() as mdlPath, \
                        NamedTemporaryFile(suffix='.tflite') as tflitePath:
                    mdl.save(mdlPath)
                    SavedModel2TFLite(mdlPath, tflitePath.name, optimize=True)

                    # Loading reference inputs and outputs.
                    refInput = RefXVectorsE2E.getInputs(name)
                    refOutput = RefXVectorsE2E.getOutputs(name)

                    # Testing inference.
                    self.checkInference(mdl, tflitePath.name, refInput, refOutput)


if __name__ == "__main__":
    unittest.main()
