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
import numpy as np
import librosa

from kaldi_tflite.lib.testdata import KaldiTestDataReader


class RefXVectorsE2E(KaldiTestDataReader):
    basePath = os.path.dirname(__file__)

    @classmethod
    def getInputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "audio.wav")
        samples, _ = librosa.load(inputFile, sr=None)
        samples = (samples * 32767.0).astype(np.int16).astype(np.float32)
        return samples.reshape(1, -1)

    @classmethod
    def getOutputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "xvector.ark.txt")
        return np.stack(list(cls.loadKaldiArk(inputFile).values()), axis=0)
