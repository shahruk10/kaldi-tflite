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

from kaldi_tflite.lib.io import ReadKaldiArray


class RefIvectorExtractor():
    basePath = os.path.dirname(__file__)

    @classmethod
    def getModel(cls, testName):
        return os.path.join(cls.basePath, "src", "dummy_ie_models", testName, "final.ie")

    @classmethod
    def getParams(cls, testName):

        mdlPath = os.path.join(cls.basePath, "src", "dummy_ie_models", testName)

        params = {}
        params["M"] = ReadKaldiArray(os.path.join(mdlPath, "M.mat.txt"), binary=False)

        # sigma_inv.mat.txt contains the lower triangular part of a diagonal matrix.
        with open(os.path.join(mdlPath, "sigma_inv.mat.txt"), 'r') as f:
            # Stripping lines and skipping first "["
            lines = [l.strip() for l in f.readlines()]
            lines[:] = lines[1:]

            # Reading values into full matrix.
            rows = len(lines)
            params["sigmaInv"] = np.zeros((rows, rows), dtype=np.float64)
            for i, l in enumerate(lines):
                values = l.split()
                if values[-1] == "]":
                    values[:] = values[:-1]
                for j, v in enumerate(values):
                    params["sigmaInv"][i][j] = float(v)

        with open(os.path.join(mdlPath, "test_params.txt"), 'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue

                key, val = line.split("=")
                if key in ["numGauss", "featDim", "ivecDim"]:
                    val = int(val)
                elif key in ["priorOffset"]:
                    val = float(val)

                params[key] = val

        return params
