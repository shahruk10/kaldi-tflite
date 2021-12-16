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


class KaldiTestDataReader:

    basePath = os.path.dirname(__file__)

    @classmethod
    def getInputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "feat.ark.txt")
        return np.stack(list(cls.loadKaldiArk(inputFile).values()), axis=0)

    @classmethod
    def getOutputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "output.ark.txt")
        return np.stack(list(cls.loadKaldiArk(inputFile).values()), axis=0)

    @classmethod
    def loadKaldiArk(cls, path, dtype=np.float32):

        def tokens2vec(tokens):
            if dtype in [np.float32, np.float64]:
                return [float(t) for t in tokens]
            elif dtype in [np.int16, np.int32, np.int64]:
                return [int(t) for t in tokens]
            else:
                raise ValueError(f"unsupported data type: {dtype}")

        ark = {}

        curID = ""
        curMat = []
        lastRow = False
        with open(path, 'r') as f:
            for line in f:
                tokens = [ t for t in line.strip().split() if t != "" ]

                # Vector ark
                if "[" in tokens and "]" in tokens:
                    if len(tokens) > 3:
                        curID = tokens[0]
                        ark[curID] = np.array([ tokens2vec(tokens[2:-1]) ], dtype=dtype)
                    continue

                # Start of a new matrix in the ark.
                if "[" in tokens:
                    curID = tokens[0]
                    curMat = []
                    continue

                # End of current matrix.
                if "]" in tokens:
                    lastRow = True
                    if len(tokens) > 1:
                        curMat.append(tokens2vec(tokens[:-1]))
                else:
                    curMat.append(tokens2vec(tokens))

                # Appending row.
                if lastRow:
                    ark[curID] = np.array(curMat, dtype=dtype)
                    curMat = []
                    curID = ""
                    lastRow = False

        return ark
