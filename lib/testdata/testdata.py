#!/usr/bin/env python3

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

                # Start of a new entry in the ark.
                if "[" in tokens:
                    curID = tokens[0]
                    curMat = []
                    continue

                # End of current entry.
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
