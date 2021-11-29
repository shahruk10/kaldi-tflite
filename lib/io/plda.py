#!/usr/bin/env python3

from typing import Tuple
import numpy as np


class KaldiPldaLoader():

    def __init__(self, plda_path: str):
        self.curPos = 0
        with open(plda_path, 'rb') as fd:
            self.data = fd.read()

        self.mean, self.transformMat, self.psi = self.parse(self.data)

    def readBytes(self, nBytes: int) -> bytes:
        if self.curPos >= len(self.data):
            return []

        buf = self.data[self.curPos:self.curPos + nBytes]
        self.curPos += len(buf)

        return buf

    def parse(self, data: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.expectToken("<Plda>")
        mean = self.readKaldiVecBinary()
        transformMat = self.readKaldiMatBinary()
        psi = self.readKaldiVecBinary()
        self.expectToken("</Plda>")

        return mean, transformMat, psi

    def expectToken(self, token: str):
        tokenLen = len(token)
        for i in range(self.curPos, len(self.data) - tokenLen):
            if self.data[i:i + tokenLen].decode() == token:
                self.curPos = i + tokenLen + 1
                return

        raise ValueError(f"failed to find expected token '{token}")

    def readKaldiVecBinary(self) -> np.ndarray:
        # Header containing data type.
        header = self.readBytes(3).decode()

        if header == "FV ":
            sampleSize = 4  # float
            vecType = np.float32
        elif header == "DV ":
            sampleSize = 8  # double
            vecType = np.float64
        else:
            raise ValueError(f"unknown header for vector type '{header.encode()}'")

        assert (sampleSize > 0)

        # Header containing dimension size type.
        dimSize = self.readBytes(1).decode()
        assert dimSize == '\4'  # int32 size

        # Getting vector dimension.
        vecDim = np.frombuffer(self.readBytes(4), dtype=np.int32, count=1)[0]
        if vecDim == 0:
            return np.array([], dtype=vecType)

        # Read whole vector,
        buf = self.readBytes(vecDim * sampleSize)
        vec = np.frombuffer(buf, dtype=vecType)

        return vec

    def readKaldiMatBinary(self) -> np.ndarray:
        # Header containing data type.
        header = self.readBytes(3).decode()

        if header.startswith('CM'):
            return NotImplementedError("can't decode compressed matrix yet")
        elif header == 'FM ':
            sampleSize = 4  # float
            matType = np.float32
        elif header == 'DM ':
            sampleSize = 8  # double
            matType = np.float64
        else:
            raise ValueError(f"unknown header for matrix type '{header}'")

        assert(sampleSize > 0)

        # Getting matrix dimension.
        s1, rows, s2, cols = np.frombuffer(self.readBytes(10), dtype='int8,int32,int8,int32', count=1)[0]

        # Read whole matrix.
        buf = self.readBytes(rows * cols * sampleSize)
        mat = np.frombuffer(buf, dtype=matType).reshape(rows, cols)

        return mat
