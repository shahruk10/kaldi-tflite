#!/usr/bin/env python3

import unittest
import numpy as np

from lib.layers import PLDA
from lib.testdata import RefPldaModel, RefXVectors, RefPldaScores

tolerance = 1e-5


class TestPLDALayer(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def getPldaLayer(self):
        return PLDA(
            RefPldaModel.dim, RefPldaModel.mean, RefPldaModel.transformMat, RefPldaModel.psi,
        )

    def test_WithoutPCA(self):
        plda = self.getPldaLayer()
        inputs = RefXVectors.pldaInput()
        scores, transformed = plda(inputs, return_transformed=True)

        refTransformed = RefXVectors.pldaTransformed(withoutPCA=True)
        refScores = RefPldaScores.scores(withoutPCA=True)

        # Expecting shapes of reference and layer output to be the same.
        self.assertEqual(refTransformed.shape, transformed.shape)
        self.assertEqual(refScores.shape, scores.shape)

        # Expecting difference between reference and layer output to be within reference.
        self.assertTrue(self.rmse(refTransformed, transformed) <= tolerance)
        self.assertTrue(self.rmse(refScores, scores) <= tolerance)


if __name__ == "__main__":
    unittest.main()
