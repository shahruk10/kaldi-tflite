#!/usr/bin/env python3

import unittest
import numpy as np

from lib.io import KaldiPldaLoader
from lib.testdata import RefPldaModel


tolerance = 1e-9


class TestKaldiPLDALoader(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def test_Read(self):
        # Loading PLDA model.
        plda = KaldiPldaLoader("lib/testdata/plda")

        # Expecting shapes of parsed data to be the same as reference.
        self.assertEqual(RefPldaModel.mean.shape, plda.mean.shape)
        self.assertEqual(RefPldaModel.psi.shape, plda.psi.shape)
        self.assertEqual(RefPldaModel.transformMat.shape, plda.transformMat.shape)

        # Expecting difference between parsed and reference values to be within reference.
        self.assertTrue(self.rmse(RefPldaModel.mean, plda.mean) <= tolerance)
        self.assertTrue(self.rmse(RefPldaModel.psi, plda.psi) <= tolerance)
        self.assertTrue(self.rmse(RefPldaModel.transformMat, plda.transformMat) <= tolerance)


if __name__ == "__main__":
    unittest.main()
