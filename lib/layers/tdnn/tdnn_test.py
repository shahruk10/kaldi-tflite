#!/usr/bin/env python3

import unittest
import numpy as np

from lib.layers import TDNN
from lib.testdata import RefTdnnSingleLayer

tolerance = 1e-6


class TestTDNNLayer(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def getTdnnLayer(self, input_shape):        
        tdnn = TDNN.from_config(RefTdnnSingleLayer.cfg)
        tdnn.build(input_shape)
        tdnn.set_weights(RefTdnnSingleLayer.weights())
        return tdnn

    def test_Output(self):
        inputs = RefTdnnSingleLayer.inputs
        wantOutputs = RefTdnnSingleLayer.outputs

        tdnn = self.getTdnnLayer(inputs.shape)
        gotOutputs = tdnn(inputs)

        # Expecting shapes of reference and layer output to be the same.
        self.assertEqual(wantOutputs.shape, gotOutputs.shape)

        # Expecting difference between reference and layer output to be within reference.
        err = self.rmse(wantOutputs, gotOutputs)
        self.assertTrue(err <= tolerance, f"rmse={err}")

if __name__ == "__main__":
    unittest.main()
