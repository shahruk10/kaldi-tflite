#!/usr/bin/env python3

import unittest
import numpy as np

from typing import Tuple

from tensorflow.keras.layers import Input, ReLU
from tensorflow.keras.models import Model

from lib.layers import TDNN, BatchNorm
from lib.testdata import RefTdnnSingleLayer, RefTdnnNarrow

tolerance = 1e-6


class TestTDNNLayer(unittest.TestCase):

    def rmse(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def getTdnnSingleLayer(self, input_shape: Tuple[int, int]) -> TDNN:
        tdnn = TDNN.from_config(RefTdnnSingleLayer.cfg)
        tdnn.build(input_shape)
        tdnn.set_weights(RefTdnnSingleLayer.weights())
        return tdnn

    def test_SingleLayer(self):
        inputs = RefTdnnSingleLayer.inputs
        wantOutputs = RefTdnnSingleLayer.outputs

        tdnn = self.getTdnnSingleLayer(inputs.shape)
        gotOutputs = tdnn(inputs)

        # Expecting shapes of reference and layer output to be the same.
        self.assertEqual(wantOutputs.shape, gotOutputs.shape)

        # Expecting difference between reference and layer output to be within reference.
        err = self.rmse(wantOutputs, gotOutputs)
        self.assertTrue(err <= tolerance, f"rmse={err}")

    def getTdnnNarrow(self) -> Model:
        """
        Implements the tdnn_narrow test model as a tensorflow function model.
        See lib/testdata/tdnn/src/make_tdnn.sh for the model specification.
        """

        def tdnn_relu_bn(x, name: str, dim: int, context: list, relu: bool, bn: bool):
            x = TDNN(dim, context=context, name=f"{name}.affine")(x)
            if relu:
                x = ReLU(name=f"{name}.relu")(x)
            if bn:
                x = BatchNorm(name=f"{name}.batchnorm")(x)
            return x

        # Instantiating model.
        layers = [
            ["tdnn1", 5, [-2, -1, 0, 1, 2], True, True],
            ["tdnn2", 8, [-2, 0, 2], True, True],
            ["tdnn3", 8, [-3, 0, 3], True, True],
            ["tdnn4", 8, [0], True, True],
            ["tdnn5", 8, [0], True, True],
            ["output", 1, [0], False, False],
        ]

        inputShape = (None, 3)

        x = Input(inputShape)
        y = x
        for cfg in layers:
            y = tdnn_relu_bn(y, *cfg)

        mdl = Model(inputs=x, outputs=y)
        mdl.build(inputShape)

        # Initializing layer weights from pretrained model.
        for c in RefTdnnNarrow.components:
            try:
                l = mdl.get_layer(c["name"])
            except:
                print(f"no layer with name = {c['name']}, not initializing")
                continue

            l.set_weights(RefTdnnNarrow.weights(c["name"]))

        return mdl

    def test_TdnnNarrow(self):
        # TODO (shahruk): regenerate better test data for this.
        inputs = RefTdnnNarrow.inputs
        wantOutputs = RefTdnnNarrow.outputs

        tdnn = self.getTdnnNarrow()
        gotOutputs = tdnn(inputs, training=False)

        # Expecting shapes of reference and layer output to be the same.
        self.assertEqual(wantOutputs.shape, gotOutputs.shape)

        # Expecting difference between reference and layer output to be within reference.
        tolerance = 5e-4
        err = self.rmse(wantOutputs, gotOutputs)
        self.assertTrue(err <= tolerance, f"rmse={err}")


if __name__ == "__main__":
    unittest.main()
