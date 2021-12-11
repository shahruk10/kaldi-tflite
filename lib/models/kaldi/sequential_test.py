#!/usr/bin/env python3

import os
import unittest
import numpy as np
import yaml

from lib.models.kaldi import SequentialModel, downloadModel
from lib.testdata import RefKaldiXVectorModels

tolerance = 1.25e-3


class TestKaldiSequentialModel(unittest.TestCase):

    def calcErr(self, want, got):
        """
        Calculates (1.0 - cosine distance)
        """
        dot = np.sum(want.flatten() * got.flatten())
        norm = np.linalg.norm(want) * np.linalg.norm(got)
        cos = np.divide(dot, norm)
        return 1.0 - cos

    def test_Sequential(self):

        kaldiMdlDir = "data/kaldi_models"
        cfgDir = os.path.join(kaldiMdlDir, "configs")

        # Model name and output layer.
        models = [
            ["0008_sitw_v2_1a", "tdnn6.affine"],
            ["0006_callhome_diarization_v2_1a", "tdnn6.affine"],
        ]

        for (model, outputLayer) in models:
            with self.subTest(model=model):
                # Loading config file.
                cfgFile = os.path.join(cfgDir, f"{model}.yml")
                with open(cfgFile) as f:
                    cfg = yaml.safe_load(f)

                # Checking if weights are present.
                mdlDir = os.path.join(kaldiMdlDir, model)
                weights = os.path.join(mdlDir, "exp", "xvector_nnet_1a", "final.raw")

                if not os.path.exists(weights):
                    # Try to download the models.
                    downloadCfg = cfg.get("download", None)
                    if downloadCfg is not None:
                        try:
                            link, sha256 = downloadCfg["link"], downloadCfg["hash"]
                            downloadModel(link, mdlDir, sha256)
                        except Exception as err:
                            print(f"failed to download model from {link}: {err}")

                    if not os.path.exists(weights):
                        weights = None

                # Creating model.
                mdl = SequentialModel(cfg["model_config"], weights)

                if weights is None:
                    self.skipTest("model weights not found")

                # Computing outputs with reference inputs and copmaring againt reference outputs.
                inputs = RefKaldiXVectorModels.getInputs(f"{model}_{outputLayer}")
                wantOutputs = RefKaldiXVectorModels.getOutputs(f"{model}_{outputLayer}")
                gotOutputs = mdl(inputs).numpy()

                err = self.calcErr(wantOutputs, gotOutputs)
                self.assertTrue(err <= tolerance, f"(1-cosine_distance)={err}, tolerance={tolerance}")


if __name__ == "__main__":
    unittest.main()
