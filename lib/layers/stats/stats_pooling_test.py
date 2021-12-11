#!/usr/bin/env python3

import unittest
import numpy as np

from lib.layers import StatsPooling
from lib.testdata import RefStatsPooling

tolerance = 4e-6


class TestStatsPoolingLayer(unittest.TestCase):

    def calcErr(self, ref, val):
        return np.sqrt(np.mean(np.power(ref - val, 2.0)))

    def defaultLayerCfg(self):
        return {
            "left_context": 0,
            "right_context": 16,
            "input_period": 1,
            "output_period": 1,
            "include_std": True,
            "padding": "SAME",
            "reduce": False,
            "epsilon": 1e-10,
            "reduce_time_axis": False,
            "name": "stats",
        }

    def test_StatsPoolingReduce(self):
        cfg = self.defaultLayerCfg()
        cfg.update({"reduce_time_axis": True})
        stats = StatsPooling(**cfg)

        name = "stats_mean_std"
        inputs = RefStatsPooling.getInputs(name)
        wantOutputs = RefStatsPooling.getOutputs(name)[:, 0:1, :]
        gotOutputs = stats(inputs)

        err = self.calcErr(wantOutputs, gotOutputs)
        self.assertTrue(err <= tolerance, f"err={err}, tolerance={tolerance}")

    def test_StatsPooling(self):

        # Default config overrides.
        configs = {
            "stats_mean": {"include_std": False},
            "stats_mean_std": {},
            "stats_mean_std_windowed": {"right_context": 4},
            "stats_mean_std_only_left_context": {"left_context": -4, "right_context": 0},
            "stats_mean_std_both_left_right_context": {"left_context": -4, "right_context": 4},
            "stats_mean_std_asymmetrical_context": {"left_context": -4, "right_context": 2},
            "stats_mean_std_subsampling": {"input_period": 4, "output_period": 4},
            "stats_mean_std_windowed_subsampling": {
                "left_context": -4, "right_context": 4, "input_period": 4, "output_period": 4,
            },
        }

        for name, overrides in configs.items():
            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultLayerCfg()
                cfg.update(overrides)
                stats = StatsPooling(**cfg)

                inputs = RefStatsPooling.getInputs(name)
                wantOutputs = RefStatsPooling.getOutputs(name)
                gotOutputs = stats(inputs)

                err = self.calcErr(wantOutputs, gotOutputs)
                self.assertTrue(err <= tolerance, f"err={err}, tolerance={tolerance}")


if __name__ == "__main__":
    unittest.main()
