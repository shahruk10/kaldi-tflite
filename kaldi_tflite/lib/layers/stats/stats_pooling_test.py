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



import unittest
import numpy as np

from kaldi_tflite.lib.layers import StatsPooling
from kaldi_tflite.lib.testdata import RefStatsPooling

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
