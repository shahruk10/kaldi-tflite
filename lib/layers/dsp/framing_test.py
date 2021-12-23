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

from lib.kaldi_numpy import PadWaveform, ExtractFrames
from lib.layers import Framing


class TestFramingLayer(unittest.TestCase):

    def compareFrames(self, want, got):

        self.assertTrue(
            want.size > 0 and got.size > 0,
            f"received empty arrays to compare, got.shape={got.shape}, want.shape={want.shape}",
        )
        self.assertTrue(
            np.array_equal(want, got),
            f"frames does not match reference, got.shape={got.shape}, want.shape={want.shape}",
        )

    def test_Framing(self):

        configs = [
            {"frame_length_ms": 25.0, "frame_shift_ms": 10.0, "sample_frequency": 8000.0},
            {"frame_length_ms": 25.0, "frame_shift_ms": 10.0, "sample_frequency": 16000.0},
            {"frame_length_ms": 32.0, "frame_shift_ms": 16.0, "sample_frequency": 16000.0},
            {"frame_length_ms": 32.0, "frame_shift_ms": 32.0, "sample_frequency": 16000.0},
            {"frame_length_ms": 32.0, "frame_shift_ms": 64.0, "sample_frequency": 16000.0},
            {"frame_length_ms": 2000.0, "frame_shift_ms": 1000.0, "sample_frequency": 16000.0},
        ]

        for cfg in configs:
            frameSizeMs, frameShiftMs, sampleFreq = cfg.values()

            N = int(10 * sampleFreq)  # 10 seconds worth of samples
            m = int(sampleFreq * frameSizeMs / 1000.0)   # Frame size as # of samples.
            k = int(sampleFreq * frameShiftMs / 1000.0)  # Frame shift as # of samples.

            x = np.arange(0, N)

            with self.subTest(cfg=cfg, snip_edges=True):
                framer = Framing(frameSizeMs, frameShiftMs, sampleFreq)

                want = ExtractFrames(x, frameSizeMs, frameShiftMs, sampleFreq, True)
                got = framer(x).numpy()
                self.compareFrames(want, got)

            with self.subTest(cfg=cfg, snip_edges=False):
                framer = Framing(frameSizeMs, frameShiftMs, sampleFreq)

                xPadded = PadWaveform(x, m, k)
                want = ExtractFrames(xPadded, frameSizeMs, frameShiftMs, sampleFreq, False)
                got = framer(xPadded).numpy()
                self.compareFrames(want, got)


if __name__ == "__main__":
    unittest.main()
