#!/usr/bin/env python3

import unittest
import numpy as np

from lib.kaldi import PadWaveform, ExtractFrames
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
