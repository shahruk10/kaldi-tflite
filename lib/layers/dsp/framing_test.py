#!/usr/bin/env python3

import unittest
import numpy as np

from lib.kaldi import MirrorPad, ExtractFrames
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
            framer = Framing(frameSizeMs, frameShiftMs, sampleFreq, True)

            N = int(10 * sampleFreq)  # 10 seconds worth of samples
            x = np.arange(0, N)

            with self.subTest(cfg=cfg, snip_edges=True):
                want = ExtractFrames(x, frameSizeMs, frameShiftMs, sampleFreq, True)
                got = framer(x).numpy()
                self.compareFrames(want, got)

            with self.subTest(cfg=cfg, snip_edges=False):
                # Amount of padding on the left and right for input sample array.
                pad = int((frameSizeMs - frameShiftMs) / 1000.0 * sampleFreq) // 2
                x = MirrorPad(x, pad)

                want = ExtractFrames(x, frameSizeMs, frameShiftMs, sampleFreq, True)
                got = framer(x).numpy()
                self.compareFrames(want, got)


if __name__ == "__main__":
    unittest.main()
