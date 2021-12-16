#!/usr/bin/env python3

import unittest
import numpy as np
from tempfile import NamedTemporaryFile, TemporaryDirectory

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential

from lib.models import SavedModel2TFLite
from lib.layers import Framing, MFCC
from lib.kaldi import PadWaveform
from lib.testdata import RefMFCC

tolerance = 2.25e-4


class TestMFCCLayer(unittest.TestCase):

    def compareFeats(self, want, got, toleranceAdj=0):

        self.assertTrue(
            want.size > 0 and got.size > 0,
            f"received empty arrays to compare, got.shape={got.shape}, want.shape={want.shape}",
        )

        self.assertTrue(
            want.shape == got.shape,
            f"reference feature shape ({want.shape}) does not match shape got ({got.shape})",
        )

        rmse = np.sqrt(np.mean(np.power(want - got, 2)))
        self.assertTrue(
            rmse < tolerance + toleranceAdj,
            f"features does not match reference, rmse={rmse}",
        )

    def defaultCfg(self):
        return {
            "snip_edges": False,
            "framing": {
                "frame_length_ms": 25.0,
                "frame_shift_ms": 10.0,
                "sample_frequency": 16000.0,
            },
            "mfcc": {
                "num_mfccs": 30,
                "num_mels": 30,
                "cepstral_lifter": 22,
                "use_energy": True,
                "sample_frequency": 16000.0,
                "high_freq_cutoff": 7600.0,
                "low_freq_cutoff": 20.0,
                "use_log_fbank": True,
                "use_power": True,
                "window_type": "povey",
                "dither": 0.0,
                "remove_dc_offset": True,
                "preemphasis_coefficient": 0.97,
                "raw_energy": True,
                "energy_floor": 0.0,
                "epsilon": np.finfo(np.float32).eps,
            }
        }

    def test_ConvertTFLite(self):

        cfg = self.defaultCfg()
        input_size = 16000 * 3

        # Creating MFCC extraction model.
        mdl = Sequential([
            Input((input_size, )),
            Framing(**cfg["framing"]),
            MFCC(**cfg["mfcc"]),
        ])

        # Saving model and converting to TF Lite.
        with TemporaryDirectory() as mdlPath, \
                NamedTemporaryFile(suffix='.tflite') as tflitePath:
            mdl.save(mdlPath)
            SavedModel2TFLite(mdlPath, tflitePath.name)

    def test_FilterBank(self):

        # Default config overrides
        testNames = [f"16000_{i:03d}" for i in range(1, 49)]

        for name in testNames:

            overrides = RefMFCC.getConfig(name)
            samples = RefMFCC.getInputs(name)
            want = RefMFCC.getOutputs(name)

            with self.subTest(name=name, overrides=overrides):
                cfg = self.defaultCfg()
                cfg["snip_edges"] = overrides["snip_edges"]
                cfg["framing"].update(overrides["framing"])
                cfg["mfcc"].update(overrides["mfcc"])

                frameSize = cfg["framing"]["frame_length_ms"]
                frameShift = cfg["framing"]["frame_shift_ms"]
                sampleFreq = cfg["framing"]["sample_frequency"]

                # Mirror padding input samples.
                if not cfg["snip_edges"]:
                    m = int(frameSize / 1000.0 * sampleFreq)
                    k = int(frameShift / 1000.0 * sampleFreq)
                    samples = PadWaveform(samples, m, k)

                # Creating MFCC extraction model.
                mdl = Sequential([
                    Input((samples.shape[-1], )),
                    Framing(**cfg["framing"]),
                    MFCC(**cfg["mfcc"]),
                ])

                got = mdl(samples).numpy()

                # np.set_printoptions(threshold=13, linewidth=120, suppress=True)
                # print(f"got = \n{got[0]}")
                # print(f"want = \n{want[0]}")
                
                # print(f"got.shape = {got.shape}")
                # print(f"want.shape = {want.shape}")

                self.compareFeats(want, got)


if __name__ == "__main__":
    unittest.main()
