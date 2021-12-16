#!/usr/bin/env python3

import os
import numpy as np
import librosa

from lib.testdata import KaldiTestDataReader


class RefMFCC(KaldiTestDataReader):
    basePath = os.path.dirname(__file__)

    @classmethod
    def getInputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "audio.wav")
        samples, _ = librosa.load(inputFile, sr=None)
        samples = (samples * 32768).astype(np.int16).astype(np.float32)
        return samples.reshape(1, -1)

    @classmethod
    def getOutputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "mfcc.ark.txt")
        return np.stack(list(cls.loadKaldiArk(inputFile).values()), axis=0)

    @classmethod
    def getConfig(cls, testName):
        cfgFile = os.path.join(cls.basePath, "src", testName, "mfcc.conf")

        cfg = {"snip_edges": False, "framing": {}, "mfcc": {}}
        with open(cfgFile, 'r') as f:
            for line in f:
                line = line.strip()
                key, val = line.split("=")

                if key == "--sample-frequency":
                    cfg["framing"]["sample_frequency"] = float(val)
                    cfg["mfcc"]["sample_frequency"] = float(val)
                elif key == "--frame-length":
                    cfg["framing"]["frame_length_ms"] = float(val)
                elif key == "--frame-shift":
                    cfg["framing"]["frame_shift_ms"] = float(val)
                elif key == "--raw-energy":
                    cfg["mfcc"]["raw_energy"] = True if val == "true" else False
                elif key == "--dither":
                    cfg["mfcc"]["dither"] = float(val)
                elif key == "--low-freq":
                    cfg["mfcc"]["low_freq_cutoff"] = float(val)
                elif key == "--high-freq":
                    cfg["mfcc"]["high_freq_cutoff"] = float(val)
                elif key == "--num-mel-bins":
                    cfg["mfcc"]["num_mels"] = int(val)
                elif key == "--num-ceps":
                    cfg["mfcc"]["num_mfccs"] = int(val)
                elif key == "--snip-edges":
                    cfg["snip_edges"] = True if val == "true" else False
                else:
                    raise ValueError(f"unrecognized config '{line}' in test conf {cfgFile}")

        return cfg


class RefFbank(KaldiTestDataReader):
    basePath = os.path.dirname(__file__)

    @classmethod
    def getInputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "audio.wav")
        samples, _ = librosa.load(inputFile, sr=None)
        samples = (samples * 32768).astype(np.int16).astype(np.float32)
        return samples.reshape(1, -1)

    @classmethod
    def getOutputs(cls, testName):
        inputFile = os.path.join(cls.basePath, "src", testName, "fbank.ark.txt")
        return np.stack(list(cls.loadKaldiArk(inputFile).values()), axis=0)

    @classmethod
    def getConfig(cls, testName):
        cfgFile = os.path.join(cls.basePath, "src", testName, "fbank.conf")

        cfg = {"snip_edges": False, "framing": {}, "windowing": {}, "fbank": {}}
        with open(cfgFile, 'r') as f:
            for line in f:
                line = line.strip()
                key, val = line.split("=")

                if key == "--sample-frequency":
                    cfg["framing"]["sample_frequency"] = float(val)
                elif key == "--frame-length":
                    cfg["framing"]["frame_length_ms"] = float(val)
                elif key == "--frame-shift":
                    cfg["framing"]["frame_shift_ms"] = float(val)
                elif key == "--raw-energy":
                    cfg["windowing"]["raw_energy"] = True if val == "true" else False
                elif key == "--dither":
                    cfg["windowing"]["dither"] = float(val)
                elif key == "--low-freq":
                    cfg["fbank"]["low_freq_cutoff"] = float(val)
                elif key == "--high-freq":
                    cfg["fbank"]["high_freq_cutoff"] = float(val)
                elif key == "--num-mel-bins":
                    cfg["fbank"]["num_bins"] = int(val)
                elif key == "--use-log-fbank":
                    cfg["fbank"]["use_log_fbank"] = True if val == "true" else False
                elif key == "--use-power":
                    cfg["fbank"]["use_power"] = True if val == "true" else False
                elif key == "--snip-edges":
                    cfg["snip_edges"] = True if val == "true" else False
                else:
                    raise ValueError(f"unrecognized config '{line}' in test conf {cfgFile}")

        return cfg
